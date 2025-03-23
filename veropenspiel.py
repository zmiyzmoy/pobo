import os
import time
import logging
import sys
import signal
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp
from typing import List, Tuple, Dict, Optional
from open_spiel.python import policy
import pyspiel
import ray
from ray.util.queue import Queue
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import shutil
from sklearn.cluster import KMeans
import joblib
import copy

# ========== КОНФИГУРАЦИЯ ==========
class Config:
    MODEL_PATH = '/home/user/poker/models/psro_model.pt'
    BEST_MODEL_PATH = '/home/user/poker/models/psro_best.pt'
    LOG_DIR = '/home/user/poker/logs/'
    KMEANS_PATH = '/home/user/poker/models/kmeans.joblib'
    NUM_EPISODES = 10000
    BATCH_SIZE = 2048
    GAMMA = 0.96
    BUFFER_CAPACITY = 1_000_000
    NUM_WORKERS = 8
    STEPS_PER_WORKER = 2000
    SELFPLAY_UPDATE_FREQ = 1000
    LOG_FREQ = 25
    TEST_INTERVAL = 3000
    LEARNING_RATE = 1e-4
    NUM_PLAYERS = 6
    GRAD_CLIP_VALUE = 5.0
    CHECKPOINT_INTERVAL = 900
    NUM_BUCKETS = 50
    GAME_NAME = "universal_poker(betting=nolimit,numPlayers=6,numRounds=4,blind=0.05 0.10,raiseSize=0.10 0.20 0.40 0.80,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1)"

config = Config()
os.makedirs(config.LOG_DIR, exist_ok=True)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(os.path.join(config.LOG_DIR, 'training.log')), logging.StreamHandler(sys.stdout)])
writer = SummaryWriter(log_dir=config.LOG_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

game = pyspiel.load_game(config.GAME_NAME)

# ========== НОРМАЛИЗАЦИЯ НАГРАД ==========
class RewardNormalizer:
    def __init__(self, eps: float = 1e-8, max_size: int = 10000):
        self.rewards = []
        self.eps = eps
        self.max_size = max_size
        self.mean = 0
        self.std = 1

    def update(self, reward: float):
        self.rewards.append(reward)
        if len(self.rewards) > self.max_size:
            self.rewards.pop(0)
        self.mean = np.mean(self.rewards) if self.rewards else 0
        self.std = np.std(self.rewards) + self.eps if self.rewards else 1

    def normalize(self, reward: float) -> float:
        return (reward - self.mean) / self.std

# ========== МОДЕЛИ ==========
class RegretNet(nn.Module):
    def __init__(self, input_size: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        nn.init.kaiming_normal_(self.net[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.net[2].weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        with autocast():
            return self.net(x)

class StrategyNet(nn.Module):
    def __init__(self, input_size: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        nn.init.kaiming_normal_(self.net[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.net[2].weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        with autocast():
            return self.net(x)

# ========== БУФЕР ОПЫТА ==========
class ReplayBuffer:
    def __init__(self, capacity: int = config.BUFFER_CAPACITY):
        self.capacity = capacity
        self.state_size = config.NUM_BUCKETS + config.NUM_PLAYERS * 2 + 4 + 5
        self.states = np.zeros((capacity, self.state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, self.state_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.player_ids = np.zeros(capacity, dtype=np.int32)
        self.bets = np.zeros((capacity, config.NUM_PLAYERS), dtype=np.float32)
        self.stacks = np.zeros((capacity, config.NUM_PLAYERS), dtype=np.float32)
        self.stages = np.zeros((capacity, 4), dtype=np.float32)
        self.size = 0
        self.pos = 0

    def add_batch(self, experiences: List[Tuple], processor: 'StateProcessor'):
        batch_size = len(experiences)
        if self.pos + batch_size > self.capacity:
            self.pos = 0
        idx = slice(self.pos, self.pos + batch_size)
        
        states, actions, rewards, next_states, dones, player_ids, bets, stacks, stages = zip(*experiences)
        self.states[idx] = processor.process(states, player_ids, bets, stacks, stages)
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_states[idx] = processor.process(next_states, player_ids, bets, stacks, stages)
        self.dones[idx] = dones
        self.player_ids[idx] = player_ids
        self.bets[idx] = bets
        self.stacks[idx] = stacks
        self.stages[idx] = stages
        
        self.size = min(self.size + batch_size, self.capacity)
        self.pos = (self.pos + batch_size) % self.capacity

    def sample(self, batch_size: int) -> List[Tuple]:
        indices = np.random.choice(self.size, min(batch_size, self.size), replace=False)
        return [(self.states[i], self.actions[i], self.rewards[i], self.next_states[i], self.dones[i], 
                 self.player_ids[i], self.bets[i], self.stacks[i], self.stages[i]) for i in indices]

    def __len__(self) -> int:
        return self.size

# ========== EMBEDDING КАРТ ==========
class CardEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank_embed = nn.Embedding(13, 8)
        self.suit_embed = nn.Embedding(4, 4)

    def forward(self, cards: List[int]) -> torch.Tensor:
        ranks = torch.tensor([c % 13 for c in cards], dtype=torch.long).to(device)
        suits = torch.tensor([c // 13 for c in cards], dtype=torch.long).to(device)
        return torch.cat([self.rank_embed(ranks).mean(dim=0), self.suit_embed(suits).mean(dim=0)])

# ========== СТАТИСТИКА ОППОНЕНТОВ ==========
class OpponentStats:
    def __init__(self):
        self.stats = {i: {'vpip': 0, 'pfr': 0, 'af': 0, 'hands': 0, 'folds': 0, 'calls': 0, 'raises': 0, 'last_bet': 0} 
                      for i in range(config.NUM_PLAYERS)}

    def update(self, player_id: int, action: int, stage: List[int], pot: float, bet_size: float = 0, is_blind: bool = False):
        stage_idx = np.argmax(stage)
        self.stats[player_id]['hands'] += 1
        if stage_idx == 0 and not is_blind:
            if action > 0:
                self.stats[player_id]['vpip'] += 1
            if action > 1:
                self.stats[player_id]['pfr'] += 1
        if action == 0:
            self.stats[player_id]['folds'] += 1
        elif action == 1:
            self.stats[player_id]['calls'] += 1
        elif action > 1:
            self.stats[player_id]['raises'] += bet_size
            self.stats[player_id]['last_bet'] = bet_size
        total_actions = self.stats[player_id]['folds'] + self.stats[player_id]['calls'] + self.stats[player_id]['raises']
        if total_actions > 0:
            self.stats[player_id]['af'] = self.stats[player_id]['raises'] / total_actions

    def get_metrics(self, player_id: int) -> Dict[str, float]:
        hands = max(self.stats[player_id]['hands'], 1)
        return {
            'vpip': self.stats[player_id]['vpip'] / hands,
            'pfr': self.stats[player_id]['pfr'] / hands,
            'af': self.stats[player_id]['af'],
            'fold_freq': self.stats[player_id]['folds'] / hands,
            'last_bet': self.stats[player_id]['last_bet']
        }

# ========== ОБРАБОТКА СОСТОЯНИЙ ==========
class StateProcessor:
    def __init__(self):
        self.card_embedding = CardEmbedding().to(device)
        self.buckets = self._load_or_precompute_buckets()
        self.state_size = config.NUM_BUCKETS + config.NUM_PLAYERS * 2 + 4 + 5

    def _load_or_precompute_buckets(self):
        if os.path.exists(config.KMEANS_PATH):
            return joblib.load(config.KMEANS_PATH)
        all_hands = []
        for r1 in range(13):
            for r2 in range(r1, 13):
                for suited in [0, 1]:
                    hand = [r1 + s1 * 13 for s1 in range(4)] + [r2 + s2 * 13 for s2 in range(4)]
                    all_hands.append(self.card_embedding(hand[:2]).cpu().detach().numpy())
        kmeans = KMeans(n_clusters=config.NUM_BUCKETS, random_state=42).fit(all_hands)
        joblib.dump(kmeans, config.KMEANS_PATH)
        return kmeans

    def augment_cards(self, cards: List[int]) -> List[int]:
        """Аугментация с сохранением структуры руки"""
        suit_mapping = {s: random.randint(0, 3) for s in set(c // 13 for c in cards)}
        return [c % 13 + suit_mapping[c // 13] * 13 for c in cards]

    def process(self, states: List, player_ids: List[int], bets: List[List[float]], stacks: List[List[float]], stages: List[List[int]], opponent_stats: Optional[OpponentStats] = None) -> np.ndarray:
        batch_size = len(states)
        assert len(player_ids) == batch_size, f"Player IDs mismatch: {len(player_ids)} vs {batch_size}"

        info_states = [s.information_state_tensor(pid) for s, pid in zip(states, player_ids)]
        cards_batch = [[int(c) for i, c in enumerate(info[:52]) if c >= 0 and i in s.player_cards(pid)] 
                       for info, s, pid in zip(info_states, states, player_ids)]
        if random.random() < 0.5:
            cards_batch = [self.augment_cards(cards) for cards in cards_batch]
        card_embs = torch.stack([self.card_embedding(cards) for cards in cards_batch]).cpu().detach().numpy()
        assert card_embs.shape == (batch_size, 12), f"Card embedding shape mismatch: {card_embs.shape}"
        bucket_idxs = self.buckets.predict(card_embs)
        bucket_one_hot = np.zeros((batch_size, config.NUM_BUCKETS))
        bucket_one_hot[np.arange(batch_size), bucket_idxs] = 1.0

        assert len(bets) == batch_size, f"Bets batch size mismatch: {len(bets)}"
        assert len(stacks) == batch_size, f"Stacks batch size mismatch: {len(stacks)}"
        assert len(stages) == batch_size, f"Stages batch size mismatch: {len(stages)}"
        for b, st, sg in zip(bets, stacks, stages):
            assert len(b) == config.NUM_PLAYERS, f"Invalid bets length: {len(b)}"
            assert len(st) == config.NUM_PLAYERS, f"Invalid stacks length: {len(st)}"
            assert len(sg) == 4 and sum(sg) == 1, f"Invalid stage: {sg}"

        pots = [sum(b) for b in bets]
        sprs = [stk[pid] / pot if pot > 0 else 10.0 for stk, pid, pot in zip(stacks, player_ids, pots)]
        positions = [(pid - s.current_player()) % config.NUM_PLAYERS / config.NUM_PLAYERS for s, pid in zip(states, player_ids)]

        opponent_metrics = [[opponent_stats.get_metrics(i) for i in range(config.NUM_PLAYERS) if i != pid] 
                           if opponent_stats else [] for pid in player_ids]
        table_aggs = [np.mean([m['af'] for m in metrics]) if metrics else 0.5 for metrics in opponent_metrics]
        last_bets = [max([m['last_bet'] for m in metrics]) / pot if pot > 0 and metrics else 0.0 
                     for metrics, pot in zip(opponent_metrics, pots)]
        all_in_flags = [1.0 if any(b >= stk[i] for i, b in enumerate(bet) if i != pid) else 0.0 
                        for bet, stk, pid in zip(bets, stacks, player_ids)]

        processed = np.concatenate([
            bucket_one_hot,
            np.array(bets) / 1000.0,
            np.array(stacks) / 1000.0,
            np.array(stages),
            np.array([sprs, table_aggs, positions, last_bets, all_in_flags]).T
        ], axis=1)
        assert processed.shape == (batch_size, self.state_size), f"Processed shape mismatch: {processed.shape}"
        return processed

# ========== АГЕНТ ==========
class PokerAgent(policy.Policy):
    def __init__(self, game, processor: StateProcessor):
        super().__init__(game)
        self.game = game
        self.processor = processor
        self.num_actions = game.num_distinct_actions()
        self.regret_net = RegretNet(processor.state_size, self.num_actions).to(device)
        self.strategy_net = StrategyNet(processor.state_size, self.num_actions).to(device)
        self.cumulative_regrets = {}
        self.cumulative_strategies = {}
        self.strategy_pool = []

    def _dynamic_bet_size(self, state, stacks: List[float], bets: List[float], position: int, opponent_metrics: Dict[str, float], q_value: float, stage: List[int], last_opp_bet: float) -> float:
        pot = sum(bets)
        my_stack = stacks[position]
        sizes = [0.33 * pot, 0.66 * pot, pot, 2 * pot, my_stack]
        idx = min(int(q_value * (len(sizes) - 1)), len(sizes) - 1)
        base = sizes[idx]

        if last_opp_bet > 0:
            opp_bet_ratio = last_opp_bet / pot
            base = max(base, last_opp_bet * 1.1) if opp_bet_ratio < 1.0 else my_stack

        if opponent_metrics.get('af', 1.0) < 0.3:
            base *= 1.5
        if opponent_metrics.get('vpip', 0.5) > 0.5:
            base *= 1.2
        
.VerticalAlignment        # Точный минимальный рейз: удвоение последней разницы ставок
        current_bet = bets[position]
        max_bet = max(bets)
        min_raise = max_bet + (max_bet - max([b for i, b in enumerate(bets) if i != position and b < max_bet] + [0])) if max_bet > current_bet else pot
        base = max(min_raise, min(base, my_stack))
        return base

    def _heuristic_reward(self, action: int, state, player_id: int, bets: List[float], opponent_stats: OpponentStats) -> float:
        pot = sum(bets)
        opp_metrics = [opponent_stats.get_metrics(i) for i in range(config.NUM_PLAYERS) if i != player_id]
        if action == 0 and pot > 0:
            return -0.1
        elif action > 1 and any(m['af'] < 0.3 for m in opp_metrics):
            return 0.2
        elif action == 1 and any(m['last_bet'] > pot * 0.5 for m in opp_metrics):
            return 0.1 if random.random() < 0.5 else -0.1
        return 0.0

    def action_probabilities(self, state, player_id: Optional[int] = None) -> Dict[int, float]:
        info_state = state.information_state_string(player_id)
        legal_actions = state.legal_actions(player_id)
        assert all(0 <= a < self.num_actions for a in legal_actions), f"Illegal actions: {legal_actions}"
        if not legal_actions:
            return {0: 1.0}

        bets = state.bets() if hasattr(state, 'bets') else [0] * config.NUM_PLAYERS
        stacks = state.stacks() if hasattr(state, 'stacks') else [1000] * config.NUM_PLAYERS
        stage = [1 if state.round() == i else 0 for i in range(4)]
        opponent_stats = OpponentStats()
        state_tensor = torch.FloatTensor(self.processor.process([state], [player_id], [bets], [stacks], [stage], opponent_stats)).to(device)

        with torch.no_grad():
            regrets = self.regret_net(state_tensor)[0]
            strategy_logits = self.strategy_net(state_tensor)[0]
            legal_mask = torch.zeros(self.num_actions, device=device)
            legal_mask[legal_actions] = 1
            strategy_logits = strategy_logits.masked_fill(legal_mask == 0, -1e9)
            strategy = torch.softmax(strategy_logits, dim=0).cpu().numpy()

        if info_state not in self.cumulative_regrets:
            self.cumulative_regrets[info_state] = np.zeros(self.num_actions)
            self.cumulative_strategies[info_state] = np.zeros(self.num_actions)

        self.cumulative_regrets[info_state][legal_actions] += regrets[legal_actions].cpu().numpy()
        positive_regrets = np.maximum(self.cumulative_regrets[info_state][legal_actions], 0)
        total_regret = positive_regrets.sum()
        probs = positive_regrets / total_regret if total_regret > 0 else strategy[legal_actions] / strategy[legal_actions].sum()
        self.cumulative_strategies[info_state][legal_actions] += probs

        if len(self.strategy_pool) < 5 and random.random() < 0.1:
            self.strategy_pool.append(copy.deepcopy(self.strategy_net.state_dict()))

        return {a: float(p) for a, p in zip(legal_actions, probs)}

    def step(self, state, player_id: int, bets: List[float], stacks: List[float], stage: List[int], opponent_stats: OpponentStats) -> int:
        probs = self.action_probabilities(state, player_id)
        action = max(probs, key=probs.get)
        if action > 1:
            state_tensor = torch.FloatTensor(self.processor.process([state], [player_id], [bets], [stacks], [stage], opponent_stats)).to(device)
            q_value = self.strategy_net(state_tensor).max().item()
            last_opp_bet = max([opponent_stats.get_metrics(i)['last_bet'] for i in range(config.NUM_PLAYERS) if i != player_id], default=0)
            bet_size = self._dynamic_bet_size(state, stacks, bets, player_id, opponent_stats.get_metrics(player_id), q_value, stage, last_opp_bet)
            if bet_size >= stacks[player_id] and 4 in state.legal_actions(player_id):
                action = 4
            elif bet_size >= sum(bets) and 3 in state.legal_actions(player_id):
                action = 3
            elif 2 in state.legal_actions(player_id):
                action = 2
        return action

# ========== СБОР ДАННЫХ ==========
@ray.remote
def collect_experience(game, agent: PokerAgent, processor: StateProcessor, steps: int, worker_id: int, queue: Queue):
    try:
        start_time = time.time()
        env = game.new_initial_state()
        agents = [agent] + [PokerAgent(game, processor) for _ in range(config.NUM_PLAYERS - 1)]
        for i, opp in enumerate(agents[1:], 1):
            if agent.strategy_pool and random.random() < 0.5:
                opp.strategy_net.load_state_dict(random.choice(agent.strategy_pool))
        experiences = []
        opponent_stats = OpponentStats()

        for _ in range(steps):
            if env.is_terminal():
                env = game.new_initial_state()
            player_id = env.current_player()
            bets = env.bets() if hasattr(env, 'bets') else [0] * config.NUM_PLAYERS
            stacks = env.stacks() if hasattr(env, 'stacks') else [1000] * config.NUM_PLAYERS
            stage = [1 if env.round() == i else 0 for i in range(4)]
            is_blind = player_id in [0, 1] and env.round() == 0 and sum(bets) <= 0.15

            action = agents[player_id].step(env, player_id, bets, stacks, stage, opponent_stats)
            next_state = env.clone()
            next_state.apply_action(action)

            final_reward = next_state.returns()[player_id]
            heuristic_reward = agents[player_id]._heuristic_reward(action, env, player_id, bets, opponent_stats)
            reward = heuristic_reward if not next_state.is_terminal() else final_reward

            experiences.append((env, action, reward, next_state, next_state.is_terminal(), player_id, bets, stacks, stage))
            opponent_stats.update(player_id, action, stage, sum(bets), action if action > 1 else 0, is_blind)
            env = next_state

        elapsed_time = time.time() - start_time
        hands_per_sec = steps / elapsed_time
        queue.put((experiences, opponent_stats, hands_per_sec))
        logging.info(f"Worker {worker_id} collected {len(experiences)} experiences at {hands_per_sec:.2f} hands/sec")
    except Exception as e:
        logging.error(f"Worker {worker_id} failed: {str(e)}")
        queue.put(([], OpponentStats(), 0))

# ========== ТЕСТИРОВАНИЕ ==========
def run_tournament(game, agent: PokerAgent, processor: StateProcessor, num_games: int, compute_exploitability: bool = True) -> Tuple[float, float]:
    from open_spiel.python.algorithms import exploitability
    total_reward = 0
    start_time = time.time()
    for _ in range(num_games):
        state = game.new_initial_state()
        opponents = [PokerAgent(game, processor) for _ in range(config.NUM_PLAYERS - 1)]
        for opp in opponents:
            if agent.strategy_pool:
                opp.strategy_net.load_state_dict(random.choice(agent.strategy_pool))
        while not state.is_terminal():
            player_id = state.current_player()
            bets = state.bets() if hasattr(state, 'bets') else [0] * config.NUM_PLAYERS
            stacks = state.stacks() if hasattr(state, 'stacks') else [1000] * config.NUM_PLAYERS
            stage = [1 if state.round() == i else 0 for i in range(4)]
            opponent_stats = OpponentStats()

            if player_id == 0:
                action = agent.step(state, 0, bets, stacks, stage, opponent_stats)
            else:
                action = opponents[player_id - 1].step(state, player_id, bets, stacks, stage, opponent_stats)
            state.apply_action(action)
        total_reward += state.returns()[0]
    
    winrate = total_reward / num_games
    exp_score = exploitability.exploitability(game, agent) if compute_exploitability else 0.0
    logging.info(f"Tournament took {time.time() - start_time:.2f} seconds for {num_games} games")
    return winrate, exp_score

# ========== ОБУЧЕНИЕ ==========
class Trainer:
    def __init__(self):
        self.game = game
        self.processor = StateProcessor()
        self.agent = PokerAgent(self.game, self.processor)
        self.regret_optimizer = optim.Adam(self.agent.regret_net.parameters(), lr=config.LEARNING_RATE)
        self.strategy_optimizer = optim.Adam(self.agent.strategy_net.parameters(), lr=config.LEARNING_RATE)
        self.buffer = ReplayBuffer()
        self.reward_normalizer = RewardNormalizer()
        self.opponent_stats = OpponentStats()
        self.global_step = 0
        self.best_winrate = -float('inf')
        self.interrupted = False
        self.last_checkpoint_time = time.time()

    def _train_step(self, experiences):
        states, actions, rewards, next_states, dones, player_ids, bets, stacks, stages = zip(*experiences)
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor([self.reward_normalizer.normalize(r) for r in rewards]).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        with autocast():
            regrets = self.agent.regret_net(states)
            strategies = self.agent.strategy_net(states)
            next_strategies = self.agent.strategy_net(next_states)
            expected_future_rewards = (torch.softmax(next_strategies, dim=1).max(dim=1)[0] * config.GAMMA * (1 - dones)).detach()
            target_rewards = rewards + expected_future_rewards

            regret_loss = nn.MSELoss()(regrets.gather(1, actions.unsqueeze(1)), target_rewards.unsqueeze(1))
            strategy_loss = -torch.mean(torch.log_softmax(strategies, dim=1) * regrets.detach())

        self.regret_optimizer.zero_grad()
        scaler.scale(regret_loss).backward()
        scaler.unscale_(self.regret_optimizer)
        torch.nn.utils.clip_grad_norm_(self.agent.regret_net.parameters(), config.GRAD_CLIP_VALUE)
        scaler.step(self.regret_optimizer)
        scaler.update()

        self.strategy_optimizer.zero_grad()
        scaler.scale(strategy_loss).backward()
        scaler.unscale_(self.strategy_optimizer)
        torch.nn.utils.clip_grad_norm_(self.agent.strategy_net.parameters(), config.GRAD_CLIP_VALUE)
        scaler.step(self.strategy_optimizer)
        scaler.update()

        return regret_loss.item() + strategy_loss.item()

    def _save_checkpoint(self, path: str, is_best: bool = False):
        checkpoint = {
            'regret_net': self.agent.regret_net.state_dict(),
            'strategy_net': self.agent.strategy_net.state_dict(),
            'regret_optimizer': self.regret_optimizer.state_dict(),
            'strategy_optimizer': self.strategy_optimizer.state_dict(),
            'step': self.global_step,
            'reward_history': self.reward_normalizer.rewards,
            'strategy_pool': self.agent.strategy_pool
        }
        torch.save(checkpoint, path)
        if is_best:
            shutil.copyfile(path, config.BEST_MODEL_PATH)
        logging.info(f"Checkpoint saved to {path}, is_best={is_best}")
        self.last_checkpoint_time = time.time()

    def _load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=device)
        self.agent.regret_net.load_state_dict(checkpoint['regret_net'])
        self.agent.strategy_net.load_state_dict(checkpoint['strategy_net'])
        self.regret_optimizer.load_state_dict(checkpoint['regret_optimizer'])
        self.strategy_optimizer.load_state_dict(checkpoint['strategy_optimizer'])
        self.global_step = checkpoint['step']
        self.reward_normalizer.rewards = checkpoint['reward_history']
        self.reward_normalizer.mean = np.mean(self.reward_normalizer.rewards) if self.reward_normalizer.rewards else 0
        self.reward_normalizer.std = np.std(self.reward_normalizer.rewards) + self.reward_normalizer.eps if self.reward_normalizer.rewards else 1
        logging.info(f"Checkpoint loaded from {path}, step {self.global_step}")

    def train(self):
        ray.init(num_gpus=1, num_cpus=config.NUM_WORKERS)
        pbar = tqdm(total=config.NUM_EPISODES, desc="Training")
        agent_stats = OpponentStats()

        def signal_handler(sig, frame):
            self.interrupted = True
            self._save_checkpoint(config.MODEL_PATH)
            ray.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        if os.path.exists(config.MODEL_PATH):
            self._load_checkpoint(config.MODEL_PATH)

        for episode in range(self.global_step // config.STEPS_PER_WORKER, config.NUM_EPISODES):
            queues = [Queue() for _ in range(config.NUM_WORKERS)]
            tasks = [collect_experience.remote(self.game, self.agent, self.processor, config.STEPS_PER_WORKER, i, queues[i])
                     for i in range(config.NUM_WORKERS)]
            results = [q.get() for q in queues]

            for experiences, local_opp_stats, hands_per_sec in results:
                if not experiences:
                    continue
                self.buffer.add_batch(experiences, self.processor)
                for exp in experiences:
                    self.reward_normalizer.update(exp[2])
                    self.global_step += 1
                    if exp[5] == 0:
                        is_blind = exp[5] in [0, 1] and np.argmax(exp[8]) == 0 and sum(exp[6]) <= 0.15
                        agent_stats.update(0, exp[1], exp[8], sum(exp[6]), exp[1] if exp[1] > 1 else 0, is_blind)
                self.opponent_stats.stats.update(local_opp_stats.stats)
                writer.add_scalar('HandsPerSec', hands_per_sec, self.global_step)

            if len(self.buffer) >= config.BATCH_SIZE:
                loss = self._train_step(self.buffer.sample(config.BATCH_SIZE))
                writer.add_scalar('Loss', loss, self.global_step)

            if self.global_step % config.SELFPLAY_UPDATE_FREQ == 0:
                logging.info(f"Self-play update at step {self.global_step}")

            if self.global_step % config.LOG_FREQ == 0:
                logging.info(f"Step {self.global_step} | Loss: {loss:.4f}")

            if self.global_step % config.TEST_INTERVAL == 0:
                winrate, exp_score = run_tournament(self.game, self.agent, self.processor, 100)
                agent_metrics = agent_stats.get_metrics(0)
                writer.add_scalar('Winrate', winrate, self.global_step)
                writer.add_scalar('Exploitability', exp_score, self.global_step)
                writer.add_scalar('Agent_VPIP', agent_metrics['vpip'], self.global_step)
                writer.add_scalar('Agent_PFR', agent_metrics['pfr'], self.global_step)
                if winrate > self.best_winrate:
                    self.best_winrate = winrate
                    self._save_checkpoint(config.BEST_MODEL_PATH, is_best=True)
                    logging.info(f"New best model saved with winrate {winrate:.4f}, exploitability {exp_score:.4f} m-a")

            if time.time() - self.last_checkpoint_time >= config.CHECKPOINT_INTERVAL:
                self._save_checkpoint(config.MODEL_PATH)

            pbar.update(1)
            if self.interrupted:
                break

        self._save_checkpoint(config.MODEL_PATH)
        ray.shutdown()
        pbar.close()
        writer.close()

# ========== ЗАПУСК ==========
if __name__ == "__main__":
    mp.set_start_method('spawn')
    trainer = Trainer()
    try:
        trainer.train()
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        trainer._save_checkpoint('error_state.pt')
