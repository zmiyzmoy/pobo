#t4_minimal.py -1
import os
import time
import logging
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import pyspiel
import ray
from tqdm.auto import tqdm
import joblib
import argparse
from sklearn.cluster import KMeans
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional
from open_spiel.python import policy

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,  # Измените с INFO на DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_output.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Конфигурация
class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Poker RL Training")
        parser.add_argument('--base_dir', type=str, default='./poker', help='Base directory for models and logs')
        args = parser.parse_args()

        self.BASE_DIR = args.base_dir
        self.MODEL_PATH = os.path.join(self.BASE_DIR, 'models', 'psro_model.pt')
        self.BEST_MODEL_PATH = os.path.join(self.BASE_DIR, 'models', 'psro_best.pt')
        self.LOG_DIR = os.path.join(self.BASE_DIR, 'logs')
        self.KMEANS_PATH = os.path.join(self.BASE_DIR, 'models', 'kmeans.joblib')
        self.NUM_EPISODES = 1  # Для теста
        self.BATCH_SIZE = 64
        self.GAMMA = 0.96
        self.BUFFER_CAPACITY = 2000
        self.NUM_WORKERS = 4  # Увеличиваем до 4 воркеров
        self.STEPS_PER_WORKER = 500
        self.SELFPLAY_UPDATE_FREQ = 50
        self.LOG_FREQ = 10
        self.TEST_INTERVAL = 50
        self.LEARNING_RATE = 1e-4
        self.NUM_PLAYERS = 6
        self.GRAD_CLIP_VALUE = 5.0
        self.CHECKPOINT_INTERVAL = 30
        self.NUM_BUCKETS = 50
        self.BB = 2
        self.MAX_STRATEGY_POOL = 10
        self.MAX_DICT_SIZE = 10000
        self.REWARD_NORMALIZATION = 'stack'
        self.GAME_NAME = "universal_poker(betting=nolimit,numPlayers=6,numRounds=4,blind=1 2 3 4 5 6,raiseSize=0.10 0.20 0.40 0.80,stack=100 100 100 100 100 100,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1)"

config = Config()
os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
os.makedirs(config.LOG_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
game = pyspiel.load_game(config.GAME_NAME)
logging.info(f"Конфигурация инициализирована, устройство: {device}, CUDA: {torch.cuda.is_available()}")

# Приоритетный буфер опыта
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = config.BUFFER_CAPACITY, alpha: float = 0.6):
        self.buffer = deque(maxlen=capacity)
        self.alpha = alpha
        self.priorities = deque(maxlen=capacity)
        self._max_priority = 1.0
        logging.info("PrioritizedReplayBuffer инициализирован")

    def add_batch(self, experiences):
        self.buffer.extend(experiences)
        self.priorities.extend([self._max_priority] * len(experiences))
        logging.debug(f"Добавлено {len(experiences)} опытов в буфер")

    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buffer) == 0:
            return [], [], np.array([])
        probs = np.array(self.priorities) ** self.alpha
        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = np.ones(len(self.buffer)) / len(self.buffer)
        else:
            probs /= probs_sum
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self._max_priority = max(self._max_priority, priority)

    def __len__(self):
        return len(self.buffer)

# Модели нейронных сетей
class RegretNet(nn.Module):
    def __init__(self, input_size: int, num_actions: int):
        super().__init__()
        assert input_size > 0, f"Invalid input_size: {input_size}"
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_actions)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        logging.info("RegretNet инициализирована")

    def forward(self, x):
        return self.net(x)

class StrategyNet(nn.Module):
    def __init__(self, input_size: int, num_actions: int):
        super().__init__()
        assert input_size > 0, f"Invalid input_size: {input_size}"
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_actions)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        logging.info("StrategyNet инициализирована")

    def forward(self, x):
        return self.net(x)

# Встраивание карт
class CardEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank_embed = nn.Embedding(13, 8).to(device)
        self.suit_embed = nn.Embedding(4, 4).to(device)
        # Установка float32 для весов
        self.rank_embed.weight.data = self.rank_embed.weight.data.to(dtype=torch.float32)
        self.suit_embed.weight.data = self.suit_embed.weight.data.to(dtype=torch.float32)
        logging.info("CardEmbedding инициализирован")

    def forward(self, cards: List[int]) -> torch.Tensor:
        ranks = torch.tensor([c % 13 for c in cards], dtype=torch.long, device=device)
        suits = torch.tensor([c // 13 for c in cards], dtype=torch.long, device=device)
        rank_emb = self.rank_embed(ranks).mean(dim=0).to(dtype=torch.float32)
        suit_emb = self.suit_embed(suits).mean(dim=0).to(dtype=torch.float32)
        return torch.cat([rank_emb, suit_emb]).to(dtype=torch.float32)

# Статистика оппонентов
class OpponentStats:
    def __init__(self):
        self.stats = {i: {
            'vpip': 0, 'pfr': 0, 'af': 0, 'hands': 0, 'folds': 0, 'calls': 0, 'raises': 0, 'last_bet': 0,
            'fold_to_cbet': 0, 'cbet_opp': 0, 'fold_to_3bet': 0, '3bet_opp': 0, 'check_raise': 0, 'check_opp': 0,
            'pos_winrate': {pos: {'wins': 0, 'hands': 0} for pos in range(config.NUM_PLAYERS)},
            'call_vs_raise_freq': 0, 'raise_opp': 0, 'street_aggression': [0] * 4
        } for i in range(config.NUM_PLAYERS)}

    def update(self, player_id: int, action: int, stage: List[int], pot: float, bet_size: float, is_blind: bool, position: int, 
               is_cbet: bool = False, is_3bet: bool = False, is_check: bool = False, won: float = 0, is_raise: bool = False):
        stage_idx = np.argmax(stage)
        self.stats[player_id]['hands'] += 1
        self.stats[player_id]['pos_winrate'][position]['hands'] += 1
        if won > 0:
            self.stats[player_id]['pos_winrate'][position]['wins'] += 1
        if stage_idx == 0 and not is_blind:
            if action > 0:
                self.stats[player_id]['vpip'] += 1
            if action > 1:
                self.stats[player_id]['pfr'] += 1
        if action == 0:
            self.stats[player_id]['folds'] += 1
            if is_cbet:
                self.stats[player_id]['fold_to_cbet'] += 1
            if is_3bet:
                self.stats[player_id]['fold_to_3bet'] += 1
        elif action == 1:
            self.stats[player_id]['calls'] += 1
            if is_raise:
                self.stats[player_id]['call_vs_raise_freq'] += 1
        elif action > 1:
            self.stats[player_id]['raises'] += bet_size
            self.stats[player_id]['last_bet'] = bet_size
            self.stats[player_id]['street_aggression'][stage_idx] += 1
            if is_check:
                self.stats[player_id]['check_raise'] += 1
        if is_cbet:
            self.stats[player_id]['cbet_opp'] += 1
        if is_3bet:
            self.stats[player_id]['3bet_opp'] += 1
        if is_check:
            self.stats[player_id]['check_opp'] += 1
        if is_raise:
            self.stats[player_id]['raise_opp'] += 1
        total_actions = self.stats[player_id]['folds'] + self.stats[player_id]['calls'] + self.stats[player_id]['raises']
        if total_actions > 0:
            self.stats[player_id]['af'] = self.stats[player_id]['raises'] / total_actions

    def get_metrics(self, player_id: int) -> Dict[str, float]:
        hands = max(self.stats[player_id]['hands'], 1)
        cbet_opp = max(self.stats[player_id]['cbet_opp'], 1)
        threebet_opp = max(self.stats[player_id]['3bet_opp'], 1)
        check_opp = max(self.stats[player_id]['check_opp'], 1)
        raise_opp = max(self.stats[player_id]['raise_opp'], 1)
        pos_stats = {pos: self.stats[player_id]['pos_winrate'][pos]['wins'] / max(self.stats[player_id]['pos_winrate'][pos]['hands'], 1)
                     for pos in range(config.NUM_PLAYERS)}
        return {
            'vpip': float(self.stats[player_id]['vpip'] / hands),
            'pfr': float(self.stats[player_id]['pfr'] / hands),
            'af': float(self.stats[player_id]['af']),
            'fold_freq': float(self.stats[player_id]['folds'] / hands),
            'last_bet': float(self.stats[player_id]['last_bet']),
            'fold_to_cbet': float(self.stats[player_id]['fold_to_cbet'] / cbet_opp),
            'fold_to_3bet': float(self.stats[player_id]['fold_to_3bet'] / threebet_opp),
            'check_raise_freq': float(self.stats[player_id]['check_raise'] / check_opp),
            'call_vs_raise_freq': float(self.stats[player_id]['call_vs_raise_freq'] / raise_opp),
            'street_aggression': [float(agg / max(hands, 1)) for agg in self.stats[player_id]['street_aggression']],
            'pos_winrate': pos_stats
        }

# Обработка состояний
# Обработка состояний
class StateProcessor:
    def __init__(self):
        self.card_embedding = CardEmbedding()
        self.buckets = self._load_or_precompute_buckets()
        self.state_size = config.NUM_BUCKETS + config.NUM_PLAYERS * 3 + 4 + 5 + 5  # +5 для opp_features
        self.cache = {}
        self.max_cache_size = 1000
        logging.info(f"StateProcessor инициализирован, state_size={self.state_size}")

    def _load_or_precompute_buckets(self):
        if os.path.exists(config.KMEANS_PATH):
            buckets = joblib.load(config.KMEANS_PATH)
            logging.info(f"Загружены бакеты из {config.KMEANS_PATH}")
            return buckets
        all_hands = []
        for r1 in range(13):
            for r2 in range(r1, 13):
                for suited in [0, 1]:
                    hand = [r1 + suited * 13, r2 + suited * 13]
                    embedding = self.card_embedding(hand).cpu().detach().numpy().astype(np.float32)
                    all_hands.append(embedding)
        # Явное приведение всего списка к float32
        all_hands = np.array(all_hands, dtype=np.float32)
        kmeans = KMeans(n_clusters=config.NUM_BUCKETS, random_state=42).fit(all_hands)
        joblib.dump(kmeans, config.KMEANS_PATH)
        logging.info(f"Бакеты вычислены и сохранены в {config.KMEANS_PATH}")
        return kmeans

    def process(self, states: List, player_ids: List[int], bets: List[List[float]] = None, 
                stacks: List[List[float]] = None, stages: List[List[int]] = None, 
                opponent_stats: Optional[OpponentStats] = None) -> np.ndarray:
        batch_size = len(states)
        if bets is None:
            bets = [[0] * config.NUM_PLAYERS] * batch_size
        if stacks is None:
            stacks = [[100] * config.NUM_PLAYERS] * batch_size
        if stages is None:
            stages = [[0, 0, 0, 0]] * batch_size
        
        state_keys = [f"{s.information_state_string(pid)}_{pid}_{tuple(b)}_{tuple(stk)}" 
                      for s, pid, b, stk in zip(states, player_ids, bets, stacks)]
        cached = [self.cache.get(key) for key in state_keys]
        if all(c is not None for c in cached):
            return np.array(cached, dtype=np.float32)
        
        info_states = [s.information_state_tensor(pid) for s, pid in zip(states, player_ids)]
        cards_batch = []
        for info in info_states:
            private_cards = [int(i) for i, c in enumerate(info[:52]) if c > 0][:2]
            if not private_cards:
                private_cards = [0, 1]
            elif len(private_cards) < 2:
                private_cards.extend([1] * (2 - len(private_cards)))
            cards_batch.append(private_cards)
        
        card_embs = torch.stack([self.card_embedding(cards) for cards in cards_batch]).to(dtype=torch.float32)
        card_embs = card_embs.cpu().detach().numpy().astype(np.float32)
        # Диагностика
        logging.debug(f"card_embs shape: {card_embs.shape}, dtype: {card_embs.dtype}, sample: {card_embs[0][:5]}")
        if card_embs.dtype != np.float32:
            logging.error(f"card_embs имеет тип {card_embs.dtype}, ожидается float32")
            card_embs = card_embs.astype(np.float32)
        bucket_idxs = self.buckets.predict(card_embs)
        bucket_one_hot = np.zeros((batch_size, config.NUM_BUCKETS), dtype=np.float32)
        bucket_one_hot[np.arange(batch_size), bucket_idxs] = 1.0
        
        bets_norm = np.array(bets, dtype=np.float32) / (np.array(stacks, dtype=np.float32) + 1e-8)
        stacks_norm = np.array(stacks, dtype=np.float32) / 1000.0
        pots = np.array([sum(b) for b in bets], dtype=np.float32)
        sprs = np.array([stk[pid] / pot if pot > 0 else 10.0 for stk, pid, pot in zip(stacks, player_ids, pots)], dtype=np.float32)
        positions = np.array([(pid - s.current_player()) % config.NUM_PLAYERS / config.NUM_PLAYERS 
                             for s, pid in zip(states, player_ids)], dtype=np.float32)
        action_history = np.array([([0] * config.NUM_PLAYERS if not hasattr(s, 'action_history') else 
                                   [min(h, 4) for h in s.action_history()[-config.NUM_PLAYERS:]]) 
                                   for s in states], dtype=np.float32)
        
        opponent_metrics = [[opponent_stats.get_metrics(i) for i in range(config.NUM_PLAYERS) if i != pid] 
                            if opponent_stats else [] for pid in player_ids]
        opp_features = []
        for metrics in opponent_metrics:
            if metrics:
                agg_vpip = float(np.mean([float(m['vpip']) for m in metrics]))
                agg_pfr = float(np.mean([float(m['pfr']) for m in metrics]))
                agg_fold_to_cbet = float(np.mean([float(m['fold_to_cbet']) for m in metrics]))
                agg_fold_to_3bet = float(np.mean([float(m['fold_to_3bet']) for m in metrics]))
                agg_street_agg = float(np.mean([float(np.mean(m['street_aggression'])) for m in metrics]))
            else:
                agg_vpip = 0.5
                agg_pfr = 0.5
                agg_fold_to_cbet = 0.5
                agg_fold_to_3bet = 0.5
                agg_street_agg = 0.5
            opp_features.append([agg_vpip, agg_pfr, agg_fold_to_cbet, agg_fold_to_3bet, agg_street_agg])
        opp_features = np.array(opp_features, dtype=np.float32)
        
        processed = np.concatenate([
            bucket_one_hot, bets_norm, stacks_norm, action_history, np.array(stages, dtype=np.float32),
            np.array([sprs, positions, np.zeros(batch_size), np.zeros(batch_size), np.zeros(batch_size)], dtype=np.float32).T,
            opp_features
        ], axis=1).astype(np.float32)
        
        if np.any(np.isnan(processed)) or np.any(np.isinf(processed)):
            logging.error(f"NaN/Inf in processed: {processed}")
            raise ValueError("Invalid state processing detected")
        
        if len(self.cache) >= self.max_cache_size:
            self.cache.clear()
        for key, proc in zip(state_keys, processed):
            self.cache[key] = proc
        
        return processed

# Агент с поддержкой PSRO
class PokerAgent(policy.Policy):
    def __init__(self, game, processor):
        player_ids = list(range(game.num_players()))
        super().__init__(game, player_ids)
        self.game = game
        self.processor = processor
        self.num_actions = game.num_distinct_actions()
        self.regret_net = RegretNet(processor.state_size, self.num_actions).to(device)
        self.strategy_net = StrategyNet(processor.state_size, self.num_actions).to(device)
        self.optimizer = optim.Adam(
            list(self.regret_net.parameters()) + list(self.strategy_net.parameters()),
            lr=config.LEARNING_RATE
        )
        self.global_step = 0
        self.cumulative_regrets = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.float32))
        self.cumulative_strategies = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.float32))
        self.strategy_pool = []
        logging.info(f"PokerAgent инициализирован, state_size={processor.state_size}, num_actions={self.num_actions}")

    def action_probabilities(self, state, player_id: Optional[int] = None, opponent_stats: Optional[OpponentStats] = None) -> Dict[int, float]:
        legal_actions = state.legal_actions(player_id)
        if not legal_actions:
            return {0: 1.0}
        bets = state.bets() if hasattr(state, 'bets') else [0] * config.NUM_PLAYERS
        stacks = state.stacks() if hasattr(state, 'stacks') else [100] * config.NUM_PLAYERS
        info_tensor = state.information_state_tensor(player_id)
        board_cards = [int(c) for i, c in enumerate(info_tensor[52:]) if c >= 0]
        stage = [0] * 4
        if len(board_cards) == 0:
            stage[0] = 1
        elif len(board_cards) == 3:
            stage[1] = 1
        elif len(board_cards) == 4:
            stage[2] = 1
        elif len(board_cards) == 5:
            stage[3] = 1
        state_tensor = torch.tensor(self.processor.process([state], [player_id], [bets], [stacks], [stage], opponent_stats), 
                                  dtype=torch.float32, device=device)
        state_key = state.information_state_string(player_id)
        
        with torch.no_grad():
            self.strategy_net.eval()
            self.regret_net.eval()
            logging.debug(f"RegretNet in eval mode: {not self.regret_net.training}")
            strategy_logits = self.strategy_net(state_tensor)[0]
            q_values = self.regret_net(state_tensor)[0]
            self.strategy_net.train()
            self.regret_net.train()
            logging.debug(f"RegretNet back in train mode: {self.regret_net.training}")
            legal_mask = torch.zeros(self.num_actions, device=device)
            legal_mask[legal_actions] = 1
            strategy_logits = strategy_logits.masked_fill(legal_mask == 0, -1e9)
            strategy = torch.softmax(strategy_logits, dim=0).cpu().numpy()
            regrets = torch.relu(q_values - q_values.max()).cpu().numpy()
        
        self.cumulative_regrets[state_key] += regrets
        self.cumulative_strategies[state_key] += strategy
        if len(self.cumulative_regrets) > config.MAX_DICT_SIZE:
            self.cumulative_regrets.clear()
            self.cumulative_strategies.clear()
        
        positive_regrets = np.maximum(self.cumulative_regrets[state_key], 0)
        regret_sum = positive_regrets.sum()
        if regret_sum > 0:
            probs = positive_regrets / regret_sum
        else:
            probs = np.ones(self.num_actions) / len(legal_actions) if legal_actions else np.array([1.0])
        probs_dict = {a: float(probs[a]) if a in legal_actions else 0.0 for a in range(self.num_actions)}
        return {k: v for k, v in probs_dict.items() if k in legal_actions}

    def step(self, state, player_id, opponent_stats: OpponentStats):
        probs = self.action_probabilities(state, player_id, opponent_stats)
        action = random.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]
        return action

    def update_strategy_pool(self):
        strategy_dict = {}
        for state_key, cum_strategy in self.cumulative_strategies.items():
            total = cum_strategy.sum()
            if total > 0:
                strategy_dict[state_key] = cum_strategy / total
            else:
                strategy_dict[state_key] = np.ones(self.num_actions) / self.num_actions
        if len(self.strategy_pool) >= config.MAX_STRATEGY_POOL:
            self.strategy_pool.pop(0)
        self.strategy_pool.append(strategy_dict)
        self.cumulative_strategies.clear()
        logging.info(f"Strategy pool updated, size: {len(self.strategy_pool)}")
# Сбор опыта
# Сбор опыта
@ray.remote(num_cpus=1, num_gpus=0.5)
def collect_experience(game, agent, processor, steps, worker_id):
    try:
        env = game.new_initial_state()
        experiences = []
        opponent_stats = OpponentStats()
        step_count = 0
        while step_count < steps:
            if env.is_terminal():
                returns = env.returns()
                for pid in range(config.NUM_PLAYERS):
                    pos = (pid - (env.current_player() if not env.is_terminal() else 0) + config.NUM_PLAYERS) % config.NUM_PLAYERS
                    opponent_stats.update(pid, 0, [0, 0, 0, 0], sum(env.bets() if hasattr(env, 'bets') else [0] * config.NUM_PLAYERS), 
                                        0, False, pos, won=returns[pid])
                env = game.new_initial_state()
                continue
            player_id = env.current_player()
            if player_id < 0:
                action = random.choice(env.legal_actions())
                env.apply_action(action)
                continue
            bets = env.bets() if hasattr(env, 'bets') else [0] * config.NUM_PLAYERS
            stacks = env.stacks() if hasattr(env, 'stacks') else [100] * config.NUM_PLAYERS
            info_tensor = env.information_state_tensor(player_id)
            board_cards = [int(c) for i, c in enumerate(info_tensor[52:]) if c >= 0]
            stage = [0] * 4
            if len(board_cards) == 0:
                stage[0] = 1
            elif len(board_cards) == 3:
                stage[1] = 1
            elif len(board_cards) == 4:
                stage[2] = 1
            elif len(board_cards) == 5:
                stage[3] = 1
            action = agent.step(env, player_id, opponent_stats)
            next_state = env.clone()
            next_state.apply_action(action)
            raw_reward = next_state.returns()[player_id] if next_state.is_terminal() else 0
            if config.REWARD_NORMALIZATION == 'stack':
                reward = raw_reward / max(stacks[player_id], 1e-8)
            elif config.REWARD_NORMALIZATION == 'bb':
                reward = raw_reward / config.BB
            else:
                reward = raw_reward
            is_blind = player_id in [0, 1] and sum(bets) <= config.BB * 2
            position = (player_id - env.current_player() + config.NUM_PLAYERS) % config.NUM_PLAYERS
            bet_size = bets[player_id] if action > 1 else 0
            opponent_stats.update(player_id, action, stage, sum(bets), bet_size, is_blind, position)
            experiences.append((env.clone(), player_id, action, reward, next_state.clone(), next_state.is_terminal(), bets, stacks, stage))
            env = next_state
            step_count += 1
        logging.info(f"Worker {worker_id} collected {len(experiences)} experiences")
        return experiences
    except Exception as e:
        logging.error(f"Worker {worker_id} failed: {str(e)}")
        raise
# Обучение
# Обучение
class Trainer:
    def __init__(self, game, agent, processor):
        self.game = game
        self.agent = agent
        self.processor = processor
        self.buffer = PrioritizedReplayBuffer(config.BUFFER_CAPACITY)
        self.global_step = 0
        self.beta = 0.4
        self.reward_normalizer = RewardNormalizer()  # Добавляем нормализатор
        self.load_checkpoint()

    def load_checkpoint(self):
    checkpoint_path = config.MODEL_PATH
    logging.debug(f"Checking for checkpoint at {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path)
            self.agent.regret_net.load_state_dict(checkpoint['regret_net'])
            self.agent.strategy_net.load_state_dict(checkpoint['strategy_net'])
            self.agent.optimizer.load_state_dict(checkpoint['optimizer'])
            self.buffer.buffer = checkpoint['buffer']
            self.buffer.priorities = checkpoint['priorities']
            self.buffer._max_priority = checkpoint['max_priority']
            self.global_step = checkpoint['global_step']
            self.beta = checkpoint['beta']
            # Загружаем состояние нормализатора
            self.reward_normalizer.mean = checkpoint.get('reward_mean', 0.0)
            self.reward_normalizer.std = checkpoint.get('reward_std', 1.0)
            self.reward_normalizer.count = checkpoint.get('reward_count', 0)
            self.reward_normalizer.m2 = checkpoint.get('reward_m2', 0.0)
            logging.info(f"Loaded checkpoint from {checkpoint_path} at step {self.global_step}")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {str(e)}")
    else:
        logging.info(f"No checkpoint found at {checkpoint_path}, starting fresh")

    def save_checkpoint(self):
        checkpoint_path = config.MODEL_PATH
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            logging.debug(f"Created directory {checkpoint_dir}")
        try:
            checkpoint = {
                'regret_net': self.agent.regret_net.state_dict(),
                'strategy_net': self.agent.strategy_net.state_dict(),
                'optimizer': self.agent.optimizer.state_dict(),
                'buffer': self.buffer.buffer,
                'priorities': self.buffer.priorities,
                'max_priority': self.buffer._max_priority,
                'global_step': self.global_step,
                'beta': self.beta,
                # Сохраняем состояние нормализатора
                'reward_mean': self.reward_normalizer.mean,
                'reward_std': self.reward_normalizer.std,
                'reward_count': self.reward_normalizer.count,
                'reward_m2': self.reward_normalizer.m2
            }
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path} at step {self.global_step}")
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {str(e)}")
    def run_tournament(self):
        logging.info(f"Starting tournament with {len(self.agent.strategy_pool)} strategies in pool")
        if not self.agent.strategy_pool:
            logging.info("Tournament skipped: strategy pool is empty")
            return
        total_reward = 0
        num_games = 10
        game_count = 0
        opponent_stats = OpponentStats()
        for opponent_strategy in self.agent.strategy_pool:
            for _ in range(num_games // len(self.agent.strategy_pool)):
                game_count += 1
                logging.info(f"Playing tournament game {game_count}/{num_games}")
                env = self.game.new_initial_state()
                step_count = 0
                while not env.is_terminal():
                    step_count += 1
                    if step_count > 1000:
                        logging.error(f"Game {game_count} exceeded 1000 steps, aborting. Last state: {env.information_state_string(0)}")
                        break
                    player_id = env.current_player()
                    legal_actions = env.legal_actions()
                    if not legal_actions:
                        logging.warning(f"Game {game_count}, step {step_count}: No legal actions for player {player_id}")
                        break
                    if player_id < 0:
                        action = random.choice(legal_actions)
                        logging.debug(f"Game {game_count}, step {step_count}: Chance node, action={action}")
                    else:
                        bets = env.bets() if hasattr(env, 'bets') else [0] * config.NUM_PLAYERS
                        stacks = env.stacks() if hasattr(env, 'stacks') else [100] * config.NUM_PLAYERS
                        info_tensor = env.information_state_tensor(player_id)
                        board_cards = [int(c) for i, c in enumerate(info_tensor[52:]) if c >= 0]
                        stage = [0] * 4
                        if len(board_cards) == 0:
                            stage[0] = 1
                        elif len(board_cards) == 3:
                            stage[1] = 1
                        elif len(board_cards) == 4:
                            stage[2] = 1
                        elif len(board_cards) == 5:
                            stage[3] = 1
                        state_tensor = torch.tensor(self.processor.process([env], [player_id], [bets], [stacks], [stage], opponent_stats),
                                                  dtype=torch.float32, device=device)
                        with torch.no_grad():
                            self.agent.strategy_net.eval()
                            logits = self.agent.strategy_net(state_tensor)[0]
                            legal_mask = torch.zeros(self.agent.num_actions, device=device)
                            legal_mask[legal_actions] = 1
                            logits = logits.masked_fill(legal_mask == 0, -1e9)
                            probs = torch.softmax(logits, dim=0).cpu().numpy()
                            action = np.random.choice(legal_actions, p=probs[legal_actions] / probs[legal_actions].sum())
                        logging.debug(f"Game {game_count}, step {step_count}: player {player_id}, legal_actions={legal_actions}, action={action}, probs={probs[legal_actions]}")
                    env.apply_action(action)
                total_reward += env.returns()[0]
                logging.info(f"Game {game_count} completed, reward: {env.returns()[0]}")
        avg_reward = total_reward / num_games
        logging.info(f"Tournament completed: average reward = {avg_reward:.4f}")

    def train(self):
        pbar = tqdm(total=config.NUM_EPISODES, desc="Training")
        for episode in range(config.NUM_EPISODES):
            logging.info(f"Starting episode {episode}")
            logging.debug(f"Launching {config.NUM_WORKERS} workers")
            futures = [collect_experience.remote(self.game, self.agent, self.processor, config.STEPS_PER_WORKER, i)
                       for i in range(config.NUM_WORKERS)]
            logging.debug(f"Waiting for {len(futures)} futures")
            all_experiences = ray.get(futures)
            
            experiences = []
            for i, worker_experiences in enumerate(all_experiences):
                logging.info(f"Worker {i} returned {len(worker_experiences)} experiences")
                experiences.extend(worker_experiences)
            
            if experiences:
                # Нормализуем награды перед добавлением в буфер
                normalized_experiences = []
                for exp in experiences:
                    reward = exp[3]  # Предполагаем, что reward — это 4-й элемент кортежа
                    self.reward_normalizer.update(reward)
                    normalized_reward = self.reward_normalizer.normalize(reward)
                    normalized_exp = (exp[0], exp[1], exp[2], normalized_reward, exp[4], exp[5], exp[6], exp[7], exp[8])
                    normalized_experiences.append(normalized_exp)
                
                self.buffer.add_batch(normalized_experiences)
                logging.info(f"Collected {len(normalized_experiences)} total experiences from {config.NUM_WORKERS} workers")
                
                if len(self.buffer) >= config.BATCH_SIZE:
                    batch, indices, weights = self.buffer.sample(config.BATCH_SIZE, self.beta)
                    states, player_ids, actions, rewards, next_states, dones, bets, stacks, stages = zip(*batch)
                    # Дальше код остаётся без изменений
                    # ...
                    
                    opponent_stats = OpponentStats()
                    for exp in experiences:
                        is_blind = exp[1] in [0, 1] and sum(exp[6]) <= config.BB * 2
                        position = (exp[1] - exp[0].current_player() + config.NUM_PLAYERS) % config.NUM_PLAYERS
                        bet_size = exp[6][exp[1]] if exp[2] > 1 else 0
                        opponent_stats.update(exp[1], exp[2], exp[8], sum(exp[6]), bet_size, is_blind, position)
                    
                    states = torch.tensor(self.processor.process(states, player_ids, bets, stacks, stages, opponent_stats), 
                                        dtype=torch.float32, device=device)
                    actions = torch.tensor(actions, dtype=torch.long, device=device)
                    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                    next_player_ids = [s.current_player() if s.current_player() >= 0 else 0 for s in next_states]
                    next_states = torch.tensor(self.processor.process(next_states, next_player_ids, stacks, stacks, stages, opponent_stats), 
                                             dtype=torch.float32, device=device)
                    weights = torch.tensor(weights, dtype=torch.float32, device=device)
                    
                    self.agent.optimizer.zero_grad()
                    q_values = self.agent.regret_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    next_q_values = self.agent.strategy_net(next_states).max(1)[0].detach()
                    targets = rewards + config.GAMMA * next_q_values * (1 - torch.tensor(dones, dtype=torch.float32, device=device))
                    td_errors = (q_values - targets).abs().detach().cpu().numpy()
                    loss = (weights * nn.MSELoss(reduction='none')(q_values, targets)).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.agent.regret_net.parameters()) + list(self.agent.strategy_net.parameters()), 
                        config.GRAD_CLIP_VALUE
                    )
                    self.agent.optimizer.step()
                    
                    self.buffer.update_priorities(indices, td_errors + 1e-5)
                    self.global_step += 1
                    logging.info(f"Step {self.global_step} | Loss: {loss.item():.4f}")
                    self.beta = min(1.0, self.beta + 0.001)
                
                self.agent.update_strategy_pool()
                self.save_checkpoint()
            
            pbar.update(1)
            pbar.refresh()
        pbar.close()
        logging.info("Training completed, starting tournament phase")
        self.run_tournament()
# Запуск
if __name__ == "__main__":
    mp.set_start_method('spawn')
    ray.init(num_gpus=1, num_cpus=config.NUM_WORKERS, ignore_reinit_error=True)  # Устанавливаем num_cpus равным NUM_WORKERS
    game = pyspiel.load_game(config.GAME_NAME)
    processor = StateProcessor()
    agent = PokerAgent(game, processor)
    trainer = Trainer(game, agent, processor)
    trainer.train()
    ray.shutdown()
