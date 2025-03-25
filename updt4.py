#work
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
import pyspiel
import ray
from tqdm.auto import tqdm
import joblib
import argparse
from sklearn.cluster import KMeans
from collections import deque
from typing import List, Tuple, Dict, Optional
from open_spiel.python import policy

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
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
        self.NUM_EPISODES = 10  # Увеличено для лучшего обучения
        self.BATCH_SIZE = 64
        self.GAMMA = 0.96
        self.BUFFER_CAPACITY = 10000  # Увеличено для A100
        self.NUM_WORKERS = 8  # Оптимизировано для A100
        self.STEPS_PER_WORKER = 500
        self.SELFPLAY_UPDATE_FREQ = 50
        self.LOG_FREQ = 10
        self.TEST_INTERVAL = 50
        self.LEARNING_RATE = 1e-4
        self.NUM_PLAYERS = 6
        self.GRAD_CLIP_VALUE = 5.0
        self.CHECKPOINT_INTERVAL = 300  # Секунды
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
scaler = GradScaler()  # Для AMP на A100
game = pyspiel.load_game(config.GAME_NAME)
logging.info(f"Конфигурация инициализирована, устройство: {device}, CUDA: {torch.cuda.is_available()}")

# Нормализация наград
class RewardNormalizer:
    def __init__(self, eps: float = 1e-8, max_size: int = 10000):
        self.rewards = deque(maxlen=max_size)
        self.eps = eps
        self.mean = 0.0
        self.std = 1.0
        logging.info("RewardNormalizer инициализирован")

    def update(self, reward: float):
        self.rewards.append(reward)
        self.mean = np.mean(self.rewards) if self.rewards else 0.0
        self.std = np.std(self.rewards) + self.eps if self.rewards else 1.0

    def normalize(self, reward: float) -> float:
        return (reward - self.mean) / self.std if self.rewards else reward

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

# Модели с AMP
class RegretNet(nn.Module):
    def __init__(self, input_size: int, num_actions: int):
        super().__init__()
        assert input_size > 0, f"Invalid input_size: {input_size}"
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
        with autocast():
            return self.net(x)

class StrategyNet(nn.Module):
    def __init__(self, input_size: int, num_actions: int):
        super().__init__()
        assert input_size > 0, f"Invalid input_size: {input_size}"
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
        with autocast():
            return self.net(x)

# Встраивание карт
class CardEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank_embed = nn.Embedding(13, 8).to(device)
        self.suit_embed = nn.Embedding(4, 4).to(device)
        self.rank_embed.weight.data = self.rank_embed.weight.data.to(dtype=torch.float32)
        self.suit_embed.weight.data = self.suit_embed.weight.data.to(dtype=torch.float32)
        logging.info("CardEmbedding инициализирован")

    def forward(self, cards: List[int]) -> torch.Tensor:
        ranks = torch.tensor([c % 13 for c in cards], dtype=torch.long, device=device)
        suits = torch.tensor([c // 13 for c in cards], dtype=torch.long, device=device)
        rank_emb = self.rank_embed(ranks).mean(dim=0).to(dtype=torch.float32)
        suit_emb = self.suit_embed(suits).mean(dim=0).to(dtype=torch.float32)
        return torch.cat([rank_emb, suit_emb]).to(dtype=torch.float32)

# Обработка состояний
class StateProcessor:
    def __init__(self):
        self.card_embedding = CardEmbedding()
        self.buckets = self._load_or_precompute_buckets()
        self.state_size = config.NUM_BUCKETS + config.NUM_PLAYERS * 3 + 4 + 5
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
        kmeans = KMeans(n_clusters=config.NUM_BUCKETS, random_state=42).fit(np.array(all_hands, dtype=np.float32))
        joblib.dump(kmeans, config.KMEANS_PATH)
        logging.info(f"Бакеты вычислены и сохранены в {config.KMEANS_PATH}")
        return kmeans

    def process(self, states: List, player_ids: List[int], bets: List[List[float]] = None, 
                stacks: List[List[float]] = None, stages: List[List[int]] = None, 
                opponent_stats: Optional['OpponentStats'] = None) -> np.ndarray:
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

        card_embs = torch.stack([self.card_embedding(cards) for cards in cards_batch]).cpu().detach().numpy().astype(np.float32)
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

        processed = np.concatenate([
            bucket_one_hot, bets_norm, stacks_norm, action_history, np.array(stages, dtype=np.float32),
            np.array([sprs, positions, np.zeros(batch_size), np.zeros(batch_size), np.zeros(batch_size)], dtype=np.float32).T
        ], axis=1).astype(np.float32)

        if np.any(np.isnan(processed)) or np.any(np.isinf(processed)):
            logging.error(f"NaN/Inf in processed: {processed}")
            raise ValueError("Invalid state processing detected")

        if len(self.cache) >= self.max_cache_size:
            self.cache.clear()
        for key, proc in zip(state_keys, processed):
            self.cache[key] = proc
        return processed

# Статистика оппонентов
class OpponentStats:
    def __init__(self):
        self.stats = {i: {
            'vpip': 0, 'pfr': 0, 'af': 0, 'hands': 0, 'folds': 0, 'calls': 0, 'raises': 0, 'last_bet': 0,
            'fold_to_cbet': 0, 'cbet_opp': 0, 'fold_to_3bet': 0, '3bet_opp': 0
        } for i in range(config.NUM_PLAYERS)}

    def update(self, player_id: int, action: int, stage: List[int], pot: float, bet_size: float, is_blind: bool):
        stage_idx = np.argmax(stage)
        self.stats[player_id]['hands'] += 1
        if stage_idx == 0 and not is_blind:
            if action > 0:
                self.stats[player_id]['vpip'] += 1
            if action > 1:
                self.stats[player_id]['pfr'] += 1
                self.stats[player_id]['3bet_opp'] += 1  # Считаем возможность 3-бета
                if bet_size > pot:  # Условно считаем 3-бетом, если ставка больше пота
                    self.stats[player_id]['raises'] += bet_size
        if action == 0:
            self.stats[player_id]['folds'] += 1
            if stage_idx > 0 and self.stats[player_id]['3bet_opp'] > 0:
                self.stats[player_id]['fold_to_3bet'] += 1  # Увеличиваем, если фолд после 3-бета
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
        three_bet_opp = max(self.stats[player_id]['3bet_opp'], 1)
        return {
            'vpip': float(self.stats[player_id]['vpip'] / hands),
            'pfr': float(self.stats[player_id]['pfr'] / hands),
            'af': float(self.stats[player_id]['af']),
            'last_bet': float(self.stats[player_id]['last_bet']),
            'fold_to_3bet': float(self.stats[player_id]['fold_to_3bet'] / three_bet_opp)
        }

# Агент с динамическими ставками и эвристическими наградами
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
        self.cumulative_regrets = {}
        self.cumulative_strategies = {}
        self.strategy_pool = []
        self.global_step = 0
        logging.info(f"PokerAgent инициализирован, state_size={processor.state_size}, num_actions={self.num_actions}")

    def _heuristic_reward(self, action: int, state, player_id: int, bets: List[float], opponent_stats: OpponentStats) -> float:
        pot = sum(bets)
        opp_metrics = [opponent_stats.get_metrics(i) for i in range(config.NUM_PLAYERS) if i != player_id]
        info_tensor = state.information_state_tensor(player_id)
        cards = [int(c) for i, c in enumerate(info_tensor[:52]) if c > 0][:2]
        hand_strength = 0.5 if len(set([c // 13 for c in cards])) == 1 or max([c % 13 for c in cards]) >= 10 else 0.2
        if action == 0 and pot > 0:
            return -0.1 * (1 + pot / config.BB)
        elif action > 1 and any(m['fold_to_3bet'] > 0.7 for m in opp_metrics):
            return 0.3 * hand_strength
        return 0.0

    def _dynamic_bet_size(self, state, stacks: List[float], bets: List[float], player_id: int, q_value: float, stage: List[int]) -> float:
        legal_actions = state.legal_actions(player_id)
        if not any(a in legal_actions for a in [2, 3]):
            return 0.0
        pot = sum(bets)
        my_stack = stacks[player_id]
        min_raise = pot * 0.1
        max_raise = my_stack
        base = min_raise + (max_raise - min_raise) * q_value
        return min(max(base, min_raise), max_raise)

    def action_probabilities(self, state, player_id: Optional[int] = None, opponent_stats: Optional[OpponentStats] = None) -> Dict[int, float]:
        legal_actions = state.legal_actions(player_id)
        if not legal_actions:
            return {0: 1.0}
    
        bets = state.bets() if hasattr(state, 'bets') else [0] * config.NUM_PLAYERS
        stacks = state.stacks() if hasattr(state, 'stacks') else [100] * config.NUM_PLAYERS
        stage = [0] * 4
        info_tensor = state.information_state_tensor(player_id)
        board_cards = [int(c) for i, c in enumerate(info_tensor[52:]) if c >= 0]
        if len(board_cards) == 0:
            stage[0] = 1
        elif len(board_cards) == 3:
            stage[1] = 1
        elif len(board_cards) == 4:
            stage[2] = 1
        elif len(board_cards) == 5:
            stage[3] = 1
        state_tensor = torch.tensor(self.processor.process([state], [player_id], [bets], [stacks], [stage]), 
                                    dtype=torch.float32, device=device)
    
        self.regret_net.eval()
        self.strategy_net.eval()
        with torch.no_grad():
            regrets = self.regret_net(state_tensor)[0]
            strategy_logits = self.strategy_net(state_tensor)[0]
            legal_mask = torch.zeros(self.num_actions, device=device)
            legal_mask[legal_actions] = 1
            strategy_logits = strategy_logits.masked_fill(legal_mask == 0, -1e4)
            strategy = torch.softmax(strategy_logits, dim=0).cpu().numpy()
    
        self.regret_net.train()
        self.strategy_net.train()
    
        state_key = state.information_state_string(player_id)
        if state_key not in self.cumulative_regrets:
            # Инициализируем регреты небольшими положительными значениями для легальных действий
            self.cumulative_regrets[state_key] = np.ones(self.num_actions) * 0.01
            self.cumulative_strategies[state_key] = np.zeros(self.num_actions)
        self.cumulative_regrets[state_key] += regrets.cpu().numpy()
        self.cumulative_strategies[state_key] += strategy
    
        if len(self.cumulative_regrets) > config.MAX_DICT_SIZE:
            self.cumulative_regrets.pop(next(iter(self.cumulative_regrets)))
            self.cumulative_strategies.pop(next(iter(self.cumulative_strategies)))
    
        # Учитываем только легальные действия
        positive_regrets = np.maximum(self.cumulative_regrets[state_key][legal_actions], 0)
        regret_sum = positive_regrets.sum()
        if regret_sum > 0:
            probs = positive_regrets / regret_sum
        else:
            probs = np.ones(len(legal_actions)) / len(legal_actions)
    
        # Создаём словарь только для легальных действий
        legal_probs = {legal_actions[i]: float(probs[i]) for i in range(len(legal_actions))}
        total_prob = sum(legal_probs.values())
        if total_prob <= 0:  # Это теперь редкий случай
            legal_probs = {a: 1.0 / len(legal_actions) for a in legal_actions}
            logging.warning(f"Zero total probability detected for state {state_key}, using uniform distribution over legal actions {legal_actions}")
        
        return legal_probs

    def step(self, state, player_id: int, bets: List[float], stacks: List[float], stage: List[int], opponent_stats: OpponentStats) -> int:
        probs = self.action_probabilities(state, player_id, opponent_stats)
        total_weight = sum(probs.values())
        if total_weight <= 0:
            logging.error(f"Invalid probabilities in step: {probs}")
            raise ValueError("Total of weights must be greater than zero")
        actions, weights = zip(*probs.items())
        action = random.choices(actions, weights=weights, k=1)[0]
        if action > 1:
            state_tensor = torch.tensor(self.processor.process([state], [player_id], [bets], [stacks], [stage]), 
                                        dtype=torch.float32, device=device)
            self.strategy_net.eval()
            with torch.no_grad():
                q_value = self.strategy_net(state_tensor).max().item()
            self.strategy_net.train()
            bet_size = self._dynamic_bet_size(state, stacks, bets, player_id, q_value, stage)
            if bet_size >= stacks[player_id] and 3 in state.legal_actions(player_id):
                action = 3
        return action

    def update_strategy_pool(self):
        strategy_dict = {}
        for state_key, cum_strategy in self.cumulative_strategies.items():
            total = cum_strategy.sum()
            strategy_dict[state_key] = cum_strategy / total if total > 0 else np.ones(self.num_actions) / self.num_actions
        if len(self.strategy_pool) >= config.MAX_STRATEGY_POOL:
            self.strategy_pool.pop(0)
        self.strategy_pool.append(strategy_dict)
        self.cumulative_strategies.clear()
        logging.info(f"Strategy pool updated, size: {len(self.strategy_pool)}")
# Сбор опыта
@ray.remote(num_cpus=1, num_gpus=0.125)
def collect_experience(game, agent, processor, steps, worker_id):
    env = game.new_initial_state()
    experiences = []
    opponent_stats = OpponentStats()
    for _ in range(steps):
        if env.is_terminal():
            env = game.new_initial_state()
            continue
        player_id = env.current_player()
        if player_id < 0:
            action = random.choice(env.legal_actions())
            env.apply_action(action)
            continue
        bets = env.bets() if hasattr(env, 'bets') else [0] * config.NUM_PLAYERS
        stacks = env.stacks() if hasattr(env, 'stacks') else [100] * config.NUM_PLAYERS
        stage = [0] * 4
        info_tensor = env.information_state_tensor(player_id)
        board_cards = [int(c) for i, c in enumerate(info_tensor[52:]) if c >= 0]
        if len(board_cards) == 0:
            stage[0] = 1
        elif len(board_cards) == 3:
            stage[1] = 1
        elif len(board_cards) == 4:
            stage[2] = 1
        elif len(board_cards) == 5:
            stage[3] = 1
        action = agent.step(env, player_id, bets, stacks, stage, opponent_stats)
        next_state = env.clone()
        next_state.apply_action(action)
        reward = next_state.returns()[player_id] if next_state.is_terminal() else agent._heuristic_reward(action, env, player_id, bets, opponent_stats)
        if config.REWARD_NORMALIZATION == 'stack':
            reward /= max(stacks[player_id], 1e-8)
        experiences.append((env.clone(), player_id, action, reward, next_state.clone(), next_state.is_terminal(), bets, stacks, stage))
        env = next_state
    logging.info(f"Worker {worker_id} collected {len(experiences)} experiences")
    return experiences

# Агенты для турнира
class TightAggressiveAgent(policy.Policy):
    def __init__(self, game):
        super().__init__(game, list(range(game.num_players())))

    def action_probabilities(self, state, player_id: Optional[int] = None) -> Dict[int, float]:
        legal_actions = state.legal_actions(player_id)
        if not legal_actions:
            return {0: 1.0}
        info = state.information_state_tensor(player_id)
        cards = [int(i) for i, c in enumerate(info[:52]) if c > 0][:2]
        strength = 0.7 if len(set([c // 13 for c in cards])) == 1 or max([c % 13 for c in cards]) >= 10 else 0.2
        probs = {a: 0.0 for a in legal_actions}
        if strength > 0.5 and 3 in legal_actions:
            probs[3] = 0.8
            probs[1] = 0.2
        elif 1 in legal_actions:
            probs[1] = 0.3
            probs[0] = 0.7
        else:
            probs[0] = 1.0
        total = sum(probs.values())
        if total <= 0:  # Гарантируем ненулевую сумму
            probs = {a: 1.0 / len(legal_actions) for a in legal_actions}
            logging.warning(f"TightAggressiveAgent: Zero probability sum, using uniform distribution over {legal_actions}")
        return probs

class LooseAggressiveAgent(policy.Policy):
    def __init__(self, game):
        super().__init__(game, list(range(game.num_players())))

    def action_probabilities(self, state, player_id: Optional[int] = None) -> Dict[int, float]:
        legal_actions = state.legal_actions(player_id)
        if not legal_actions:
            return {0: 1.0}
        probs = {a: 0.0 for a in legal_actions}
        if 3 in legal_actions:
            probs[3] = 0.7
            probs[1] = 0.3
        elif 1 in legal_actions:
            probs[1] = 0.6
            probs[0] = 0.4
        else:
            probs[0] = 1.0
        total = sum(probs.values())
        if total <= 0:  # Гарантируем ненулевую сумму
            probs = {a: 1.0 / len(legal_actions) for a in legal_actions}
            logging.warning(f"LooseAggressiveAgent: Zero probability sum, using uniform distribution over {legal_actions}")
        return probs

# Обучение
class Trainer:
    def __init__(self, game, agent, processor):
        self.game = game
        self.agent = agent
        self.processor = processor
        self.buffer = PrioritizedReplayBuffer(config.BUFFER_CAPACITY)
        self.reward_normalizer = RewardNormalizer()
        self.global_step = 0
        self.beta = 0.4
        self.interrupted = False
        self.last_checkpoint_time = time.time()
        self.best_avg_reward = float('-inf')
        self.load_checkpoint()

    def load_checkpoint(self):
        if os.path.exists(config.MODEL_PATH):
            checkpoint = torch.load(config.MODEL_PATH)
            self.agent.regret_net.load_state_dict(checkpoint['regret_net'])
            self.agent.strategy_net.load_state_dict(checkpoint['strategy_net'])
            self.agent.optimizer.load_state_dict(checkpoint['optimizer'])
            self.global_step = checkpoint['global_step']
            self.beta = checkpoint['beta']
            self.reward_normalizer.mean = checkpoint.get('reward_mean', 0.0)
            self.reward_normalizer.std = checkpoint.get('reward_std', 1.0)
            self.reward_normalizer.rewards = checkpoint.get('reward_buffer', deque(maxlen=config.BUFFER_CAPACITY))
            logging.info(f"Loaded checkpoint from {config.MODEL_PATH} at step {self.global_step}")

    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'regret_net': self.agent.regret_net.state_dict(),
            'strategy_net': self.agent.strategy_net.state_dict(),
            'optimizer': self.agent.optimizer.state_dict(),
            'global_step': self.global_step,
            'beta': self.beta,
            'reward_mean': self.reward_normalizer.mean,
            'reward_std': self.reward_normalizer.std,
            'reward_buffer': self.reward_normalizer.rewards
        }
        torch.save(checkpoint, config.MODEL_PATH)
        if is_best:
            torch.save(checkpoint, config.BEST_MODEL_PATH)
            logging.info(f"Saved best model to {config.BEST_MODEL_PATH} at step {self.global_step}")
        self.last_checkpoint_time = time.time()
        logging.info(f"Saved checkpoint to {config.MODEL_PATH} at step {self.global_step}")

    def run_tournament(self):
        num_games = 20
        total_reward = {'self': 0, 'tight': 0, 'loose': 0}
        tag_agent = TightAggressiveAgent(self.game)
        lag_agent = LooseAggressiveAgent(self.game)
        opponent_stats = OpponentStats()
    
        for opp_type, opp in [('self', self.agent), ('tight', tag_agent), ('loose', lag_agent)]:
            for game_idx in range(num_games):
                state = self.game.new_initial_state()
                while not state.is_terminal():
                    player_id = state.current_player()
                    if player_id < 0:  # Обработка шансных нод
                        action = random.choice(state.legal_actions())
                        state.apply_action(action)
                        continue
                    bets = state.bets() if hasattr(state, 'bets') else [0] * config.NUM_PLAYERS
                    stacks = state.stacks() if hasattr(state, 'stacks') else [100] * config.NUM_PLAYERS
                    stage = [0] * 4
                    info_tensor = state.information_state_tensor(player_id)
                    board_cards = [int(c) for i, c in enumerate(info_tensor[52:]) if c >= 0]
                    if len(board_cards) == 0:
                        stage[0] = 1
                    elif len(board_cards) == 3:
                        stage[1] = 1
                    elif len(board_cards) == 4:
                        stage[2] = 1
                    elif len(board_cards) == 5:
                        stage[3] = 1
                    if player_id == 0:
                        action = self.agent.step(state, player_id, bets, stacks, stage, opponent_stats)
                    else:
                        probs = opp.action_probabilities(state, player_id)
                        action = random.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]
                    # Проверка валидности действия
                    legal_actions = state.legal_actions(player_id)
                    if action not in legal_actions:
                        logging.error(f"Invalid action {action} chosen, legal actions: {legal_actions}")
                        action = random.choice(legal_actions)  # Fallback на случайное легальное действие
                    state.apply_action(action)
                total_reward[opp_type] += state.returns()[0]
                logging.info(f"Game {game_idx+1}/{num_games} vs {opp_type} completed, reward: {state.returns()[0]}")
            avg_reward = total_reward[opp_type] / num_games
            logging.info(f"Avg reward vs {opp_type}: {avg_reward:.4f}")
    
        overall_avg_reward = sum(total_reward.values()) / (num_games * 3)
        if overall_avg_reward > self.best_avg_reward:
            self.best_avg_reward = overall_avg_reward
            self.save_checkpoint(is_best=True)
        return overall_avg_reward
    def train(self):
        pbar = tqdm(total=config.NUM_EPISODES, desc="Training")

        def signal_handler(sig, frame):
            self.interrupted = True
            self.save_checkpoint()
            ray.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        for episode in range(config.NUM_EPISODES):
            logging.info(f"Starting episode {episode}")
            futures = [collect_experience.remote(self.game, self.agent, self.processor, config.STEPS_PER_WORKER, i)
                       for i in range(config.NUM_WORKERS)]
            all_experiences = ray.get(futures)
            experiences = []
            for worker_experiences in all_experiences:
                experiences.extend(worker_experiences)

            if experiences:
                normalized_experiences = []
                for exp in experiences:
                    reward = exp[3]
                    self.reward_normalizer.update(reward)
                    normalized_reward = self.reward_normalizer.normalize(reward)
                    normalized_experiences.append((exp[0], exp[1], exp[2], normalized_reward, exp[4], exp[5], exp[6], exp[7], exp[8]))
                self.buffer.add_batch(normalized_experiences)

                if len(self.buffer) >= config.BATCH_SIZE:
                    batch, indices, weights = self.buffer.sample(config.BATCH_SIZE, self.beta)
                    states, player_ids, actions, rewards, next_states, dones, bets, stacks, stages = zip(*batch)
                    states = torch.tensor(self.processor.process(states, player_ids, bets, stacks, stages), 
                                        dtype=torch.float32, device=device)
                    actions = torch.tensor(actions, dtype=torch.long, device=device)
                    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                    next_states = torch.tensor(self.processor.process(next_states, player_ids, bets, stacks, stages), 
                                             dtype=torch.float32, device=device)

                    self.agent.optimizer.zero_grad()
                    with autocast():
                        q_values = self.agent.regret_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                        next_q_values = self.agent.strategy_net(next_states).max(1)[0].detach()
                        targets = rewards + config.GAMMA * next_q_values * (1 - torch.tensor(dones, dtype=torch.float32, device=device))
                        loss = nn.MSELoss()(q_values, targets)
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.agent.optimizer)
                    torch.nn.utils.clip_grad_norm_(list(self.agent.regret_net.parameters()) + list(self.agent.strategy_net.parameters()), config.GRAD_CLIP_VALUE)
                    scaler.step(self.agent.optimizer)
                    scaler.update()

                    td_errors = (q_values - targets).abs().detach().cpu().numpy()
                    self.buffer.update_priorities(indices, td_errors + 1e-5)
                    self.global_step += 1
                    logging.info(f"Step {self.global_step} | Loss: {loss.item():.4f}")
                    self.beta = min(1.0, self.beta + 0.001)

                self.agent.update_strategy_pool()
                if time.time() - self.last_checkpoint_time >= config.CHECKPOINT_INTERVAL:
                    self.save_checkpoint()

            pbar.update(1)
            pbar.refresh()

            if self.global_step % config.TEST_INTERVAL == 0 and self.global_step > 0:
                logging.info(f"Running tournament at step {self.global_step}")
                self.run_tournament()

            if self.interrupted:
                break

        pbar.close()
        logging.info("Training completed, running final tournament")
        final_avg_reward = self.run_tournament()
        logging.info(f"Final average reward across all opponents: {final_avg_reward:.4f}")
        self.save_checkpoint()

# Запуск
if __name__ == "__main__":
    mp.set_start_method('spawn')
    ray.init(num_gpus=1, num_cpus=config.NUM_WORKERS, ignore_reinit_error=True)
    game = pyspiel.load_game(config.GAME_NAME)
    processor = StateProcessor()
    agent = PokerAgent(game, processor)
    trainer = Trainer(game, agent, processor)
    trainer.train()
    ray.shutdown()
