# Импорт всех необходимых библиотек3456780
import os
import time
import logging
import sys
import signal
import random
import gc
from collections import deque
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp
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
import traceback
from functools import lru_cache
import argparse

# ===== КОНФИГУРАЦИЯ =====
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
        self.NUM_WORKERS = 1
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
        self.GAME_NAME = "universal_poker(betting=nolimit,numPlayers=6,numRounds=4,blind=1 2 3 4 5 6,raiseSize=0.10 0.20 0.40 0.80,stack=100 100 100 100 100 100,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1)"

# Инициализация
config = Config()
os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
os.makedirs(config.LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.LOG_DIR, 'training.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
writer = SummaryWriter(log_dir=config.LOG_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
game = pyspiel.load_game(config.GAME_NAME)
logging.info(f"Конфигурация инициализирована, устройство: {device}, CUDA: {torch.cuda.is_available()}, Версия CUDA: {torch.version.cuda}")

# ===== НОРМАЛИЗАЦИЯ НАГРАД =====
class RewardNormalizer:
    def __init__(self, eps: float = 1e-8, max_size: int = 10000):
        self.rewards = []
        self.eps = eps
        self.max_size = max_size
        self.mean = 0
        self.std = 1
        logging.info("RewardNormalizer инициализирован")

    def update(self, reward: float):
        try:
            self.rewards.append(reward)
            if len(self.rewards) > self.max_size:
                self.rewards.pop(0)
            self.mean = np.mean(self.rewards) if self.rewards else 0
            self.std = np.std(self.rewards) + self.eps if self.rewards else 1
        except Exception as e:
            logging.error(f"Ошибка в update RewardNormalizer: {traceback.format_exc()}")
            raise

    def normalize(self, reward: float) -> float:
        if not self.rewards:
            logging.debug("Reward buffer empty, returning default normalized value 0")
            return 0.0
        return (reward - self.mean) / self.std

# ===== МОДЕЛИ =====
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
        with autocast():
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
        with autocast():
            return self.net(x)

# ===== БУФЕР ОПЫТА =====
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = config.BUFFER_CAPACITY, alpha: float = 0.6):
        self.buffer = deque(maxlen=capacity)
        self._lock = mp.Lock()
        self.alpha = alpha
        self.priorities = deque(maxlen=capacity)
        self._max_priority = 1.0
        logging.info("PrioritizedReplayBuffer инициализирован")

    def add_batch(self, experiences: List[Tuple], processor: 'StateProcessor'):
        try:
            with self._lock:
                processed_experiences = []
                states, actions, rewards, next_states, dones, player_ids, bets, stacks, stages = zip(*experiences)
                processed_states = processor.process(states, player_ids, bets, stacks, stages)
                processed_next_states = processor.process(next_states, player_ids, bets, stacks, stages)
                for i in range(len(experiences)):
                    processed_experiences.append((
                        processed_states[i], actions[i], rewards[i], processed_next_states[i], dones[i],
                        player_ids[i], bets[i], stacks[i], stages[i]
                    ))
                self.buffer.extend(processed_experiences)
                self.priorities.extend([self._max_priority] * len(experiences))
            logging.debug(f"Добавлено {len(experiences)} опытов в буфер")
        except Exception as e:
            logging.error(f"Ошибка в add_batch: {traceback.format_exc()}")
            raise

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Tuple], List[int], np.ndarray]:
        with self._lock:
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
            writer.add_scalar('Buffer/Size', len(self.buffer), self.global_step)
            return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        with self._lock:
            for idx, priority in zip(indices, priorities):
                self.priorities[idx] = priority
                self._max_priority = max(self._max_priority, priority)

    def __len__(self) -> int:
        return len(self.buffer)

    def set_global_step(self, step: int):
        self.global_step = step

# ===== EMBEDDING КАРТ =====
class CardEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank_embed = nn.Embedding(13, 8).to(device)
        self.suit_embed = nn.Embedding(4, 4).to(device)
        logging.info("CardEmbedding инициализирован")

    def forward(self, cards: List[int]) -> torch.Tensor:
        ranks = torch.tensor([c % 13 for c in cards], dtype=torch.long, device=device)
        suits = torch.tensor([c // 13 for c in cards], dtype=torch.long, device=device)
        return torch.cat([self.rank_embed(ranks).mean(dim=0), self.suit_embed(suits).mean(dim=0)])

# ===== СТАТИСТИКА ОППОНЕНТОВ =====
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

# ===== ОБРАБОТКА СОСТОЯНИЙ =====
class StateProcessor:
    def __init__(self):
        self.card_embedding = CardEmbedding()
        self.buckets = self._load_or_precompute_buckets()
        self.cache = {}
        self.max_cache_size = 1000
        self.state_size = (
            config.NUM_BUCKETS + config.NUM_PLAYERS * 3 + 4 + 5
        )
        logging.info(f"StateProcessor инициализирован, state_size={self.state_size}")

    def _load_or_precompute_buckets(self):
        if os.path.exists(config.KMEANS_PATH):
            try:
                buckets = joblib.load(config.KMEANS_PATH)
                logging.info(f"Загружены бакеты из {config.KMEANS_PATH}")
                return buckets
            except Exception as e:
                logging.warning(f"Не удалось загрузить KMeans: {e}, пересчитываем")
        all_hands = []
        for r1 in range(13):
            for r2 in range(r1, 13):
                for suited in [0, 1]:
                    hand = [r1 + s1 * 13 for s1 in range(4)] + [r2 + s2 * 13 for s2 in range(4)]
                    # Исправлено: явное преобразование в float32
                    embedding = self.card_embedding(hand[:2]).float().cpu().detach().numpy().astype(np.float32)
                    all_hands.append(embedding)
        kmeans = KMeans(n_clusters=config.NUM_BUCKETS, random_state=42).fit(all_hands)
        joblib.dump(kmeans, config.KMEANS_PATH)
        logging.info(f"Бакеты вычислены и сохранены в {config.KMEANS_PATH}")
        return kmeans

    def process(self, states: List, player_ids: List[int], bets: List[List[float]], stacks: List[List[float]], stages: List[List[int]], 
                opponent_stats: Optional[OpponentStats] = None) -> np.ndarray:
        batch_size = len(states)
        logging.debug(f"Processing batch_size={batch_size}, player_ids={player_ids}")
        
        state_keys = [f"{s.information_state_string(pid)}_{pid}_{tuple(b)}_{tuple(stk)}" 
                      for s, pid, b, stk in zip(states, player_ids, bets, stacks)]
        
        cached = [self.cache.get(key) for key in state_keys]
        if all(c is not None for c in cached):
            logging.debug("Returning cached data")
            return np.array(cached, dtype=np.float32)
        
        info_states = [s.information_state_tensor(pid) for s, pid in zip(states, player_ids)]
        
        cards_batch = []
        for info in info_states:
            private_cards = [int(i) for i, c in enumerate(info[:52]) if c > 0][:2]
            if not private_cards:
                logging.warning("No private cards found, using default [0, 1]")
                private_cards = [0, 1]
            elif len(private_cards) < 2:
                logging.warning(f"Found {len(private_cards)} private cards, padding to 2: {private_cards}")
                private_cards.extend([1] * (2 - len(private_cards)))
            cards_batch.append(private_cards)
        
        # Исправленный блок обработки карт
        card_embs = torch.stack([self.card_embedding(cards).float() for cards in cards_batch]).cpu().detach().numpy()
        card_embs = card_embs.astype(np.float32)  # Явное преобразование в float32
        
        # Проверка перед predict
        logging.debug(f"card_embs shape={card_embs.shape}, dtype={card_embs.dtype}")
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
                                   [min(h, 4) for h in s.action_history()[-config.NUM_PLAYERS:]]) for s in states], dtype=np.float32)
        
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
        
        table_aggs = np.array([np.mean([m['af'] for m in metrics]) if metrics else 0.5 for metrics in opponent_metrics], dtype=np.float32)
        last_bets = np.array([max([m['last_bet'] for m in metrics]) / pot if pot > 0 and metrics else 0.0 
                             for metrics, pot in zip(opponent_metrics, pots)], dtype=np.float32)
        all_in_flags = np.array([1.0 if any(b >= stk[i] for i, b in enumerate(bet) if i != pid) else 0.0 
                                for bet, stk, pid in zip(bets, stacks, player_ids)], dtype=np.float32)
        
        processed = np.concatenate([
            bucket_one_hot, bets_norm, stacks_norm, action_history, np.array(stages, dtype=np.float32),
            np.array([sprs, table_aggs, positions, last_bets, all_in_flags], dtype=np.float32).T, opp_features
        ], axis=1).astype(np.float32)
        
        if np.any(np.isnan(processed)) or np.any(np.isinf(processed)):
            logging.error(f"NaN/Inf in processed: {processed}")
            raise ValueError("Invalid state processing detected")
        
        if len(self.cache) >= self.max_cache_size:
            self.cache.clear()
        for key, proc in zip(state_keys, processed):
            self.cache[key] = proc
        
        gc.collect()
        return processed

# ===== АГЕНТ =====
class PokerAgent(policy.Policy):
    def __init__(self, game, processor: StateProcessor):
        player_ids = list(range(game.num_players()))
        super().__init__(game, player_ids)
        self.game = game
        self.processor = processor
        self.num_actions = game.num_distinct_actions()
        self.regret_net = RegretNet(processor.state_size, self.num_actions).to(device)
        self.strategy_net = StrategyNet(processor.state_size, self.num_actions).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.regret_net.parameters()) + list(self.strategy_net.parameters()),
            lr=config.LEARNING_RATE
        )
        self.cumulative_regrets = {}
        self.cumulative_strategies = {}
        self.max_dict_size = 10000
        self.strategy_pool = []
        self.best_winrate = -float('inf')
        self.global_step = 0
        logging.info(f"PokerAgent инициализирован, state_size={processor.state_size}, num_actions={self.num_actions}")

    def _hand_strength(self, state, player_id: int) -> float:
        info = state.information_state_tensor(player_id)
        cards = [int(c) for i, c in enumerate(info[:52]) if c >= 0 and i in state.player_cards(player_id)]
        ranks = sorted([c % 13 for c in cards], reverse=True)
        is_suited = len(set([c // 13 for c in cards])) == 1
        if len(ranks) < 2:
            return 0.1
        if ranks[0] >= 10:
            return 0.6 if is_suited else 0.5
        if ranks[0] == ranks[1]:
            return 0.7
        return 0.3 if is_suited else 0.2

    def _heuristic_reward(self, action: int, state, player_id: int, bets: List[float], opponent_stats: OpponentStats) -> float:
        pot = sum(bets)
        opp_metrics = [opponent_stats.get_metrics(i) for i in range(config.NUM_PLAYERS) if i != player_id]
        hand_strength = self._hand_strength(state, player_id)
        info_tensor = state.information_state_tensor(player_id)
        board_cards = [int(c) for i, c in enumerate(info_tensor[52:]) if c >= 0]
        num_board_cards = len(board_cards)
        stage_idx = 0
        if num_board_cards == 0:
            stage_idx = 0
        elif num_board_cards == 3:
            stage_idx = 1
        elif num_board_cards == 4:
            stage_idx = 2
        elif num_board_cards == 5:
            stage_idx = 3
        is_drawy = len(set([c // 13 for c in board_cards])) < 3 if board_cards else False

        if action == 0 and pot > 0:
            reward = -0.1 * (1 + pot / config.BB) * (1.5 if is_drawy else 1.0)
        elif action > 1:
            if any(m['fold_to_3bet'] > 0.7 for m in opp_metrics) and stage_idx == 0:
                reward = 0.3 * hand_strength
            elif any(m['street_aggression'][stage_idx] < 0.2 for m in opp_metrics):
                reward = 0.2 + 0.1 * hand_strength * (1.2 if is_drawy else 1.0)
            else:
                reward = 0.1 * hand_strength
        elif action == 1 and any(m['last_bet'] > pot * 0.5 for m in opp_metrics):
            reward = 0.1 * hand_strength if random.random() < 0.5 else -0.1
        else:
            reward = 0.0
        logging.debug(f"Heuristic reward for action {action}: {reward}")
        return reward

    def _dynamic_bet_size(self, state, stacks: List[float], bets: List[float], position: int, opponent_metrics: Dict[str, float], 
                         q_value: float, stage: List[int], last_opp_bet: float) -> float:
        legal_actions = state.legal_actions(position)
        if not any(a in legal_actions for a in [2, 3, 4]):
            return 0.0
        pot = sum(bets)
        my_stack = stacks[position]
        stage_idx = np.argmax(stage)
        min_raise, max_raise = state.raise_bounds() if hasattr(state, 'raise_bounds') else (pot, my_stack)
        min_bet = state.min_bet() if hasattr(state, 'min_bet') else config.BB
        base = min_bet * (1 + q_value * 4) if 2 in legal_actions else min_raise + (max_raise - min_raise) * q_value
        base = min(max(base, min_raise if 3 in legal_actions else min_bet), max_raise)
        board_cards = [int(c) for i, c in enumerate(state.information_state_tensor(position)[52:]) if c >= 0]
        is_monotone = len(set([c // 13 for c in board_cards])) == 1 and len(board_cards) >= 3
        base *= 1.3 if is_monotone and stage_idx > 1 else 1.0
        if opponent_metrics.get('fold_to_cbet', 0) > 0.6 and stage_idx == 1:
            base = max(base, min_bet * 1.5)
        if last_opp_bet > 0:
            base = max(base, last_opp_bet * 1.1)
        logging.debug(f"Dynamic bet size: {base}, q_value={q_value}, stage_idx={stage_idx}")
        return base

    def action_probabilities(self, state, player_id: Optional[int] = None, opponent_stats: Optional[OpponentStats] = None) -> Dict[int, float]:
        legal_actions = state.legal_actions(player_id)
        if not legal_actions:
            return {0: 1.0}
        info_state = state.information_state_string(player_id)
        bets = state.bets() if hasattr(state, 'bets') else [0] * config.NUM_PLAYERS
        stacks = state.stacks() if hasattr(state, 'stacks') else [1000] * config.NUM_PLAYERS
        info_tensor = state.information_state_tensor(player_id)
        board_cards = [int(c) for i, c in enumerate(info_tensor[52:]) if c >= 0]
        num_board_cards = len(board_cards)
        stage = [0] * 4
        if num_board_cards == 0:
            stage[0] = 1
        elif num_board_cards == 3:
            stage[1] = 1
        elif num_board_cards == 4:
            stage[2] = 1
        elif num_board_cards == 5:
            stage[3] = 1
        opponent_stats = opponent_stats if opponent_stats is not None else OpponentStats()
        state_tensor = torch.FloatTensor(self.processor.process([state], [player_id], [bets], [stacks], [stage], opponent_stats)).to(device)
    
        with torch.no_grad():
            regrets = self.regret_net(state_tensor)[0]
            strategy_logits = self.strategy_net(state_tensor)[0]
            legal_mask = torch.zeros(self.num_actions, device=device)
            legal_mask[legal_actions] = 1
            strategy_logits = strategy_logits.masked_fill(legal_mask == 0, -1e9)
            strategy = torch.softmax(strategy_logits, dim=0).cpu().numpy()
    
        if len(self.cumulative_regrets) >= self.max_dict_size:
            self.cumulative_regrets.pop(next(iter(self.cumulative_regrets)))
            self.cumulative_strategies.pop(next(iter(self.cumulative_strategies)))
        
        if info_state not in self.cumulative_regrets:
            self.cumulative_regrets[info_state] = np.zeros(self.num_actions)
            self.cumulative_strategies[info_state] = np.zeros(self.num_actions)
        self.cumulative_regrets[info_state][legal_actions] += regrets[legal_actions].cpu().numpy()
        positive_regrets = np.maximum(self.cumulative_regrets[info_state][legal_actions], 0)
        total_regret = positive_regrets.sum()
        probs = positive_regrets / total_regret if total_regret > 0 else strategy[legal_actions] / strategy[legal_actions].sum()
        self.cumulative_strategies[info_state][legal_actions] += probs
        
        if len(self.strategy_pool) < 5 and random.random() < 0.1:
            self.strategy_pool.append({'weights': copy.deepcopy(self.strategy_net.state_dict()), 'winrate': 0.0})
        
        probs_dict = {a: float(p) for a, p in zip(legal_actions, probs)}
        logging.debug(f"Action probabilities for player {player_id}: {probs_dict}")
        return probs_dict

    def step(self, state, player_id: int, bets: List[float], stacks: List[float], stage: List[int], opponent_stats: OpponentStats) -> int:
        epsilon = max(0.1, 1.0 - self.global_step / 100000)
        probs = self.action_probabilities(state, player_id, opponent_stats)
        action = random.choice(list(probs.keys())) if random.random() < epsilon else max(probs, key=probs.get)
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
        logging.debug(f"Step for player {player_id}: action={action}, epsilon={epsilon}")
        return action

    def update_strategy_pool(self, winrate: float):
        for strat in self.strategy_pool:
            if strat['winrate'] == 0.0:
                strat['winrate'] = winrate
        if winrate > self.best_winrate * 0.95:
            self.strategy_pool.append({'weights': copy.deepcopy(self.strategy_net.state_dict()), 'winrate': winrate})
            self.best_winrate = max(self.best_winrate, winrate)
        if len(self.strategy_pool) > 10:
            self.strategy_pool = sorted(self.strategy_pool, key=lambda x: x['winrate'], reverse=True)[:10]
        logging.debug(f"Strategy pool updated, size={len(self.strategy_pool)}, best_winrate={self.best_winrate}")

# ===== СБОР ДАННЫХ =====
@ray.remote(num_cpus=1, num_gpus=0.5)
def collect_experience(game, agent: PokerAgent, processor: StateProcessor, steps: int, worker_id: int, queue=None):
    try:
        start_time = time.time()
        env = game.new_initial_state()
        logging.debug(f"Worker {worker_id}: Initial state created, terminal={env.is_terminal()}")

        if env.is_terminal():
            logging.error(f"Worker {worker_id}: Initial state is terminal, returns={env.returns()}")
            return ([], OpponentStats(), 0)

        agents = [agent] + [copy.deepcopy(agent) for _ in range(config.NUM_PLAYERS - 1)]
        experiences = []
        opponent_stats = OpponentStats()

        for step in range(steps):
            if env.is_terminal():
                returns = env.returns()
                for pid in range(config.NUM_PLAYERS):
                    pos = (pid - (env.current_player() if not env.is_terminal() else 0) + config.NUM_PLAYERS) % config.NUM_PLAYERS
                    opponent_stats.update(pid, 0, [0, 0, 0, 0], sum(env.bets()), 0, False, pos, won=returns[pid])
                env = game.new_initial_state()
                logging.debug(f"Worker {worker_id}: Reset state at step {step}")
                continue

            player_id = env.current_player()
            if player_id < 0:
                legal_actions = env.legal_actions()
                action = random.choice(legal_actions)
                next_state = env.clone()
                next_state.apply_action(action)
                env = next_state
                logging.debug(f"Worker {worker_id}: Chance node at step {step}, action={action}")
                continue

            bets = env.bets() if hasattr(env, 'bets') else [0] * config.NUM_PLAYERS
            stacks = env.stacks() if hasattr(env, 'stacks') else [1000] * config.NUM_PLAYERS
            
            info_tensor = env.information_state_tensor(player_id)
            board_cards = [int(c) for i, c in enumerate(info_tensor[52:]) if c >= 0]
            num_board_cards = len(board_cards)
            stage = [0] * 4
            if num_board_cards == 0:
                stage[0] = 1
            elif num_board_cards == 3:
                stage[1] = 1
            elif num_board_cards == 4:
                stage[2] = 1
            elif num_board_cards == 5:
                stage[3] = 1

            action = agents[player_id].step(env, player_id, bets, stacks, stage, opponent_stats)
            next_state = env.clone()
            next_state.apply_action(action)

            reward = next_state.returns()[player_id] if next_state.is_terminal() else agents[player_id]._heuristic_reward(action, env, player_id, bets, opponent_stats)
            experiences.append((env.clone(), action, reward, next_state.clone(), next_state.is_terminal(), player_id, bets[:], stacks[:], stage[:]))
            env = next_state

            if step % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        elapsed_time = time.time() - start_time
        hands_per_sec = steps / elapsed_time
        logging.info(f"Worker {worker_id} collected {len(experiences)} experiences at {hands_per_sec:.2f} hands/sec")
        return (experiences, opponent_stats, hands_per_sec)
    except Exception as e:
        logging.error(f"Worker {worker_id} failed: {traceback.format_exc()}")
        return ([], OpponentStats(), 0)

# ===== ТЕСТИРОВАНИЕ =====
class TightAggressiveAgent(pyspiel.Policy):
    def __init__(self, game):
        super().__init__()
        self.game = game
        self.player_ids = list(range(game.num_players()))

    def _hand_strength(self, state, player_id: int) -> float:
        info = state.information_state_tensor(player_id)
        cards = [int(i) for i, c in enumerate(info[:52]) if c > 0][:2]
        if len(cards) != 2:
            return 0.1
        ranks = sorted([c % 13 for c in cards], reverse=True)
        is_suited = len(set([c // 13 for c in cards])) == 1
        if ranks[0] >= 10 or (ranks[0] == ranks[1]):
            return 0.7 if is_suited else 0.6
        return 0.2 if is_suited else 0.1

    def action_probabilities(self, state, player_id: Optional[int] = None) -> Dict[int, float]:
        legal_actions = state.legal_actions(player_id)
        if not legal_actions:
            return {0: 1.0}
        strength = self._hand_strength(state, player_id)
        info_tensor = state.information_state_tensor(player_id)
        board_cards = [int(c) for i, c in enumerate(info_tensor[52:]) if c >= 0]
        num_board_cards = len(board_cards)
        stage = 0
        if num_board_cards == 0:
            stage = 0
        elif num_board_cards == 3:
            stage = 1
        elif num_board_cards == 4:
            stage = 2
        elif num_board_cards == 5:
            stage = 3
        bets = state.bets() if hasattr(state, 'bets') else [0] * config.NUM_PLAYERS
        pot = sum(bets)
        probs = {a: 0.0 for a in legal_actions}
        if strength > 0.5:
            if 3 in legal_actions and stage < 2:
                probs[3] = 0.8
                probs[1] = 0.2
            elif 2 in legal_actions and pot > 0:
                probs[2] = 0.7
                probs[1] = 0.3
            else:
                probs[1] = 1.0
        else:
            probs[0] = 0.7 if pot > 0 else 0.3
            probs[1] = 1.0 - probs[0]
        return probs

class LooseAggressiveAgent(policy.Policy):
    def __init__(self, game):
        super().__init__(game, list(range(game.num_players())))
        self.game = game

    def _hand_strength(self, state, player_id: int) -> float:
        info = state.information_state_tensor(player_id)
        cards = [int(i) for i, c in enumerate(info[:52]) if c > 0][:2]
        if len(cards) != 2:
            return 0.1
        ranks = sorted([c % 13 for c in cards], reverse=True)
        is_suited = len(set([c // 13 for c in cards])) == 1
        return 0.5 if is_suited or ranks[0] >= 8 else 0.3

    def action_probabilities(self, state, player_id: Optional[int] = None) -> Dict[int, float]:
        legal_actions = state.legal_actions(player_id)
        if not legal_actions:
            return {0: 1.0}
        strength = self._hand_strength(state, player_id)
        probs = {a: 0.0 for a in legal_actions}
        if strength > 0.3 or random.random() < 0.3:
            if 3 in legal_actions:
                probs[3] = 0.7
                probs[1] = 0.3
            elif 2 in legal_actions:
                probs[2] = 0.6
                probs[1] = 0.4
            else:
                probs[1] = 1.0
        else:
            probs[0] = 0.5
            probs[1] = 0.5
        return probs

def run_tournament(game, agent: PokerAgent, processor: StateProcessor, num_games: int = 50, compute_exploitability: bool = False) -> Tuple[float, float]:
    from open_spiel.python.algorithms import exploitability, cfr
    num_games = 200 if config.NUM_EPISODES > 10 else 50
    total_reward_vs_pool = 0
    total_reward_vs_tag = 0
    total_reward_vs_lag = 0
    total_reward_vs_random = 0
    tag_agent = TightAggressiveAgent(game)
    lag_agent = LooseAggressiveAgent(game)
    cfr_agent = cfr.CFRSolver(game)
    winrates = {'Pool': [], 'TAG': [], 'LAG': [], 'Random': [], 'CFR': []}

    for _ in range(num_games):
        state = game.new_initial_state()
        opponents = [PokerAgent(game, processor) for _ in range(config.NUM_PLAYERS - 1)]
        for opp in opponents:
            if agent.strategy_pool:
                opp.strategy_net.load_state_dict(random.choice([s['weights'] for s in agent.strategy_pool]))
        while not state.is_terminal():
            player_id = state.current_player()
            bets = state.bets() if hasattr(state, 'bets') else [0] * config.NUM_PLAYERS
            stacks = state.stacks() if hasattr(state, 'stacks') else [1000] * config.NUM_PLAYERS
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
            opponent_stats = OpponentStats()
            action = agent.step(state, 0, bets, stacks, stage, opponent_stats) if player_id == 0 else opponents[player_id - 1].step(state, player_id, bets, stacks, stage, opponent_stats)
            state.apply_action(action)
        total_reward_vs_pool += state.returns()[0]
    winrates['Pool'].append(total_reward_vs_pool / num_games)

    for _ in range(num_games):
        state = game.new_initial_state()
        tag_opponents = [tag_agent] * (config.NUM_PLAYERS - 1)
        while not state.is_terminal():
            player_id = state.current_player()
            bets = state.bets() if hasattr(state, 'bets') else [0] * config.NUM_PLAYERS
            stacks = state.stacks() if hasattr(state, 'stacks') else [1000] * config.NUM_PLAYERS
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
            opponent_stats = OpponentStats()
            action = agent.step(state, 0, bets, stacks, stage, opponent_stats) if player_id == 0 else max(tag_opponents[player_id - 1].action_probabilities(state, player_id), key=lambda x: tag_opponents[player_id - 1].action_probabilities(state, player_id)[x])
            state.apply_action(action)
        total_reward_vs_tag += state.returns()[0]
    winrates['TAG'].append(total_reward_vs_tag / num_games)

    for _ in range(num_games):
        state = game.new_initial_state()
        lag_opponents = [lag_agent] * (config.NUM_PLAYERS - 1)
        while not state.is_terminal():
            player_id = state.current_player()
            bets = state.bets() if hasattr(state, 'bets') else [0] * config.NUM_PLAYERS
            stacks = state.stacks() if hasattr(state, 'stacks') else [1000] * config.NUM_PLAYERS
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
            opponent_stats = OpponentStats()
            action = agent.step(state, 0, bets, stacks, stage, opponent_stats) if player_id == 0 else max(lag_opponents[player_id - 1].action_probabilities(state, player_id), key=lambda x: lag_opponents[player_id - 1].action_probabilities(state, player_id)[x])
            state.apply_action(action)
        total_reward_vs_lag += state.returns()[0]
    winrates['LAG'].append(total_reward_vs_lag / num_games)

    for _ in range(num_games):
        state = game.new_initial_state()
        while not state.is_terminal():
            player_id = state.current_player()
            bets = state.bets() if hasattr(state, 'bets') else [0] * config.NUM_PLAYERS
            stacks = state.stacks() if hasattr(state, 'stacks') else [1000] * config.NUM_PLAYERS
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
            opponent_stats = OpponentStats()
            action = agent.step(state, 0, bets, stacks, stage, opponent_stats) if player_id == 0 else random.choice(state.legal_actions(player_id))
            state.apply_action(action)
        total_reward_vs_random += state.returns()[0]
    winrates['Random'].append(total_reward_vs_random / num_games)

    total_reward_vs_cfr = 0
    for _ in range(num_games // 2):
        state = game.new_initial_state()
        while not state.is_terminal():
            player_id = state.current_player()
            bets = state.bets() if hasattr(state, 'bets') else [0] * config.NUM_PLAYERS
            stacks = state.stacks() if hasattr(state, 'stacks') else [1000] * config.NUM_PLAYERS
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
            opponent_stats = OpponentStats()
            action = agent.step(state, 0, bets, stacks, stage, opponent_stats) if player_id == 0 else cfr_agent.action(state, player_id)
            state.apply_action(action)
        total_reward_vs_cfr += state.returns()[0]
    winrates['CFR'].append(total_reward_vs_cfr / (num_games // 2))

    winrate_vs_pool = total_reward_vs_pool / num_games
    exp_score = exploitability.exploitability(game, agent) if compute_exploitability else 0.0
    agent.update_strategy_pool(winrate_vs_pool)
    for key, values in winrates.items():
        writer.add_scalar(f'Winrate/{key}', np.mean(values), agent.global_step)
        writer.add_scalar(f'Winrate/{key}_Std', np.std(values), agent.global_step)
    writer.add_scalar('Exploitability', exp_score, agent.global_step)
    logging.info(f"Tournament: Pool={winrate_vs_pool:.4f}, Exploitability={exp_score:.4f}")
    return winrate_vs_pool, exp_score

# ===== ОБУЧЕНИЕ =====
class Trainer:
    def __init__(self, game, agent: PokerAgent, processor: StateProcessor):
        self.game = game
        self.agent = agent
        self.processor = processor
        self.buffer = PrioritizedReplayBuffer(config.BUFFER_CAPACITY)
        self.reward_normalizer = RewardNormalizer()
        self.global_step = 0
        self.best_winrate = -float('inf')
        self.interrupted = False
        self.last_checkpoint_time = time.time()
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.agent.optimizer, mode='max', factor=0.5, patience=5)
        self.opponent_stats = OpponentStats()

    def _train_step(self, batch):
        try:
            samples, indices, weights = batch
            states, actions, rewards, next_states, dones, _, _, _, _ = zip(*samples)
        
            states = torch.tensor(states, dtype=torch.float32, device=device)
            actions = torch.tensor(actions, dtype=torch.long, device=device)
            rewards = torch.tensor([self.reward_normalizer.normalize(r) for r in rewards], dtype=torch.float32, device=device)
            next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
            dones = torch.tensor(dones, dtype=torch.float32, device=device)
            weights = torch.tensor(weights, dtype=torch.float32, device=device)
        
            self.agent.optimizer.zero_grad()
            q_values = self.agent.regret_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = self.agent.strategy_net(next_states).max(1)[0]
            targets = rewards + config.GAMMA * next_q_values * (1 - dones)
            loss = (weights * (q_values - targets.detach()) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.regret_net.parameters(), config.GRAD_CLIP_VALUE)
            self.agent.optimizer.step()
        
            new_priorities = torch.abs(q_values - targets).detach().cpu().numpy() + 1e-5
            self.buffer.update_priorities(indices, new_priorities)
            return loss.item()
        except Exception as e:
            logging.error(f"Ошибка в _train_step: {traceback.format_exc()}")
            raise

    def _save_checkpoint(self, path, is_best=False):
        checkpoint = {
            'regret_net': self.agent.regret_net.state_dict(),
            'strategy_net': self.agent.strategy_net.state_dict(),
            'optimizer': self.agent.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_winrate': self.best_winrate,
            'opponent_stats': self.opponent_stats.stats,
        }
        torch.save(checkpoint, path)
        self.last_checkpoint_time = time.time()
        logging.info(f"Checkpoint saved to {path}")

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.agent.regret_net.load_state_dict(checkpoint['regret_net'])
        self.agent.strategy_net.load_state_dict(checkpoint['strategy_net'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer'])
        self.global_step = checkpoint['global_step']
        self.best_winrate = checkpoint['best_winrate']
        self.opponent_stats.stats.update(checkpoint['opponent_stats'])
        self.agent.global_step = self.global_step
        logging.info(f"Checkpoint loaded from {path}")

    def train(self):
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
    
        try:
            for episode in range(self.global_step // config.STEPS_PER_WORKER, config.NUM_EPISODES):
                logging.info(f"Starting episode {episode}")
                futures = [collect_experience.remote(self.game, self.agent, self.processor, config.STEPS_PER_WORKER, i) 
                           for i in range(config.NUM_WORKERS)]
                results = ray.get(futures)
    
                all_experiences = []
                for worker_id, (experiences, local_opp_stats, hands_per_sec) in enumerate(results):
                    if not experiences:
                        logging.warning(f"Episode {episode}, Worker {worker_id}: No experiences collected")
                        continue
                    all_experiences.extend(experiences)
                    self.opponent_stats.stats.update(local_opp_stats.stats)
                    writer.add_scalar(f'HandsPerSec/Worker_{worker_id}', hands_per_sec, self.global_step)
    
                if all_experiences:
                    self.buffer.add_batch(all_experiences, self.processor)
                    for exp in all_experiences:
                        self.reward_normalizer.update(exp[2])
                        self.global_step += 1
                        self.agent.global_step = self.global_step
                        self.buffer.global_step = self.global_step
                        if exp[5] == 0:
                            is_blind = exp[5] in [0, 1] and np.argmax(exp[8]) == 0 and sum(exp[6]) <= 0.15
                            pos = (exp[5] - exp[0].current_player() + config.NUM_PLAYERS) % config.NUM_PLAYERS
                            is_raise = any(b > 0 for i, b in enumerate(exp[6]) if i != exp[5])
                            agent_stats.update(0, exp[1], exp[8], sum(exp[6]), exp[1] if exp[1] > 1 else 0, is_blind, pos, is_raise=is_raise)
    
                if len(self.buffer) >= config.BATCH_SIZE:
                    loss = self._train_step(self.buffer.sample(config.BATCH_SIZE))
                    logging.info(f"Step {self.global_step} | Loss: {loss:.4f}")
    
                if self.global_step % config.TEST_INTERVAL == 0:
                    winrate, exp_score = run_tournament(self.game, self.agent, self.processor)
                    agent_metrics = agent_stats.get_metrics(0)
                    writer.add_scalar('Winrate', winrate, self.global_step)
                    writer.add_scalar('Agent_VPIP', agent_metrics['vpip'], self.global_step)
                    self.lr_scheduler.step(winrate)
                    if winrate > self.best_winrate:
                        self.best_winrate = winrate
                        self._save_checkpoint(config.BEST_MODEL_PATH, is_best=True)
    
                if time.time() - self.last_checkpoint_time >= config.CHECKPOINT_INTERVAL:
                    self._save_checkpoint(config.MODEL_PATH)
    
                gc.collect()
                torch.cuda.empty_cache()
    
                pbar.update(1)
                logging.info(f"Episode {episode} completed, global_step={self.global_step}")
                if self.interrupted:
                    break
    
            self._save_checkpoint(config.MODEL_PATH)
        except Exception as e:
            last_exp = all_experiences[-1] if 'all_experiences' in locals() and all_experiences else 'N/A'
            error_msg = f"Training crashed: {traceback.format_exc()}\nLast experiences: {last_exp}"
            logging.error(error_msg)
            with open(os.path.join(config.LOG_DIR, 'crash_details.txt'), 'a') as f:
                f.write(f"{time.ctime()}: {error_msg}\n")
            self._save_checkpoint(os.path.join(config.LOG_DIR, 'crash_recovery.pt'))
            ray.shutdown()
            sys.exit(1)
    
        pbar.close()
        writer.close()
        ray.shutdown()

# ===== ЗАПУСК =====
if __name__ == "__main__":
    mp.set_start_method('spawn')
    config = Config()

    if ray.is_initialized():
        print("Ray уже инициализирован другой сессией. Завершаем новый запуск.")
        sys.exit(1)

    ray.init(
        num_gpus=1,
        num_cpus=8,
        object_store_memory=2 * 1024 * 1024 * 1024,
        _memory=4 * 1024 * 1024 * 1024,
        ignore_reinit_error=True,
        configure_logging=False,
        _system_config={
            "object_spilling_threshold": 0.8,
            "max_io_workers": 1,
        }
    )

    game = pyspiel.load_game(config.GAME_NAME)
    processor = StateProcessor()
    agent = PokerAgent(game, processor)
    trainer = Trainer(game, agent, processor)
    trainer.train()
