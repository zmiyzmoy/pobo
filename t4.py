# Импорт всех необходимых библиотек
import os
import time
import logging
import sys
import signal
import random
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
        self.NUM_EPISODES = 1  # Для теста, увеличьте до 50 для A100
        self.BATCH_SIZE = 512  # Увеличьте до 2048 для A100
        self.GAMMA = 0.96
        self.BUFFER_CAPACITY = 100_000  # Увеличьте до 1M для A100
        self.NUM_WORKERS = 1  # Увеличьте до 8 для A100
        self.STEPS_PER_WORKER = 1000  # Увеличьте до 2500 для 1M рук
        self.SELFPLAY_UPDATE_FREQ = 500
        self.LOG_FREQ = 100
        self.TEST_INTERVAL = 500
        self.LEARNING_RATE = 1e-4
        self.NUM_PLAYERS = 6
        self.GRAD_CLIP_VALUE = 5.0
        self.CHECKPOINT_INTERVAL = 300  # 3600 для A100
        self.NUM_BUCKETS = 50
        self.BB = 0.10
        self.GAME_NAME = "universal_poker(betting=nolimit,numPlayers=6,numRounds=4,blind=0.05 0.10,raiseSize=0.10 0.20 0.40 0.80,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1)"

# Инициализация
config = Config()
os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
os.makedirs(config.LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
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
        self.rank_embed = nn.Embedding(13, 8)
        self.suit_embed = nn.Embedding(4, 4)
        logging.info("CardEmbedding инициализирован")

    def forward(self, cards: List[int]) -> torch.Tensor:
        ranks = torch.tensor([c % 13 for c in cards], dtype=torch.long).to(device)
        suits = torch.tensor([c // 13 for c in cards], dtype=torch.long).to(device)
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
            'vpip': self.stats[player_id]['vpip'] / hands,
            'pfr': self.stats[player_id]['pfr'] / hands,
            'af': self.stats[player_id]['af'],
            'fold_freq': self.stats[player_id]['folds'] / hands,
            'last_bet': self.stats[player_id]['last_bet'],
            'fold_to_cbet': self.stats[player_id]['fold_to_cbet'] / cbet_opp,
            'fold_to_3bet': self.stats[player_id]['fold_to_3bet'] / threebet_opp,
            'check_raise_freq': self.stats[player_id]['check_raise'] / check_opp,
            'call_vs_raise_freq': self.stats[player_id]['call_vs_raise_freq'] / raise_opp,
            'street_aggression': [agg / max(hands, 1) for agg in self.stats[player_id]['street_aggression']],
            'pos_winrate': pos_stats
        }

# ===== ОБРАБОТКА СОСТОЯНИЙ =====
class StateProcessor:
    def __init__(self):
        self.card_embedding = CardEmbedding().to(device)
        self.buckets = self._load_or_precompute_buckets()
        self.state_size = config.NUM_BUCKETS + config.NUM_PLAYERS * 3 + 4 + 5
        self.cache = lru_cache(maxsize=10000)
        logging.info("StateProcessor инициализирован")

    def _load_or_precompute_buckets(self):
        try:
            if os.path.exists(config.KMEANS_PATH):
                buckets = joblib.load(config.KMEANS_PATH)
                logging.info(f"Загружены бакеты из {config.KMEANS_PATH}")
                return buckets
        except Exception as e:
            logging.warning(f"Не удалось загрузить KMeans из {config.KMEANS_PATH}: {e}, пересчитываем")
        try:
            all_hands = []
            for r1 in range(13):
                for r2 in range(r1, 13):
                    for suited in [0, 1]:
                        hand = [r1 + s1 * 13 for s1 in range(4)] + [r2 + s2 * 13 for s2 in range(4)]
                        embedding = self.card_embedding(hand[:2]).cpu().detach().numpy()
                        all_hands.append(embedding)
            kmeans = KMeans(n_clusters=config.NUM_BUCKETS, random_state=42).fit(all_hands)
            joblib.dump(kmeans, config.KMEANS_PATH)
            logging.info(f"Бакеты вычислены и сохранены в {config.KMEANS_PATH}")
            return kmeans
        except Exception as e:
            logging.error(f"Не удалось вычислить KMeans: {traceback.format_exc()}")
            raise

    def augment_cards(self, cards: List[int]) -> List[int]:
        original_ranks = [c % 13 for c in cards]
        original_suits = [c // 13 for c in cards]
        is_suited = len(set(original_suits)) == 1
        if is_suited:
            new_suit = random.randint(0, 3)
            return [r + new_suit * 13 for r in original_ranks]
        else:
            suit_mapping = {s: random.randint(0, 3) for s in set(original_suits)}
            return [r + suit_mapping[s] * 13 for r, s in zip(original_ranks, original_suits)]

        def process(self, states: List, player_ids: List[int], bets: List[List[float]], stacks: List[List[float]], stages: List[List[int]], 
                opponent_stats: Optional[OpponentStats] = None) -> np.ndarray:
            batch_size = len(states)
            state_keys = [f"{s.information_state_string(pid)}_{pid}_{tuple(b)}_{tuple(stk)}" 
                          for s, pid, b, stk in zip(states, player_ids, bets, stacks)]
            cached = [self.cache.__get__(key, None) for key in state_keys]
            if all(c is not None for c in cached):
                logging.debug(f"Использован кэш для {batch_size} состояний")
                return np.array(cached)
    
            info_states = [s.information_state_tensor(pid) for s, pid in zip(states, player_ids)]
            cards_batch = [[int(c) for i, c in enumerate(info[:52]) if c >= 0 and i in s.player_cards(pid)] 
                           for info, s, pid in zip(info_states, states, player_ids)]
            if random.random() < 0.5:
                cards_batch = [self.augment_cards(cards) for cards in cards_batch]
            card_embs = torch.stack([self.card_embedding(cards) for cards in cards_batch]).cpu().detach().numpy()
            bucket_idxs = self.buckets.predict(card_embs)
            bucket_one_hot = np.zeros((batch_size, config.NUM_BUCKETS))
            bucket_one_hot[np.arange(batch_size), bucket_idxs] = 1.0
    
            bets_norm = np.array(bets) / (np.array(stacks) + 1e-8)
            stacks_norm = np.array(stacks) / 1000.0
            pots = [sum(b) for b in bets]
            sprs = [stk[pid] / pot if pot > 0 else 10.0 for stk, pid, pot in zip(stacks, player_ids, pots)]
            positions = [(pid - s.current_player()) % config.NUM_PLAYERS / config.NUM_PLAYERS for s, pid in zip(states, player_ids)]
    
            action_history = [([0] * config.NUM_PLAYERS if not hasattr(s, 'action_history') else 
                              [min(h, 4) for h in s.action_history()[-config.NUM_PLAYERS:]]) for s in states]
            action_history = np.array(action_history)
    
            opponent_metrics = [[opponent_stats.get_metrics(i) for i in range(config.NUM_PLAYERS) if i != pid] 
                               if opponent_stats else [] for pid in player_ids]
            table_aggs = [np.mean([m['af'] for m in metrics]) if metrics else 0.5 for metrics in opponent_metrics]
            last_bets = [max([m['last_bet'] for m in metrics]) / pot if pot > 0 and metrics else 0.0 
                         for metrics, pot in zip(opponent_metrics, pots)]
            all_in_flags = [1.0 if any(b >= stk[i] for i, b in enumerate(bet) if i != pid) else 0.0 
                            for bet, stk, pid in zip(bets, stacks, player_ids)]
    
            processed = np.concatenate([
                bucket_one_hot, bets_norm, stacks_norm, action_history, np.array(stages),
                np.array([sprs, table_aggs, positions, last_bets, all_in_flags]).T
            ], axis=1)
    
            if np.any(np.isnan(processed)) or np.any(np.isinf(processed)):
                problematic_idx = np.where(np.isnan(processed) | np.isinf(processed))[0][0]
                logging.error(f"Invalid state detected: NaN or Inf in processed data\n"
                              f"Batch size: {batch_size}\n"
                              f"Problematic index: {problematic_idx}\n"
                              f"State: {states[problematic_idx].information_state_string(player_ids[problematic_idx])}\n"
                              f"Bets: {bets[problematic_idx]}\nStacks: {stacks[problematic_idx]}\n"
                              f"Processed row: {processed[problematic_idx]}\n"
                              f"Card embeddings: {card_embs[problematic_idx]}")
                raise ValueError("Invalid state processing detected")
    
            for key, proc in zip(state_keys, processed):
                self.cache.__set__(key, proc)
            return processed
# ===== АГЕНТ =====
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
        self.best_winrate = -float('inf')
        self.global_step = 0

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
        stage_idx = state.round()
        board_cards = [int(c) for i, c in enumerate(state.information_state_tensor(player_id)[52:]) if c >= 0]
        is_drawy = len(set([c // 13 for c in board_cards])) < 3 if board_cards else False

        if action == 0 and pot > 0:
            return -0.1 * (1 + pot / config.BB) * (1.5 if is_drawy else 1.0)
        elif action > 1:
            if any(m['fold_to_3bet'] > 0.7 for m in opp_metrics) and stage_idx == 0:
                return 0.3 * hand_strength
            if any(m['street_aggression'][stage_idx] < 0.2 for m in opp_metrics):
                return 0.2 + 0.1 * hand_strength * (1.2 if is_drawy else 1.0)
            return 0.1 * hand_strength
        elif action == 1 and any(m['last_bet'] > pot * 0.5 for m in opp_metrics):
            return 0.1 * hand_strength if random.random() < 0.5 else -0.1
        return 0.0

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
        return base

    def action_probabilities(self, state, player_id: Optional[int] = None) -> Dict[int, float]:
        legal_actions = state.legal_actions(player_id)
        if not legal_actions:
            return {0: 1.0}
        info_state = state.information_state_string(player_id)
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
            self.strategy_pool.append({'weights': copy.deepcopy(self.strategy_net.state_dict()), 'winrate': 0.0})
        return {a: float(p) for a, p in zip(legal_actions, probs)}

    def step(self, state, player_id: int, bets: List[float], stacks: List[float], stage: List[int], opponent_stats: OpponentStats) -> int:
        epsilon = max(0.1, 1.0 - self.global_step / 100000)
        probs = self.action_probabilities(state, player_id)
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

# ===== СБОР ДАННЫХ =====
@ray.remote
def collect_experience(game, agent: PokerAgent, processor: StateProcessor, steps: int, worker_id: int, queue: Queue):
    try:
        start_time = time.time()
        env = game.new_initial_state()
        agents = [agent] + [PokerAgent(game, processor) for _ in range(config.NUM_PLAYERS - 1)]
        for i, opp in enumerate(agents[1:], 1):
            if agent.strategy_pool and random.random() < 0.5:
                opp.strategy_net.load_state_dict(random.choice([s['weights'] for s in agent.strategy_pool]))
        experiences = []
        opponent_stats = OpponentStats()

        for _ in range(steps):
            if env.is_terminal():
                returns = env.returns()
                for pid in range(config.NUM_PLAYERS):
                    pos = (pid - env.current_player() + config.NUM_PLAYERS) % config.NUM_PLAYERS
                    opponent_stats.update(pid, 0, [0, 0, 0, 0], sum(env.bets()), 0, False, pos, won=returns[pid])
                env = game.new_initial_state()
            player_id = env.current_player()
            bets = env.bets() if hasattr(env, 'bets') else [0] * config.NUM_PLAYERS
            stacks = env.stacks() if hasattr(env, 'stacks') else [1000] * config.NUM_PLAYERS
            stage = [1 if env.round() == i else 0 for i in range(4)]
            is_blind = player_id in [0, 1] and env.round() == 0 and sum(bets) <= 0.15
            position = (player_id - env.current_player() + config.NUM_PLAYERS) % config.NUM_PLAYERS
            is_cbet = env.round() == 1 and sum(bets) > 0 and env.current_player() == 0
            is_3bet = env.round() == 0 and any(b > config.BB for b in bets) and sum(bets) > 2 * config.BB
            is_check = sum(bets) == 0 and env.round() > 0
            is_raise = any(b > 0 for i, b in enumerate(bets) if i != player_id)

            action = agents[player_id].step(env, player_id, bets, stacks, stage, opponent_stats)
            next_state = env.clone()
            next_state.apply_action(action)

            final_reward = next_state.returns()[player_id]
            heuristic_reward = agents[player_id]._heuristic_reward(action, env, player_id, bets, opponent_stats)
            reward = heuristic_reward if not next_state.is_terminal() else final_reward

            experiences.append((env, action, reward, next_state, next_state.is_terminal(), player_id, bets, stacks, stage))
            opponent_stats.update(player_id, action, stage, sum(bets), action if action > 1 else 0, is_blind, position, is_cbet, is_3bet, is_check, is_raise=is_raise)
            env = next_state

        elapsed_time = time.time() - start_time
        hands_per_sec = steps / elapsed_time
        queue.put((experiences, opponent_stats, hands_per_sec))
        logging.info(f"Worker {worker_id} collected {len(experiences)} experiences at {hands_per_sec:.2f} hands/sec")
    except Exception as e:
        logging.error(f"Worker {worker_id} failed: {traceback.format_exc()}\nLast state: {env.information_state_string()}\nBets: {bets}\nStacks: {stacks}")
        queue.put(([], OpponentStats(), 0))

# ===== ТЕСТИРОВАНИЕ =====
class TightAggressiveAgent(policy.Policy):
    def __init__(self, game):
        super().__init__(game)
        self.game = game

    def _hand_strength(self, state, player_id: int) -> float:
        info = state.information_state_tensor(player_id)
        cards = [int(c) for i, c in enumerate(info[:52]) if c >= 0 and i in state.player_cards(player_id)]
        ranks = sorted([c % 13 for c in cards], reverse=True)
        is_suited = len(set([c // 13 for c in cards])) == 1
        if len(ranks) < 2:
            return 0.1
        if ranks[0] >= 10 or (ranks[0] == ranks[1]):
            return 0.7 if is_suited else 0.6
        return 0.2 if is_suited else 0.1

    def action_probabilities(self, state, player_id: Optional[int] = None) -> Dict[int, float]:
        legal_actions = state.legal_actions(player_id)
        if not legal_actions:
            return {0: 1.0}
        strength = self._hand_strength(state, player_id)
        stage = state.round()
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
        super().__init__(game)
        self.game = game

    def _hand_strength(self, state, player_id: int) -> float:
        info = state.information_state_tensor(player_id)
        cards = [int(c) for i, c in enumerate(info[:52]) if c >= 0 and i in state.player_cards(player_id)]
        ranks = sorted([c % 13 for c in cards], reverse=True)
        is_suited = len(set([c // 13 for c in cards])) == 1
        if len(ranks) < 2:
            return 0.1
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
            stage = [1 if state.round() == i else 0 for i in range(4)]
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
            stage = [1 if state.round() == i else 0 for i in range(4)]
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
            stage = [1 if state.round() == i else 0 for i in range(4)]
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
            stage = [1 if state.round() == i else 0 for i in range(4)]
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
            stage = [1 if state.round() == i else 0 for i in range(4)]
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
    def __init__(self):
        self.game = game
        self.processor = StateProcessor()
        self.agent = PokerAgent(self.game, self.processor)
        self.regret_optimizer = optim.Adam(self.agent.regret_net.parameters(), lr=config.LEARNING_RATE)
        self.strategy_optimizer = optim.Adam(self.agent.strategy_net.parameters(), lr=config.LEARNING_RATE)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.strategy_optimizer, 'max', factor=0.5, patience=5)
        self.buffer = PrioritizedReplayBuffer()
        self.reward_normalizer = RewardNormalizer()
        self.opponent_stats = OpponentStats()
        self.global_step = 0
        self.best_winrate = -float('inf')
        self.interrupted = False
        self.last_checkpoint_time = time.time()
        self.agent.global_step = self.global_step
        self.buffer.global_step = self.global_step

    def _train_step(self, experiences, beta: float = 0.4):
        samples, indices, weights = self.buffer.sample(config.BATCH_SIZE, beta)
        if not samples:
            return 0.0
        states, actions, rewards, next_states, dones, player_ids, bets, stacks, stages = zip(*samples)
        
        states = torch.FloatTensor(np.array(states)).to(device, non_blocking=True)
        actions = torch.LongTensor(actions).to(device, non_blocking=True)
        rewards = torch.FloatTensor([self.reward_normalizer.normalize(r) for r in rewards]).to(device, non_blocking=True)
        next_states = torch.FloatTensor(np.array(next_states)).to(device, non_blocking=True)
        dones = torch.FloatTensor(dones).to(device, non_blocking=True)
        weights = torch.FloatTensor(weights).to(device, non_blocking=True)

        with autocast():
            regrets = self.agent.regret_net(states)
            strategies = self.agent.strategy_net(states)
            next_strategies = self.agent.strategy_net(next_states)
            expected_future_rewards = (torch.softmax(next_strategies, dim=1).max(dim=1)[0] * config.GAMMA * (1 - dones)).detach()
            target_rewards = rewards + expected_future_rewards

            regret_loss = (nn.MSELoss(reduction='none')(regrets.gather(1, actions.unsqueeze(1)), target_rewards.unsqueeze(1)) * weights.unsqueeze(1)).mean()
            strategy_loss = -torch.mean((torch.log_softmax(strategies, dim=1) * regrets.detach()) * weights.unsqueeze(1))

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

        td_errors = (target_rewards - regrets.gather(1, actions.unsqueeze(1))).abs().detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors.flatten() + 1e-6)
        writer.add_scalar('Loss', regret_loss.item() + strategy_loss.item(), self.global_step)
        return regret_loss.item() + strategy_loss.item()

    def _save_checkpoint(self, path: str, is_best: bool = False):
        checkpoint = {
            'regret_net': self.agent.regret_net.state_dict(),
            'strategy_net': self.agent.strategy_net.state_dict(),
            'regret_optimizer': self.regret_optimizer.state_dict(),
            'strategy_optimizer': self.strategy_optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'step': self.global_step,
            'reward_history': self.reward_normalizer.rewards,
            'strategy_pool': self.agent.strategy_pool
        }
        torch.save(checkpoint, path)
        if is_best:
            shutil.copyfile(path, config.BEST_MODEL_PATH)
        self.last_checkpoint_time = time.time()

    def _load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=device)
        self.agent.regret_net.load_state_dict(checkpoint['regret_net'])
        self.agent.strategy_net.load_state_dict(checkpoint['strategy_net'])
        self.regret_optimizer.load_state_dict(checkpoint['regret_optimizer'])
        self.strategy_optimizer.load_state_dict(checkpoint['strategy_optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.global_step = checkpoint['step']
        self.reward_normalizer.rewards = checkpoint['reward_history']
        self.reward_normalizer.mean = np.mean(self.reward_normalizer.rewards) if self.reward_normalizer.rewards else 0
        self.reward_normalizer.std = np.std(self.reward_normalizer.rewards) + self.reward_normalizer.eps if self.reward_normalizer.rewards else 1
        self.agent.strategy_pool = checkpoint['strategy_pool']

    def train(self):
    try:
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
            ray.init(num_gpus=1, num_cpus=config.NUM_WORKERS, ignore_reinit_error=True)
            queues = [Queue() for _ in range(config.NUM_WORKERS)]
            tasks = [collect_experience.remote(self.game, self.agent, self.processor, config.STEPS_PER_WORKER, i, queues[i])
                     for i in range(config.NUM_WORKERS)]
        except Exception as e:
            logging.warning(f"Ray initialization failed: {e}, falling back to single-threaded mode")
            def single_thread_collect():
                queue = Queue()
                collect_experience(self.game, self.agent, self.processor, config.STEPS_PER_WORKER * config.NUM_WORKERS, 0, queue)
                return [queue.get()]
            tasks = single_thread_collect

        for episode in range(self.global_step // config.STEPS_PER_WORKER, config.NUM_EPISODES):
            results = tasks() if callable(tasks) else ray.get(tasks)
            for experiences, local_opp_stats, hands_per_sec in results:
                if not experiences:
                    continue
                self.buffer.add_batch(experiences, self.processor)
                for exp in experiences:
                    self.reward_normalizer.update(exp[2])
                    self.global_step += 1
                    self.agent.global_step = self.global_step
                    self.buffer.global_step = self.global_step
                    if exp[5] == 0:
                        is_blind = exp[5] in [0, 1] and np.argmax(exp[8]) == 0 and sum(exp[6]) <= 0.15
                        pos = (exp[5] - exp[0].current_player() + config.NUM_PLAYERS) % config.NUM_PLAYERS
                        is_raise = any(b > 0 for i, b in enumerate(exp[6]) if i != exp[5])
                        agent_stats.update(0, exp[1], exp[8], sum(exp[6]), exp[1] if exp[1] > 1 else 0, is_blind, pos, is_raise=is_raise)
                self.opponent_stats.stats.update(local_opp_stats.stats)
                writer.add_scalar('HandsPerSec', hands_per_sec, self.global_step)

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

            pbar.update(1)
            if self.interrupted:
                break

        self._save_checkpoint(config.MODEL_PATH)
        ray.shutdown()
    except Exception as e:
        error_msg = f"Training crashed: {traceback.format_exc()}\nLast experiences: {experiences[-1] if 'experiences' in locals() else 'N/A'}"
        logging.error(error_msg)
        with open(os.path.join(config.LOG_DIR, 'crash_details.txt'), 'a') as f:
            f.write(f"{time.ctime()}: {error_msg}\n")
        self._save_checkpoint(os.path.join(config.LOG_DIR, 'crash_recovery.pt'))
        ray.shutdown()
        sys.exit(1)
    pbar.close()
    writer.close()

# ===== ЗАПУСК =====
if __name__ == "__main__":
    mp.set_start_method('spawn')
    trainer = Trainer()
    trainer.train()
