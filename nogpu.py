import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import os
import random
import collections
from collections import OrderedDict
import rlcard
from tqdm.auto import tqdm
from multiprocessing import Process, Queue
from rlcard.agents import RandomAgent
import time
import logging
import math
from treys import Evaluator, Card
from functools import lru_cache
from torch.utils.tensorboard import SummaryWriter
import shutil
import sys
import copy

# ========== КОНФИГУРАЦИЯ ==========
model_path = '/home/gunelmikayilova91/rlcard/pai.pt'
best_model_path = '/home/gunelmikayilova91/rlcard/pai_best.pt'
log_dir = '/home/gunelmikayilova91/rlcard/logs/'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'training.log')),
        logging.StreamHandler()
    ]
)
writer = SummaryWriter(log_dir=log_dir)

num_episodes = 10
batch_size = 512
gamma = 0.96
epsilon_start = 1.0
epsilon_end = 0.01
target_update_freq = 1000
buffer_capacity = 500_000
num_workers = 1
steps_per_worker = 1000
selfplay_update_freq = 5000
log_freq = 25
test_interval = 2000
early_stop_patience = 1000
early_stop_threshold = 0.001
learning_rate = 1e-4
noise_scale = 0.1
num_tables = 1
num_players = 6
code_version = "v1.9_test"
grad_clip_value = 5.0

device = torch.device("cpu")

# ========== НОРМАЛИЗАЦИЯ НАГРАД ==========
class RewardNormalizer:
    def __init__(self, eps=1e-8, init_mean=0, init_std=100):
        self.mean = init_mean
        self.std = init_std
        self.count = 1
        self.eps = eps

    def update(self, reward):
        self.count += 1
        old_mean = self.mean
        self.mean = (self.mean * (self.count - 1) + reward) / self.count
        self.std = math.sqrt(((self.std ** 2) * (self.count - 1) + (reward - old_mean) * (reward - self.mean)) / self.count)

    def normalize(self, reward):
        return (reward - self.mean) / (self.std + self.eps)

# ========== МОДЕЛЬ ==========
class DuelingPokerNN(nn.Module):
    def __init__(self, input_size, num_actions):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.3)
        self.value_fc = nn.Linear(512, 256)
        self.value_ln = nn.LayerNorm(256)
        self.value_out = nn.Linear(256, 1)
        self.adv_fc = nn.Linear(512, 256)
        self.adv_ln = nn.LayerNorm(256)
        self.adv_out = nn.Linear(256, num_actions)

    def forward(self, x):
        if x.shape[-1] != self.input_size:
            raise ValueError(f"Input size mismatch. Expected {self.input_size}, got {x.shape[-1]}")
        identity = x
        x = torch.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.ln2(self.fc2(x)) + identity[:, :512])
        value = self.value_out(torch.relu(self.value_ln(self.value_fc(x))))
        adv = self.adv_out(torch.relu(self.adv_ln(self.adv_fc(x))))
        return value + (adv - adv.mean(dim=1, keepdim=True))

# ========== БУФЕР ОПЫТА ==========
class PrioritizedExperienceBuffer:
    def __init__(self, capacity=buffer_capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.priorities = collections.deque(maxlen=capacity)
        self.visit_counts = collections.deque(maxlen=capacity)

    def add(self, experience, priority):
        self.buffer.append(experience)
        self.priorities.append(priority)
        self.visit_counts.append(0)

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities, dtype=np.float32) + 1e-5
        visits = np.array(self.visit_counts, dtype=np.float32) + 1
        adjusted_priorities = priorities / visits
        probs = adjusted_priorities ** 0.6 / adjusted_priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        for idx in indices:
            self.visit_counts[idx] += 1
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, torch.FloatTensor(weights).to(device)

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + 1e-5

    def merge(self, other_buffer):
        for exp, pri, vis in zip(other_buffer.buffer, other_buffer.priorities, other_buffer.visit_counts):
            if len(self.buffer) < self.buffer.maxlen:
                self.buffer.append(exp)
                self.priorities.append(pri)
                self.visit_counts.append(vis)

    def __len__(self):
        return len(self.buffer)

# ========== ОБРАБОТКА КАРТ ==========
def card_to_number(card_str):
    """Convert a card string (e.g., 'SJ') to a number (0-51)."""
    suits = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
    ranks = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, '10': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    
    if not isinstance(card_str, str) or len(card_str) < 2:
        logging.error(f"Invalid card format: {card_str}")
        return None
    
    suit = card_str[0].upper()
    rank = card_str[1:]
    if suit not in suits or rank not in ranks:
        logging.error(f"Invalid suit or rank in card {card_str}: suit={suit}, rank={rank}")
        return None
    
    return ranks[rank] + suits[suit] * 13

def number_to_card(card_num):
    """Convert a card number (0-51) back to a string (e.g., 'SJ')."""
    suits = {0: 'C', 1: 'D', 2: 'H', 3: 'S'}
    ranks = {0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 7: '9', 8: '10', 9: 'J', 10: 'Q', 11: 'K', 12: 'A'}
    
    if not (0 <= card_num <= 51):
        logging.error(f"Invalid card number: {card_num}")
        return None
    
    suit = card_num // 13
    rank = card_num % 13
    return suits[suit] + ranks[rank]

def number_to_treys_card(card_num):
    """Convert a card number (0-51) to treys format (e.g., 'Js' for Jack of Spades)."""
    suits = {0: 'c', 1: 'd', 2: 'h', 3: 's'}
    ranks = {0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 7: '9', 8: '10', 9: 'j', 10: 'q', 11: 'k', 12: 'a'}
    
    if not (0 <= card_num <= 51):
        logging.error(f"Invalid card number for treys conversion: {card_num}")
        return None
    
    suit = card_num // 13
    rank = card_num % 13
    return Card.new(f"{ranks[rank]}{suits[suit]}")

@lru_cache(maxsize=500)
def cached_evaluate(player_cards, community_cards):
    """Evaluate hand strength using treys library."""
    evaluator = Evaluator()
    if not player_cards or len(player_cards) < 2:
        logging.warning(f"Invalid player cards for evaluation: {player_cards}")
        return 0.1
    if not isinstance(community_cards, tuple):
        logging.warning(f"Invalid community cards for evaluation: {community_cards}")
        community_cards = tuple()
    
    try:
        for card in player_cards:
            if not isinstance(card, int) or card < 0 or card > 51:
                logging.error(f"Invalid player card value: {card}")
                return 0.5
        for card in community_cards:
            if not isinstance(card, int) or card < 0 or card > 51:
                logging.error(f"Invalid community card value: {card}")
                return 0.5

        player_treys = [number_to_treys_card(card) for card in player_cards]
        community_treys = [number_to_treys_card(card) for card in community_cards]
        score = evaluator.evaluate(list(community_treys), list(player_treys))
        normalized_score = 1.0 - (score / 7462.0)  # 7462 is the worst possible hand rank
        return normalized_score
    except Exception as e:
        logging.error(f"Error in cached_evaluate: {str(e)}")
        return 0.5

def extract_cards(state_dict, stage):
    """
    Extract and convert player and community cards from the state dictionary.
    Cards are converted to integers in the range 0-51.
    """
    try:
        if 'raw_obs' not in state_dict:
            logging.warning("No 'raw_obs' in state_dict")
            return (), ()
            
        raw_obs = state_dict['raw_obs']
        if isinstance(raw_obs, dict) and 'raw_obs' in raw_obs:
            raw_obs = raw_obs['raw_obs']
            
        player_cards = raw_obs.get('hand', [])
        community_cards = raw_obs.get('public_cards', [])
        
        if not player_cards:
            logging.warning("No player cards found in state")
            return (), ()
        if not isinstance(community_cards, list):
            logging.warning(f"Community cards is not a list: {community_cards}")
            community_cards = []

        logging.debug(f"Raw player_cards: {player_cards}, raw community_cards: {community_cards}")

        player_cards_converted = []
        for card in player_cards:
            card_num = card_to_number(card)
            if card_num is not None:
                player_cards_converted.append(card_num)
            else:
                logging.error(f"Failed to convert player card: {card}")

        community_cards_converted = []
        for card in community_cards:
            card_num = card_to_number(card)
            if card_num is not None:
                community_cards_converted.append(card_num)
            else:
                logging.error(f"Failed to convert community card: {card}")

        logging.debug(f"Converted player_cards: {player_cards_converted}, converted community_cards: {community_cards_converted}")
        return tuple(player_cards_converted), tuple(community_cards_converted)
    except Exception as e:
        logging.error(f"Error in extract_cards: {str(e)}")
        return (), ()

# ========== EMBEDDING КАРТ ==========
class CardEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank_embed = nn.Embedding(13, 8)
        self.suit_embed = nn.Embedding(4, 4)
        
    def forward(self, cards):
        try:
            if not cards:
                return torch.zeros(12).to(device)
                
            ranks = []
            suits = []
            for card in cards:
                if not (0 <= card <= 51):
                    logging.error(f"Invalid card index for embedding: {card}")
                    return torch.zeros(12).to(device)
                rank = card % 13  # 0 to 12
                suit = card // 13  # 0 to 3
                ranks.append(rank)
                suits.append(suit)
            
            ranks = torch.tensor(ranks, dtype=torch.long).to(device)
            suits = torch.tensor(suits, dtype=torch.long).to(device)
            
            return torch.cat([
                self.rank_embed(ranks).mean(dim=0),
                self.suit_embed(suits).mean(dim=0)
            ])
        except Exception as e:
            logging.error(f"Card embedding failed: {str(e)}")
            return torch.zeros(12).to(device)

# ========== ПОТОКОБЕЗОПАСНАЯ СТАТИСТИКА ОППОНЕНТОВ ==========
class SharedOpponentStats:
    def __init__(self):
        self.stats = [[0] * 16 for _ in range(num_players)]
        self._initialize()

    def _initialize(self):
        for pid in range(num_players):
            for stage in range(4):
                for metric in range(4):
                    self.stats[pid][stage * 4 + metric] = 0

    def update(self, player_id, action, stage, bet_size=0):
        stage_idx = np.argmax(stage)
        idx_base = stage_idx * 4
        self.stats[player_id][idx_base + 3] += 1
        if action == 0:
            self.stats[player_id][idx_base + 1] += 1
        elif action == 1:
            self.stats[player_id][idx_base + 2] += 1
        elif action > 1:
            self.stats[player_id][idx_base + 0] += int(bet_size)

    def get_behavior(self, player_id, stage):
        stage_idx = np.argmax(stage)
        idx_base = stage_idx * 4
        total = max(self.stats[player_id][idx_base + 3], 1)
        return np.array([
            self.stats[player_id][idx_base + 0] / total,
            self.stats[player_id][idx_base + 1] / total,
            self.stats[player_id][idx_base + 2] / total
        ], dtype=np.float16)

    def merge(self, other_stats):
        for pid in range(num_players):
            for i in range(16):
                self.stats[pid][i] += other_stats.stats[pid][i]

# ========== РАСШИРЕННАЯ ОБРАБОТКА СОСТОЯНИЙ ==========
@torch.jit.script
def fast_normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + 1e-8)

class StateProcessor:
    def __init__(self, noise_intensity=0.0):
        self.noise_intensity = noise_intensity
        self.card_embedding = CardEmbedding().to(device)

    def process(self, state, position, active_players, bets, stacks, stage, opponent_behaviors, action=None):
        if isinstance(state, tuple):
            state_dict = state[0]
        else:
            state_dict = state
        if 'obs' not in state_dict:
            raise ValueError("State dict missing 'obs' key")
        stage_array = np.array(stage)
        obs_array = state_dict['obs']
        # Ensure obs_array is a list to avoid slicing issues
        if isinstance(obs_array, np.ndarray):
            obs_array = obs_array.tolist()
        obs = torch.FloatTensor(obs_array[52:] if stage_array[1:].any() else obs_array[52:]).to(device)
        if self.noise_intensity > 0:
            noise = torch.normal(0, self.noise_intensity, obs.shape).to(device)
            logging.debug(f"Added noise with mean: {noise.mean():.4f}, std: {noise.std():.4f}")
            obs += noise
        player_cards, community_cards = extract_cards(state_dict, stage)
        cards_input = player_cards + community_cards
        card_emb = self.card_embedding(cards_input).flatten() if cards_input else torch.zeros(12).to(device)
        pot = sum(bets)
        my_bet = bets[position]
        spr = stacks[position] / pot if pot > 0 else 10.0
        m_factor = stacks[position] / (pot / active_players) if active_players > 0 else 10.0
        pot_odds = my_bet / (pot + my_bet) if pot + my_bet > 0 else 0.0
        fold_equity = np.mean([b[1] for b in opponent_behaviors]) if opponent_behaviors.size else 0.5
        pos_feature = position / (num_players - 1)
        table_agg = np.mean([b[0] for b in opponent_behaviors]) if opponent_behaviors.size else 0.5
        table_fold = np.mean([b[1] for b in opponent_behaviors]) if opponent_behaviors.size else 0.5
        bets_feature = torch.FloatTensor(bets).to(device) / 1000.0
        stacks_feature = torch.FloatTensor(stacks).to(device) / 1000.0
        stage_feature = torch.FloatTensor(stage).to(device)
        opponent_features = torch.FloatTensor(opponent_behaviors).flatten() if opponent_behaviors.size else torch.zeros((num_players - 1) * 3).to(device)
        processed = torch.cat([
            obs,
            card_emb,
            torch.tensor([pos_feature, active_players / num_players, spr, m_factor, pot_odds, fold_equity], device=device),
            bets_feature,
            stacks_feature,
            stage_feature,
            opponent_features,
            torch.tensor([table_agg, table_fold], device=device)
        ])
        return fast_normalize(processed).cpu().numpy()

# ========== HAND HISTORY ==========
class HandHistory:
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.history = np.zeros((max_size, 7), dtype=np.float32)
        self.timestamps = np.zeros(max_size, dtype=np.float64)
        self.index = 0
        self.full = False

    def add(self, table_id, player_id, action, bet_size, hand_strength, stage, pot):
        idx = self.index % self.max_size
        self.history[idx] = [table_id, player_id, action, bet_size, hand_strength, np.argmax(stage), pot]
        self.timestamps[idx] = time.time()
        self.index += 1
        if self.index >= self.max_size:
            self.full = True

    def analyze_errors(self):
        size = self.max_size if self.full else self.index
        if size == 0:
            return 0, 0
        folds_strong = np.sum((self.history[:size, 2] == 0) & (self.history[:size, 4] > 0.7))
        bluffs_weak = np.sum((self.history[:size, 2] > 1) & (self.history[:size, 4] < 0.2))
        total = size
        return folds_strong / total, bluffs_weak / total

# ========== АГЕНТ ==========
class CustomDQNAgent:
    def __init__(self, model, processor, style='default'):
        self.model = model
        self.processor = processor
        self.epsilon = epsilon_start
        self.style = style
        self.total_updates = 0
        self.reward_buffer = collections.deque(maxlen=100)
        self.action_history = collections.deque(maxlen=5)

    def dynamic_bet_size(self, stacks, bets, position, opponent_behaviors, hand_strength, stage):
        try:
            my_stack = stacks[position]
            pot = sum(bets)
            big_blind = 2
            min_raise = max(big_blind, pot * 0.5)
            
            if hand_strength > 0.85:
                bet = min(my_stack, pot * 2.5)
            elif hand_strength < 0.3:
                bet = 0 if random.random() < 0.7 else pot * 0.5
            else:
                base = pot * 0.75
                aggression = np.mean([b[0] for b in opponent_behaviors]) if opponent_behaviors.size else 0.5
                bet = base * (1 + aggression)
                
            return max(min_raise, int(bet // big_blind * big_blind))
        except Exception as e:
            logging.error(f"Bet calc error: {str(e)}")
            return 2

    def adaptive_epsilon_decay(self, reward):
        self.reward_buffer.append(reward)
        self.total_updates += 1
        if self.total_updates % 100 == 0:
            recent_avg = np.mean(self.reward_buffer)
            decay_factor = 0.98 if recent_avg < 0 else 0.95
            self.epsilon = max(epsilon_end, self.epsilon * decay_factor)
            logging.debug(f"Epsilon updated to {self.epsilon:.4f}, recent reward avg: {recent_avg:.4f}")

    def step(self, state, position, active_players, bets, stacks, stage, opponent_behaviors):
        try:
            logging.debug(f"Agent step called with state: {state}")
            
            if not isinstance(state, dict):
                logging.error(f"State is not a dict: {type(state)}, value: {state}")
                state = {'obs': np.zeros(82), 'legal_actions': OrderedDict({0: None, 1: None, 3: None, 4: None}), 'raw_obs': {}}

            # Standardize legal_actions handling
            legal_actions = []
            if 'legal_actions' in state:
                la = state['legal_actions']
                logging.debug(f"Raw legal_actions: {la}, type: {type(la)}")
                if isinstance(la, (dict, OrderedDict)):
                    legal_actions = list(la.keys())
                elif isinstance(la, (list, tuple)):
                    legal_actions = list(la)
                elif isinstance(la, int):
                    legal_actions = [la]
                else:
                    logging.warning(f"Unexpected legal_actions type: {type(la)}, value: {la}")
                    legal_actions = [0, 1, 3, 4]
            else:
                logging.warning(f"No 'legal_actions' in state: {state}")
                legal_actions = [0, 1, 3, 4]

            if not legal_actions or not all(isinstance(a, (int, np.integer)) for a in legal_actions):
                logging.error(f"Invalid legal_actions: {legal_actions}")
                legal_actions = [0, 1, 3, 4]

            logging.debug(f"Processed legal_actions: {legal_actions}")

            # Convert state['obs'] to a list to avoid slicing issues
            if 'obs' in state:
                if isinstance(state['obs'], dict) and 'obs' in state['obs']:
                    if isinstance(state['obs']['obs'], np.ndarray):
                        state['obs']['obs'] = state['obs']['obs'].tolist()
                        logging.debug(f"Converted state['obs']['obs'] to list: {state['obs']['obs'][:10]}")
                elif isinstance(state['obs'], np.ndarray):
                    state['obs'] = state['obs'].tolist()
                    logging.debug(f"Converted state['obs'] to list: {state['obs'][:10]}")

            processed_state = self.processor.process(
                state, position, active_players, 
                bets, stacks, stage, opponent_behaviors
            )
            state_tensor = torch.FloatTensor(processed_state).to(device).unsqueeze(0)

            if np.random.rand() < self.epsilon:
                # Random action: ensure it's legal
                action = random.choice(legal_actions)
                if action > 1:
                    player_cards, community_cards = extract_cards(state, stage)
                    hand_strength = cached_evaluate(player_cards, community_cards)
                    logging.debug(f"Random action, hand strength: {hand_strength:.4f}")
                    bet_size = self.dynamic_bet_size(stacks, bets, position, opponent_behaviors, hand_strength, stage)
                    logging.debug(f"Dynamic bet size calculated: {bet_size}")
                    pot = sum(bets)
                    my_stack = stacks[position]
                    # Map bet size to a legal action
                    if bet_size >= my_stack and 4 in legal_actions:
                        action = 4  # ALL_IN
                    elif bet_size >= pot and 3 in legal_actions:
                        action = 3  # RAISE_POT
                    elif 2 in legal_actions:
                        action = 2  # RAISE_HALF_POT
                    else:
                        # Fallback to a safe action
                        action = 1 if 1 in legal_actions else 0
                    logging.debug(f"Mapped bet size {bet_size} to action: {action}")
            else:
                with torch.no_grad():
                    q_values = self.model(state_tensor)
                    noise = torch.normal(0, noise_scale, q_values.shape).to(device)
                    q_values += noise
                    logging.debug(f"Q-values: {q_values}, legal_actions: {legal_actions}")
                    # Filter Q-values to only consider legal actions
                    legal_q = q_values[0, legal_actions]
                    action_idx = torch.argmax(legal_q).item()
                    action = legal_actions[action_idx]
                    if action > 1:
                        player_cards, community_cards = extract_cards(state, stage)
                        hand_strength = cached_evaluate(player_cards, community_cards)
                        logging.debug(f"Computed hand strength: {hand_strength}")
                        bet_size = self.dynamic_bet_size(stacks, bets, position, opponent_behaviors, hand_strength, stage)
                        logging.debug(f"Dynamic bet size calculated: {bet_size}")
                        pot = sum(bets)
                        my_stack = stacks[position]
                        # Map bet size to a legal action
                        if bet_size >= my_stack and 4 in legal_actions:
                            action = 4  # ALL_IN
                        elif bet_size >= pot and 3 in legal_actions:
                            action = 3  # RAISE_POT
                        elif 2 in legal_actions:
                            action = 2  # RAISE_HALF_POT
                        else:
                            # Fallback to a safe action
                            action = 1 if 1 in legal_actions else 0
                        logging.debug(f"Mapped bet size {bet_size} to action: {action}")

            self.action_history.append(action)
            logging.debug(f"Agent chose action: {action}")
            return action
        except Exception as e:
            logging.error(f"Agent step failed: {str(e)}")
            return 0

    def eval_step(self, state, position, active_players, bets, stacks, stage, opponent_behaviors):
        return self.step(state, position, active_players, bets, stacks, stage, opponent_behaviors), {}

# ========== ПАРАЛЛЕЛЬНЫЙ СБОР ДАННЫХ ==========
def collect_experience(args):
    env, agents, processor, num_steps, device, table_id, hand_history, result_queue = args
    local_opponent_stats = SharedOpponentStats()
    experiences = []
    
    try:
        logging.info(f"Starting collect_experience for table {table_id} with {num_steps} steps")
        start_time = time.time()
        state = env.reset()
        logging.debug(f"Initial state after env.reset(): {state}")
        
        # Process the initial state once and reuse for all players
        if not isinstance(state, list):
            if isinstance(state, tuple):
                obs_part = state[0]
                legal_actions_part = state[1]
                
                if isinstance(legal_actions_part, int):
                    legal_actions_dict = OrderedDict({legal_actions_part: None})
                elif isinstance(legal_actions_part, (list, tuple)):
                    legal_actions_dict = OrderedDict((a, None) for a in legal_actions_part)
                else:
                    legal_actions_dict = OrderedDict({0: None, 1: None, 3: None, 4: None})
                    logging.warning(f"Unexpected legal_actions_part type in initial state: {type(legal_actions_part)}, value: {legal_actions_part}")
                
                state_dict = {
                    'obs': obs_part,
                    'legal_actions': legal_actions_dict,
                    'raw_obs': obs_part
                }
            elif isinstance(state, dict):
                legal_actions_dict = state.get('legal_actions', OrderedDict({0: None, 1: None, 3: None, 4: None}))
                if isinstance(legal_actions_dict, list):
                    legal_actions_dict = OrderedDict((a, None) for a in legal_actions_dict)
                elif not isinstance(legal_actions_dict, (dict, OrderedDict)):
                    legal_actions_dict = OrderedDict({0: None, 1: None, 3: None, 4: None})
                state_dict = {
                    'obs': state.get('obs', np.zeros(82)),
                    'legal_actions': legal_actions_dict,
                    'raw_obs': state.get('raw_obs', {})
                }
            else:
                logging.error(f"Invalid initial state type: {type(state)}, value: {state}")
                result_queue.put(([], local_opponent_stats))
                return
        else:
            state_dict = state[0]

        # Create a list of state dictionaries for all players
        state = [copy.deepcopy(state_dict) for _ in range(env.num_players)]
        logging.debug(f"Processed initial state: {state}")

        for step_idx in range(num_steps):
            logging.debug(f"Step {step_idx + 1}/{num_steps} started for table {table_id}")
            step_start_time = time.time()
            player_id = env.timestep % env.num_players
            position = env.game.get_player_id()
            active_players = len([p for p in env.game.players if p.status == 'alive'])
            
            bets = []
            stacks = []
            for i in range(env.num_players):
                if i < len(env.game.players):
                    bets.append(env.game.players[i].in_chips if env.game.players[i].status == 'alive' else 0)
                    stacks.append(env.game.players[i].remained_chips)
                else:
                    bets.append(0)
                    stacks.append(0)
            
            stage = [int(env.game.round_counter == i) for i in range(4)]
            opponent_behaviors = np.array([
                local_opponent_stats.get_behavior(i, stage) 
                for i in range(env.num_players) 
                if i != player_id
            ])
            
            logging.debug(f"Step {step_idx + 1}/{num_steps}, player {player_id}, state: {state[player_id]}")
            
            action = agents[player_id].step(
                state[player_id], 
                position, 
                active_players,
                bets,
                stacks,
                stage,
                opponent_behaviors
            )
            logging.debug(f"Agent {player_id} chose action: {action}")
            
            try:
                next_state, _ = env.step(action)
            except Exception as e:
                logging.error(f"Experience collection crashed: {str(e)}")
                break
                
            reward = 0
            if 'raw_obs' in state[player_id] and 'my_chips' in state[player_id]['raw_obs']:
                current_chips = state[player_id]['raw_obs']['my_chips']
                if isinstance(next_state, dict) and 'raw_obs' in next_state:
                    next_chips = next_state.get('raw_obs', {}).get('my_chips', current_chips)
                    reward = next_chips - current_chips
                elif isinstance(next_state, list) and len(next_state) > player_id:
                    next_chips = next_state[player_id].get('raw_obs', {}).get('my_chips', current_chips)
                    reward = next_chips - current_chips

            done = False
            if isinstance(next_state, dict) and 'raw_obs' in next_state:
                done = next_state['raw_obs'].get('stage', 0) == 4
            elif isinstance(next_state, list) and len(next_state) > player_id:
                done = next_state[player_id].get('raw_obs', {}).get('stage', 0) == 4

            experience = (state[player_id], action, reward, next_state[player_id] if isinstance(next_state, list) else next_state, done)
            experiences.append(experience)
            logging.debug(f"Collected experience: {experience}")
            
            logging.debug(f"Next state after env.step(action={action}): {next_state}")
            
            # Process the next state once and reuse for all players
            if not isinstance(next_state, list):
                if isinstance(next_state, tuple):
                    next_obs = next_state[0]
                    next_legal = next_state[1]
                    
                    if isinstance(next_legal, int):
                        next_legal_dict = OrderedDict({next_legal: None})
                    elif isinstance(next_legal, (list, tuple)):
                        next_legal_dict = OrderedDict((a, None) for a in next_legal)
                    else:
                        next_legal_dict = OrderedDict({0: None, 1: None, 3: None, 4: None})
                        logging.warning(f"Unexpected next_legal type: {type(next_legal)}, value: {next_legal}")
                        
                    next_state_dict = {
                        'obs': next_obs,
                        'legal_actions': next_legal_dict,
                        'raw_obs': next_obs
                    }
                elif isinstance(next_state, dict):
                    next_legal_dict = next_state.get('legal_actions', OrderedDict({0: None, 1: None, 3: None, 4: None}))
                    if isinstance(next_legal_dict, list):
                        next_legal_dict = OrderedDict((a, None) for a in next_legal_dict)
                    elif not isinstance(next_legal_dict, (dict, OrderedDict)):
                        next_legal_dict = OrderedDict({0: None, 1: None, 3: None, 4: None})
                    next_state_dict = {
                        'obs': next_state.get('obs', np.zeros(82)),
                        'legal_actions': next_legal_dict,
                        'raw_obs': next_state.get('raw_obs', {})
                    }
                else:
                    logging.error(f"Invalid next_state type: {type(next_state)}, value: {next_state}")
                    break
            else:
                next_state_dict = next_state[0]

            state = [copy.deepcopy(next_state_dict) for _ in range(env.num_players)]

            total_reward = reward
            local_opponent_stats.update(player_id, action, stage, action if action > 1 else 0)
            
            logging.debug(f"Calling extract_cards with state: {state[player_id]}, stage: {stage}")
            player_cards, community_cards = extract_cards(state[player_id], stage)
            logging.debug(f"extract_cards returned player_cards: {player_cards}, community_cards: {community_cards}")
            
            hand_strength = cached_evaluate(player_cards, community_cards)
            logging.debug(f"Hand strength: {hand_strength}")
            
            hand_history.add(table_id, player_id, action, action if action > 1 else 0, hand_strength, stage, sum(bets))
            agents[player_id].adaptive_epsilon_decay(total_reward)
            
            if done:
                logging.debug(f"Episode done, resetting env")
                state = env.reset()
                if not isinstance(state, list):
                    if isinstance(state, tuple):
                        obs_part = state[0]
                        legal_actions_part = state[1]
                        
                        if isinstance(legal_actions_part, int):
                            legal_actions_dict = OrderedDict({legal_actions_part: None})
                        elif isinstance(legal_actions_part, (list, tuple)):
                            legal_actions_dict = OrderedDict((a, None) for a in legal_actions_part)
                        else:
                            legal_actions_dict = OrderedDict({0: None, 1: None, 3: None, 4: None})
                            logging.warning(f"Unexpected legal_actions_part type after reset: {type(legal_actions_part)}, value: {legal_actions_part}")
                            
                        state_dict = {
                            'obs': obs_part,
                            'legal_actions': legal_actions_dict,
                            'raw_obs': obs_part
                        }
                    elif isinstance(state, dict):
                        legal_actions_dict = state.get('legal_actions', OrderedDict({0: None, 1: None, 3: None, 4: None}))
                        if isinstance(legal_actions_dict, list):
                            legal_actions_dict = OrderedDict((a, None) for a in legal_actions_dict)
                        elif not isinstance(legal_actions_dict, (dict, OrderedDict)):
                            legal_actions_dict = OrderedDict({0: None, 1: None, 3: None, 4: None})
                        state_dict = {
                            'obs': state.get('obs', np.zeros(82)),
                            'legal_actions': legal_actions_dict,
                            'raw_obs': state.get('raw_obs', {})
                        }
                    else:
                        logging.error(f"Invalid reset state type: {type(state)}, value: {state}")
                        state_dict = {
                            'obs': np.zeros(82),
                            'legal_actions': OrderedDict({0: None, 1: None, 3: None, 4: None}),
                            'raw_obs': {}
                        }
                else:
                    state_dict = state[0]
                state = [copy.deepcopy(state_dict) for _ in range(env.num_players)]

            step_duration = time.time() - step_start_time
            logging.debug(f"Step {step_idx + 1}/{num_steps} completed in {step_duration:.2f} seconds")

        total_duration = time.time() - start_time
        logging.info(f"Finished collect_experience for table {table_id}, collected {len(experiences)} experiences in {total_duration:.2f} seconds")
        result_queue.put((experiences, local_opponent_stats))
    except Exception as e:
        logging.error(f"Experience collection crashed: {str(e)}")
        result_queue.put(([], local_opponent_stats))

# ========== ТЕСТИРОВАНИЕ С МЕТРИКАМИ ==========
def tournament(env, num):
    total_reward = 0
    action_counts = {'fold': 0, 'call': 0, 'raise': 0}
    total_actions = 0
    if env.game.players is None:
        logging.error("env.game.players is None, initializing default players")
        num_players_actual = env.num_players
    else:
        num_players_actual = len(env.game.players)
    
    logging.info(f"Starting tournament with {num} episodes")
    start_time = time.time()

    for episode in range(num):
        logging.debug(f"Tournament episode {episode + 1}/{num} started")
        state = env.reset()
        if not isinstance(state, list):
            if isinstance(state, tuple):
                state_dict = {'obs': state[0], 'legal_actions': state[1], 'raw_obs': state[0]}
            else:
                state_dict = state
            state = [copy.deepcopy(state_dict) for _ in range(env.num_players)]
        done = False
        while not done:
            player_id = env.timestep % env.num_players
            position = env.game.get_player_id()
            active_players = len([p for p in env.game.players if p.status == 'alive'])
            bets = [env.game.players[i].in_chips if i < num_players_actual and env.game.players[i].status == 'alive' else 0 for i in range(num_players)]
            stacks = [env.game.players[i].remained_chips if i < num_players_actual else 0 for i in range(num_players)]
            stage = [1 if env.game.round_counter == i else 0 for i in range(4)]
            opponent_behaviors = np.array([opponent_stats.get_behavior(i, stage) for i in range(num_players) if i != player_id and env.game.players[i].status == 'alive'])
            action, _ = env.agents[player_id].eval_step(state[player_id], position, active_players, bets, stacks, stage, opponent_behaviors)
            state, _ = env.step(action)
            reward = 0
            done = False
            if not isinstance(state, list):
                if isinstance(state, tuple):
                    state_dict = {'obs': state[0], 'legal_actions': state[1], 'raw_obs': state[0]}
                else:
                    state_dict = state
                state = [copy.deepcopy(state_dict) for _ in range(env.num_players)]
            if player_id == 0:
                total_reward += reward
                if action == 0:
                    action_counts['fold'] += 1
                elif action == 1:
                    action_counts['call'] += 1
                elif action > 1:
                    action_counts['raise'] += 1
                total_actions += 1
        logging.debug(f"Tournament episode {episode + 1}/{num} completed")

    winrate = total_reward / num
    action_freq = {k: v / total_actions if total_actions > 0 else 0 for k, v in action_counts.items()}
    total_duration = time.time() - start_time
    logging.info(f"Tournament completed in {total_duration:.2f} seconds, winrate: {winrate:.2f}, action_freq: {action_freq}")
    return winrate, action_freq

# ========== СЕССИЯ ОБУЧЕНИЯ ==========
class TrainingSession:
    def __init__(self):
        self.processor = StateProcessor(noise_intensity=0.0)
        self.opponent_stats = SharedOpponentStats()
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.reward_normalizer = RewardNormalizer()

    def initialize_models(self, input_size, num_actions):
        self.model = DuelingPokerNN(input_size, num_actions).to(device)
        self.target_model = DuelingPokerNN(input_size, num_actions).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=500)
        logging.info(f"Models initialized with input_size={input_size}, num_actions={num_actions}")

    def train_epoch(self, buffer, episode):
        batch, indices, weights = buffer.sample(batch_size)
        states = torch.stack([torch.FloatTensor(self.processor.process(x[0], x[5], x[6], x[7], x[8], x[9], x[10])) for x in batch]).to(device)
        actions = torch.LongTensor([x[1] for x in batch]).to(device)
        rewards = torch.FloatTensor([self.reward_normalizer.normalize(x[2]) for x in batch]).to(device)
        next_states = torch.stack([torch.FloatTensor(self.processor.process(x[3], x[5], x[6], x[7], x[8], x[9], x[10])) for x in batch]).to(device)
        dones = torch.FloatTensor([x[4] for x in batch]).to(device)

        with torch.no_grad():
            next_actions = self.model(next_states).max(1)[1]
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards + (gamma * next_q.squeeze() * (1 - dones))

        self.optimizer.zero_grad()
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        loss = (weights * nn.SmoothL1Loss(reduction='none')(current_q.squeeze(), target_q)).mean()

        loss.backward()
        self._check_gradients()
        self.optimizer.step()
        self.scheduler.step(loss)

        td_errors = (current_q.squeeze() - target_q).abs().cpu().numpy()
        buffer.update_priorities(indices, td_errors)
        logging.debug(f"Episode {episode}, training loss: {loss.item():.4f}")
        return loss.item()

    def _check_gradients(self):
        max_grad = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_max = p.grad.data.abs().max().item()
                if param_max > max_grad:
                    max_grad = param_max
        if max_grad > 1000:
            logging.warning(f"Exploding gradients detected: {max_grad}, clipping to {grad_clip_value}")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_value)

    def save_checkpoint(self, path, is_best=False):
        checkpoint = {
            'model': self.model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'stats': self.opponent_stats.stats,
            'version': code_version,
            'timestamp': time.time()
        }
        torch.save(checkpoint, path)
        if is_best:
            shutil.copyfile(path, best_model_path)
        logging.info(f"Checkpoint saved to {path}, is_best={is_best}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        if checkpoint['version'] != code_version:
            raise ValueError(f"Checkpoint version {checkpoint['version']} does not match current version {code_version}")
        self.model.load_state_dict(checkpoint['model'])
        self.target_model.load_state_dict(checkpoint['target_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.opponent_stats.stats = checkpoint['stats']
        logging.info(f"Checkpoint loaded from {path}")

    def train(self):
        envs = [rlcard.make('no-limit-holdem', config={'num_players': num_players, 'seed': 42 + i}) for i in range(num_tables)]
        for i, env in enumerate(envs):
            logging.info(f"Env {i}: num_players={env.num_players}, state_shape={env.state_shape}")
            if env.num_players != num_players:
                logging.warning(f"Environment {i} initialized with {env.num_players} players instead of {num_players}")
        
        base_state_size = np.prod(envs[0].state_shape[0]) - 52 + 12  # Adjusted for card embedding size
        extra_features = 2 + num_players + num_players + 4 + ((num_players - 1) * 3) + 2
        state_size = base_state_size + extra_features
        num_actions = envs[0].num_actions
        self.initialize_models(state_size, num_actions)

        agent_styles = ['tight', 'loose', 'aggressive', 'bluffer', 'default', 'passive']
        agents_per_table = [[CustomDQNAgent(self.model, self.processor, s) for s in agent_styles[:min(num_players, envs[i].num_players)]] for i in range(num_tables)]
        for env, agents in zip(envs, agents_per_table):
            env.set_agents(agents)

        buffer = PrioritizedExperienceBuffer()
        hand_history = HandHistory()

        test_env = rlcard.make('no-limit-holdem', config={'num_players': num_players, 'seed': 42})
        test_agents = [CustomDQNAgent(self.model, self.processor, s) for s in agent_styles[:min(num_players, test_env.num_players)]]
        test_env.set_agents(test_agents)

        pbar = tqdm(total=num_episodes, desc="Обучение", dynamic_ncols=True, file=sys.stdout)
        total_hands = 0
        losses = collections.deque(maxlen=early_stop_patience)
        winrates = collections.deque(maxlen=early_stop_patience)
        best_loss = float('inf')
        best_winrate = -float('inf')
        last_test_time = time.time()

        for episode in range(num_episodes):
            logging.info(f"Starting episode {episode + 1}/{num_episodes}")
            episode_start_time = time.time()

            result_queues = [Queue() for _ in range(num_workers)]
            processes = []
            
            for i in range(num_workers):
                env = envs[i % num_tables]
                agents = agents_per_table[i % num_tables]
                p = Process(
                    target=collect_experience,
                    args=((env, agents, self.processor, steps_per_worker, device, i % num_tables, hand_history, result_queues[i]),)
                )
                logging.debug(f"Starting process {i} for table {i % num_tables}")
                p.start()
                processes.append(p)
            
            results = []
            for idx, q in enumerate(result_queues):
                try:
                    result = q.get(timeout=60)
                    results.append(result)
                    logging.info(f"Received result from queue {idx}, experiences: {len(result[0])}")
                except Queue.Empty:
                    logging.error(f"Queue.get() timed out after 60 seconds for queue {idx}, process likely hung")
                    for p in processes:
                        p.terminate()
                        logging.debug(f"Terminated process {p.pid}")
                    raise RuntimeError(f"Experience collection process hung for queue {idx}")
            
            for p in processes:
                p.join()
                logging.info(f"Process {p.pid} joined")

            total_experiences = 0
            for experiences, local_stats in results:
                total_experiences += len(experiences)
                for exp in experiences:
                    self.reward_normalizer.update(exp[2])
                    buffer.add(exp, priority=1.0)
                    total_hands += 1
                self.opponent_stats.merge(local_stats)
            logging.info(f"Added {total_experiences} experiences to buffer, total hands: {total_hands}")

            if len(buffer) > batch_size:
                loss = self.train_epoch(buffer, episode)
                losses.append(loss)

            if episode % target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                logging.info(f"Target model updated at episode {episode}")

            if episode % selfplay_update_freq == 0:
                for table_agents in agents_per_table:
                    for agent in table_agents:
                        agent.model.load_state_dict(self.model.state_dict())
                logging.info(f"Agent models updated for self-play at episode {episode}")

            if episode % log_freq == 0 and len(losses) > 0:
                avg_loss = np.mean(losses)
                epsilon_val = agents_per_table[0][0].epsilon
                current_lr = self.optimizer.param_groups[0]['lr']
                folds_strong, bluffs_weak = hand_history.analyze_errors()
                logging.info(f"Episode {episode} | Avg Loss: {avg_loss:.4f} | Epsilon: {epsilon_val:.4f} | "
                             f"LR: {current_lr:.2e} | Total Hands: {total_hands} | Folds Strong: {folds_strong:.2f} | "
                             f"Bluffs Weak: {bluffs_weak:.2f}")
                writer.add_scalar('Loss/Avg', avg_loss, episode)
                writer.add_scalar('Epsilon', epsilon_val, episode)
                writer.add_scalar('Hands/Total', total_hands, episode)
                writer.add_scalar('Errors/Folds_Strong', folds_strong, episode)
                writer.add_scalar('Errors/Bluffs_Weak', bluffs_weak, episode)

            if time.time() - last_test_time >= test_interval:
                test_reward, action_freq = tournament(test_env, 20)
                winrates.append(test_reward)
                logging.info(f"Test at Episode {episode} | Winrate: {test_reward:.2f} | "
                             f"Fold: {action_freq['fold']:.2f} | Call: {action_freq['call']:.2f} | Raise: {action_freq['raise']:.2f}")
                writer.add_scalar('Test/Winrate', test_reward, episode)
                writer.add_scalars('Test/Action_Freq', action_freq, episode)
                self.save_checkpoint(model_path)
                if test_reward > best_winrate:
                    best_winrate = test_reward
                    self.save_checkpoint(best_model_path, is_best=True)
                    logging.info(f"New best model saved to {best_model_path}")
                last_test_time = time.time()

            if len(losses) == early_stop_patience and len(winrates) > 0:
                avg_loss = np.mean(losses)
                avg_winrate = np.mean(winrates)
                if (avg_loss + early_stop_threshold >= best_loss and avg_winrate <= best_winrate + early_stop_threshold) and total_hands >= 10_000:
                    logging.info(f"Early stopping at Episode {episode} | Avg Loss: {avg_loss:.4f} | Avg Winrate: {avg_winrate:.2f}")
                    break
                best_loss = min(best_loss, avg_loss)
                best_winrate = max(best_winrate, avg_winrate)

            if total_hands >= 10_000:
                logging.info(f"Достигнуто 10,000 раздач на эпизоде {episode}")
                break

            episode_duration = time.time() - episode_start_time
            logging.info(f"Episode {episode + 1}/{num_episodes} completed in {episode_duration:.2f} seconds")

            pbar.update(1)

        pbar.close()
        test_reward, action_freq = tournament(test_env, 20)
        logging.info(f"Final Winrate: {test_reward:.2f} | "
                     f"Fold: {action_freq['fold']:.2f} | Call: {action_freq['call']:.2f} | Raise: {action_freq['raise']:.2f}")
        writer.add_scalar('Final/Winrate', test_reward, num_episodes)
        writer.add_scalars('Final/Action_Freq', action_freq, num_episodes)
        self.save_checkpoint(model_path)
        writer.close()

# ========== ЗАПУСК ==========
if __name__ == "__main__":
    mp.set_start_method('spawn')
    opponent_stats = SharedOpponentStats()
    session = TrainingSession()
    try:
        session.train()
    except KeyboardInterrupt:
        session.save_checkpoint('interrupted.pt')
        logging.info("Training interrupted, checkpoint saved")
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        session.save_checkpoint('error_state.pt')
