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
from collections import deque
from typing import List, Tuple, Dict, Optional
from open_spiel.python import policy

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
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
        logging.info("CardEmbedding инициализирован")

    def forward(self, cards: List[int]) -> torch.Tensor:
        ranks = torch.tensor([c % 13 for c in cards], dtype=torch.long, device=device)
        suits = torch.tensor([c // 13 for c in cards], dtype=torch.long, device=device)
        return torch.cat([self.rank_embed(ranks).mean(dim=0), self.suit_embed(suits).mean(dim=0)])

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
                    embedding = self.card_embedding(hand).cpu().detach().numpy()
                    all_hands.append(embedding)
        kmeans = KMeans(n_clusters=config.NUM_BUCKETS, random_state=42).fit(all_hands)
        joblib.dump(kmeans, config.KMEANS_PATH)
        logging.info(f"Бакеты вычислены и сохранены в {config.KMEANS_PATH}")
        return kmeans

    def process(self, states: List, player_ids: List[int], bets: List[List[float]] = None, 
                stacks: List[List[float]] = None, stages: List[List[int]] = None) -> np.ndarray:
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
        
        card_embs = torch.stack([self.card_embedding(cards).float() for cards in cards_batch]).cpu().detach().numpy()
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
        logging.info(f"PokerAgent инициализирован, state_size={processor.state_size}, num_actions={self.num_actions}")

    def action_probabilities(self, state, player_id: Optional[int] = None) -> Dict[int, float]:
        legal_actions = state.legal_actions(player_id)
        if not legal_actions:
            return {0: 1.0}
        state_tensor = torch.tensor(self.processor.process([state], [player_id]), 
                                  dtype=torch.float32, device=device)
        with torch.no_grad():
            self.strategy_net.eval()
            strategy_logits = self.strategy_net(state_tensor)[0]
            self.strategy_net.train()
            legal_mask = torch.zeros(self.num_actions, device=device)
            legal_mask[legal_actions] = 1
            strategy_logits = strategy_logits.masked_fill(legal_mask == 0, -1e9)
            strategy = torch.softmax(strategy_logits, dim=0).cpu().numpy()
        probs_dict = {a: float(strategy[a]) for a in legal_actions}
        return probs_dict

    def step(self, state, player_id):
        probs = self.action_probabilities(state, player_id)
        action = random.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]
        return action

# Сбор опыта
@ray.remote(num_cpus=1, num_gpus=0.5)
def collect_experience(game, agent, processor, steps, worker_id):
    try:
        env = game.new_initial_state()
        experiences = []
        step_count = 0
        while step_count < steps:
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
            action = agent.step(env, player_id)
            next_state = env.clone()
            next_state.apply_action(action)
            reward = next_state.returns()[player_id] if next_state.is_terminal() else 0
            experiences.append((env.clone(), player_id, action, reward, next_state.clone(), next_state.is_terminal(), bets, stacks, stage))
            env = next_state
            step_count += 1
        logging.info(f"Worker {worker_id} collected {len(experiences)} experiences")
        return experiences
    except Exception as e:
        logging.error(f"Worker {worker_id} failed: {str(e)}")
        raise

# Обучение
class Trainer:
    def __init__(self, game, agent, processor):
        self.game = game
        self.agent = agent
        self.processor = processor
        self.buffer = PrioritizedReplayBuffer(config.BUFFER_CAPACITY)
        self.global_step = 0
        self.beta = 0.4

    def train(self):
        pbar = tqdm(total=config.NUM_EPISODES, desc="Training")
        for episode in range(config.NUM_EPISODES):
            logging.info(f"Starting episode {episode}")
            futures = [collect_experience.remote(self.game, self.agent, self.processor, config.STEPS_PER_WORKER, 0)]
            experiences = ray.get(futures)[0]
            
            if experiences:
                self.buffer.add_batch(experiences)
                
                if len(self.buffer) >= config.BATCH_SIZE:
                    batch, indices, weights = self.buffer.sample(config.BATCH_SIZE, self.beta)
                    states, player_ids, actions, rewards, next_states, dones, bets, stacks, stages = zip(*batch)
                    
                    states = torch.tensor(self.processor.process(states, player_ids, bets, stacks, stages), 
                                        dtype=torch.float32, device=device)
                    actions = torch.tensor(actions, dtype=torch.long, device=device)
                    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                    next_player_ids = [s.current_player() if s.current_player() >= 0 else 0 for s in next_states]
                    next_states = torch.tensor(self.processor.process(next_states, next_player_ids, stacks, stacks, stages), 
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
            
            pbar.update(1)
        pbar.close()

# Запуск
if __name__ == "__main__":
    mp.set_start_method('spawn')
    ray.init(num_gpus=1, num_cpus=4, ignore_reinit_error=True)
    game = pyspiel.load_game(config.GAME_NAME)
    processor = StateProcessor()
    agent = PokerAgent(game, processor)
    trainer = Trainer(game, agent, processor)
    trainer.train()
    ray.shutdown()
