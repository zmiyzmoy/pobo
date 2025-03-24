import os
import time
import logging
import sys
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
        self.KMEANS_PATH = os.path.join(self.BASE_DIR, 'models', 'kmeans.joblib')
        self.LOG_DIR = os.path.join(self.BASE_DIR, 'logs')
        self.NUM_EPISODES = 10  # Уменьшено для теста
        self.BATCH_SIZE = 32    # Уменьшено для стабильности
        self.BUFFER_CAPACITY = 1000
        self.NUM_WORKERS = 1    # Один воркер для простоты
        self.STEPS_PER_WORKER = 100  # Уменьшено
        self.LEARNING_RATE = 1e-4
        self.NUM_PLAYERS = 6
        self.NUM_BUCKETS = 50
        self.GAME_NAME = "universal_poker(betting=nolimit,numPlayers=6,numRounds=4,blind=1 2 3 4 5 6,raiseSize=0.10 0.20 0.40 0.80,stack=100 100 100 100 100 100,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1)"

config = Config()
os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
os.makedirs(config.LOG_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
game = pyspiel.load_game(config.GAME_NAME)
logging.info(f"Конфигурация инициализирована, устройство: {device}, CUDA: {torch.cuda.is_available()}")

# Модель нейронной сети
class SimpleNet(nn.Module):
    def __init__(self, input_size: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        logging.info("SimpleNet инициализирована")

    def forward(self, x):
        return self.net(x)

# Простая обработка состояний
class StateProcessor:
    def __init__(self):
        self.card_embedding = nn.Embedding(52, 8).to(device)  # Упрощенное вложение карт
        self.buckets = self._load_or_precompute_buckets()
        self.state_size = config.NUM_BUCKETS + config.NUM_PLAYERS

    def _load_or_precompute_buckets(self):
        if os.path.exists(config.KMEANS_PATH):
            buckets = joblib.load(config.KMEANS_PATH)
            logging.info(f"Загружены бакеты из {config.KMEANS_PATH}")
            return buckets
        
        # Упрощенная генерация данных для KMeans
        all_hands = np.random.rand(169, 8).astype(np.float32)  # 169 уникальных комбинаций рук
        kmeans = KMeans(n_clusters=config.NUM_BUCKETS, random_state=42)
        kmeans.fit(all_hands)
        joblib.dump(kmeans, config.KMEANS_PATH)
        logging.info(f"Бакеты вычислены и сохранены в {config.KMEANS_PATH}")
        return kmeans

    def process(self, states, player_ids, bets, stacks):
        batch_size = len(states)
        cards_batch = []
        for state, pid in zip(states, player_ids):
            info = state.information_state_tensor(pid)
            private_cards = [int(i) for i, c in enumerate(info[:52]) if c > 0][:2] or [0, 1]
            cards_batch.append(private_cards)
        
        # Явное приведение к float32
        card_embs = torch.stack([self.card_embedding(torch.tensor(cards, device=device)) for cards in cards_batch])
        card_embs = card_embs.mean(dim=1).to(dtype=torch.float32).cpu().detach().numpy().astype(np.float32)
        
        bucket_idxs = self.buckets.predict(card_embs)
        bucket_one_hot = np.zeros((batch_size, config.NUM_BUCKETS), dtype=np.float32)
        bucket_one_hot[np.arange(batch_size), bucket_idxs] = 1.0
        
        bets_norm = np.array(bets, dtype=np.float32) / 100.0  # Упрощенная нормализация
        return np.concatenate([bucket_one_hot, bets_norm], axis=1)

# Упрощенный агент
class PokerAgent:
    def __init__(self, game, processor):
        self.game = game
        self.processor = processor
        self.num_actions = game.num_distinct_actions()
        self.net = SimpleNet(processor.state_size, self.num_actions).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.LEARNING_RATE)
        self.global_step = 0
        logging.info(f"PokerAgent инициализирован")

    def step(self, state, player_id, bets, stacks):
        state_tensor = torch.tensor(self.processor.process([state], [player_id], [bets], [stacks]), 
                                  dtype=torch.float32, device=device)
        with torch.no_grad():
            q_values = self.net(state_tensor)[0]
        legal_actions = state.legal_actions(player_id)
        q_values = q_values[legal_actions]
        return legal_actions[torch.argmax(q_values).item()]

# Сбор опыта
@ray.remote(num_cpus=1, num_gpus=0.5)
def collect_experience(game, agent, processor, steps, worker_id):
    try:
        env = game.new_initial_state()
        experiences = []
        for _ in range(steps):
            if env.is_terminal():
                env = game.new_initial_state()
                continue
            player_id = env.current_player()
            if player_id < 0:  # Chance node
                action = random.choice(env.legal_actions())
            else:
                bets = env.bets() if hasattr(env, 'bets') else [0] * config.NUM_PLAYERS
                stacks = env.stacks() if hasattr(env, 'stacks') else [100] * config.NUM_PLAYERS
                action = agent.step(env, player_id, bets, stacks)
            next_state = env.clone()
            next_state.apply_action(action)
            reward = next_state.returns()[player_id] if next_state.is_terminal() else 0
            experiences.append((env.clone(), action, reward, next_state.clone(), next_state.is_terminal()))
            env = next_state
        logging.info(f"Worker {worker_id} collected {len(experiences)} experiences")
        return experiences
    except Exception as e:
        logging.error(f"Worker {worker_id} failed: {str(e)}")
        return []

# Обучение
class Trainer:
    def __init__(self, game, agent, processor):
        self.game = game
        self.agent = agent
        self.processor = processor
        self.buffer = []
        self.global_step = 0

    def train(self):
        pbar = tqdm(total=config.NUM_EPISODES, desc="Training")
        for episode in range(config.NUM_EPISODES):
            logging.info(f"Starting episode {episode}")
            futures = [collect_experience.remote(self.game, self.agent, self.processor, config.STEPS_PER_WORKER, 0)]
            experiences = ray.get(futures)[0]
            
            if experiences:
                self.buffer.extend(experiences)
                self.buffer = self.buffer[-config.BUFFER_CAPACITY:]  # Ограничение буфера
                
                if len(self.buffer) >= config.BATCH_SIZE:
                    batch = random.sample(self.buffer, config.BATCH_SIZE)
                    states, actions, rewards, next_states, dones = zip(*batch)
                    
                    states = torch.tensor(self.processor.process(states, list(range(len(states))), 
                                                               [e[0].bets() for e in batch], 
                                                               [e[0].stacks() for e in batch]), 
                                        dtype=torch.float32, device=device)
                    actions = torch.tensor(actions, dtype=torch.long, device=device)
                    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                    next_states = torch.tensor(self.processor.process(next_states, list(range(len(next_states))), 
                                                                   [e[3].bets() for e in batch], 
                                                                   [e[3].stacks() for e in batch]), 
                                             dtype=torch.float32, device=device)
                    
                    self.agent.optimizer.zero_grad()
                    q_values = self.agent.net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    next_q_values = self.agent.net(next_states).max(1)[0].detach()
                    targets = rewards + 0.95 * next_q_values * (1 - torch.tensor(dones, dtype=torch.float32, device=device))
                    loss = nn.MSELoss()(q_values, targets)
                    loss.backward()
                    self.agent.optimizer.step()
                    
                    self.global_step += 1
                    logging.info(f"Step {self.global_step} | Loss: {loss.item():.4f}")
            
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
