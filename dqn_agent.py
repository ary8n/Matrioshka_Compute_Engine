import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np

class DQNAgent:
    def __init__(self, state_dim, action_dim, device="cpu"):
        self.device = device

        self.q = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        ).to(device)

        self.q_target = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        ).to(device)

        self.q_target.load_state_dict(self.q.state_dict())
        self.action_dim = action_dim
        self.optimizer = optim.Adam(self.q.parameters(), lr=1e-4)
        self.memory = deque(maxlen=20000)

        self.batch_size = 64
        self.gamma = 0.95

        self.eps = 1.0
        self.eps_min = 0.08
        self.eps_decay = 0.995

    def select_action(self, state, valid_actions):
        if random.random() < self.eps:
            return random.choice(valid_actions if valid_actions else list(range(self.action_dim)))

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        q_vals = self.q(state)

        mask = torch.full_like(q_vals, -1e9)
        mask[valid_actions] = 0
        return int((q_vals + mask).argmax().item())

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        s, a, r, ns, d = zip(*batch)

        s = torch.tensor(np.array(s), dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.int64).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        ns = torch.tensor(np.array(ns), dtype=torch.float32).to(self.device)
        d = torch.tensor(d, dtype=torch.float32).to(self.device)

        q_pred = self.q(s)[range(self.batch_size), a]

        with torch.no_grad():
            q_next = self.q_target(ns).max(1)[0]
            q_target = r + self.gamma * q_next * (1 - d)

        loss = nn.MSELoss()(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.eps = max(self.eps_min, self.eps * self.eps_decay)

        if random.random() < 0.005:
            self.q_target.load_state_dict(self.q.state_dict())
