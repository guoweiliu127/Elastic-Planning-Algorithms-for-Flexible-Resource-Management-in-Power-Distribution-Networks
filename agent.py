import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, total_obs_dim, total_act_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, acts):
        x = torch.cat([obs, acts], dim=-1)
        return self.net(x)

class ReplayBuffer:
    def __init__(self, max_size=int(1e6)):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in idx))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class MADDPGAgent:
    def __init__(self, obs_dim, act_dim, agent_id, n_agents, lr=1e-3, gamma=0.99, tau=0.01):
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau

        # Actor networks
        self.actor = Actor(obs_dim, act_dim)
        self.target_actor = Actor(obs_dim, act_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic networks
        tot_obs = obs_dim * n_agents
        tot_act = act_dim * n_agents
        self.critic = Critic(tot_obs, tot_act)
        self.target_critic = Critic(tot_obs, tot_act)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self._hard_update(self.target_actor, self.actor)
        self._hard_update(self.target_critic, self.critic)

    def select_action(self, obs, noise=0.1):
        obs_t = torch.FloatTensor(obs)
        action = self.actor(obs_t).detach().numpy()
        action += noise * np.random.randn(self.act_dim)
        return np.clip(action, -1, 1)

    def update(self, agents, buffer, batch_size):
        if len(buffer) < batch_size:
            return

        # Sample
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        # Convert to tensors
        # Flatten states/actions per agent
        # ... implement batching
        # Compute targets and losses, update critics and actors
        # Soft update targets
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)

    def _hard_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(s.data)

    def _soft_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(t.data * (1.0 - self.tau) + s.data * self.tau)
