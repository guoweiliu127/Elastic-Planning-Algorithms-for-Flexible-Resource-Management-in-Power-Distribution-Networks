import os
import torch
import numpy as np
from environment import PowerDistributionEnv
from agent import MADDPGAgent, ReplayBuffer

def train():
    # Paths & configs
    dss_file = os.path.abspath("circuit_master.dss")
    agent_configs = [
        {
            'elements': [
                {'type': 'Battery', 'name': 'BESS1', 'param': 'kW', 'max_kw': 100},
                {'type': 'Generator', 'name': 'Gen1', 'param': 'kW', 'max_kw': 500},
            ],
            'buses': ['bus1', 'bus2']
        },
        {
            'elements': [
                {'type': 'Battery', 'name': 'BESS2', 'param': 'kW', 'max_kw': 100},
                {'type': 'Generator', 'name': 'Gen2', 'param': 'kW', 'max_kw': 500},
            ],
            'buses': ['bus3', 'bus4']
        },
    ]
    env = PowerDistributionEnv(dss_file, agent_configs, step_horizon=24)
    n_agents = env.n_agents
    obs_dims = [env.observation_space[i].shape[0] for i in range(n_agents)]
    act_dims = [env.action_space[i].shape[0] for i in range(n_agents)]

    # Initialize agents and buffer
    agents = [MADDPGAgent(obs_dims[i], act_dims[i], i, n_agents) for i in range(n_agents)]
    buffer = ReplayBuffer()

    n_episodes = 1000
    for ep in range(n_episodes):
        obs_n = env.reset()
        total_reward = np.zeros(n_agents)
        done = False
        while not done:
            # Select actions
            actions = [agents[i].select_action(obs_n[i]) for i in range(n_agents)]
            # Step
            next_obs_n, rewards, dones, info = env.step(actions)
            done = all(dones)
            # Store transitions
            buffer.push(obs_n, actions, rewards, next_obs_n, done)
            obs_n = next_obs_n
            total_reward += rewards
            # Update
            for agent in agents:
                agent.update(agents, buffer, batch_size=64)
        print(f"Episode {ep}: Rewards {total_reward}")

if __name__ == "__main__":
    train()
