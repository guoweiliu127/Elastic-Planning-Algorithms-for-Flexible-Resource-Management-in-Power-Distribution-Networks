# Elastic Planning Algorithm (EPA) with Multi-Agent Deep Reinforcement Learning for Power Distribution Networks

This repository contains an open-source implementation of the Elastic Planning Algorithm (EPA) leveraging Multi-Agent Deep Reinforcement Learning (MADRL) for flexible resource management in power distribution networks.

## Structure

- `environment.py` — Gym environment wrapping OpenDSS for simulating a distribution network.
- `agent.py` — MADDPG agent implementation (actor-critic, replay buffer, training utilities).
- `train.py` — Training script to train multiple agents in the environment.
- `requirements.txt` — Python dependencies.

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/epa-madr.git
   cd epa-madr
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your OpenDSS master file (`circuit_master.dss`) in the root directory.

## Usage

Train the agents:
```bash
python train.py
```

## License

This project is licensed under the MIT License.
