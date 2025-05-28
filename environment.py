import gym
import numpy as np
import opendssdirect as dss

class PowerDistributionEnv(gym.Env):
    """
    Gym environment for a power distribution network using OpenDSS.
    Each agent controls configurable elements (e.g., batteries, generators, switches).
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, dss_master_file, agent_configs, step_horizon=24):
        """
        dss_master_file: path to the OpenDSS master file (.dss)
        agent_configs: list of dicts, each with:
            - 'elements': list of dicts {'type': 'Battery'|'Generator'|'Switch', 'name': str, 'param': str}
            - 'buses': list of bus names to observe
        step_horizon: number of timesteps per episode (e.g., 24 for 24h)
        """
        super().__init__()
        self.dss_master_file = dss_master_file
        self.agent_configs = agent_configs
        self.n_agents = len(agent_configs)
        self.step_horizon = step_horizon
        self.current_step = 0

        # Initialize OpenDSS circuit
        dss.Basic.ClearAll()
        dss.Text.Command(f"Compile [{self.dss_master_file}]")

        # Define obs/action spaces
        # Obs: for each agent, voltages at its buses + local line losses + global battery SOCs
        obs_dim = max(len(cfg['buses']) for cfg in agent_configs) + 2 + len(agent_configs)
        act_dim = max(len(cfg['elements']) for cfg in agent_configs)
        self.observation_space = [gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
                                  for _ in range(self.n_agents)]
        self.action_space = [gym.spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)
                             for _ in range(self.n_agents)]

    def reset(self):
        """Reset the simulation and return initial observations."""
        self.current_step = 0
        dss.Text.Command("Reset")
        dss.Text.Command("Solve mode=daily number=1 stepsize=1h")
        obs_n = [self._get_obs(i) for i in range(self.n_agents)]
        return obs_n

    def step(self, actions):
        """
        actions: list of action arrays for each agent
        Returns: obs_n, rewards, dones, infos
        """
        # Apply each agent's actions
        for agent_id, action in enumerate(actions):
            self._apply_actions(agent_id, action)

        # Advance simulation by one hour
        dss.Text.Command("Solve mode=daily number=1 stepsize=1h")

        self.current_step += 1
        obs_n = [self._get_obs(i) for i in range(self.n_agents)]
        reward = self._compute_reward()
        rewards = [reward] * self.n_agents
        done = (self.current_step >= self.step_horizon)
        dones = [done] * self.n_agents
        info = {'step': self.current_step}
        return obs_n, rewards, dones, info

    def _get_obs(self, agent_id):
        """Construct observation vector for a given agent."""
        cfg = self.agent_configs[agent_id]
        # Voltages at agent's buses
        voltages = []
        all_volts = np.array(dss.Circuit.AllBusVolts())  # flat list [V1a,V1b,V1c, V2a...]
        bus_names = dss.Circuit.AllBusNames()
        for bus in cfg['buses']:
            if bus in bus_names:
                idx = bus_names.index(bus) * dss.Circuit.NumNodes()
                phases = dss.Circuit.NumNodes()
                ph_volts = all_volts[idx:idx+phases]
                voltages.append(np.mean(np.abs(ph_volts)))
            else:
                voltages.append(0.0)
        voltages = np.array(voltages, dtype=np.float32)

        # Local losses: sum of losses on lines connected to agent's buses
        local_loss = 0.0
        for line in dss.Circuit.LinesI():
            # simplistic: add small placeholder
            local_loss += 0.0

        # Global battery states (for all batteries)
        socs = []
        for cfg2 in self.agent_configs:
            for el in cfg2['elements']:
                if el['type'] == 'Battery':
                    dss.Circuit.SetActiveElement(f"EnergyStorage.{el['name']}")
                    socs.append(dss.Properties.Value('pctCapacity'))

        obs = np.concatenate([
            voltages,
            np.array([local_loss], dtype=np.float32),
            np.array(socs, dtype=np.float32),
        ])
        # pad to fixed size
        pad = self.observation_space[agent_id].shape[0] - obs.size
        if pad > 0:
            obs = np.concatenate([obs, np.zeros(pad, dtype=np.float32)])
        return obs

    def _apply_actions(self, agent_id, action):
        """Map normalized action vector [-1,1] to control element parameters."""
        cfg = self.agent_configs[agent_id]
        for idx, el in enumerate(cfg['elements']):
            if idx >= len(action):
                break
            val = float(action[idx])
            # map -1..1 to 0..1 range
            scale = (val + 1) / 2
            if el['type'] == 'Battery':
                # set dispatch kW
                max_kw = float(el.get('max_kw', 100.0))
                dispatch = max_kw * (2*scale - 1)  # allow charge/discharge
                dss.Circuit.SetActiveElement(f"EnergyStorage.{el['name']}")
                dss.Properties.Name('kW')
                dss.Properties.Value(str(dispatch))
            elif el['type'] == 'Generator':
                max_kw = float(el.get('max_kw', 500.0))
                gen_kw = max_kw * scale
                dss.Circuit.SetActiveElement(f"Generator.{el['name']}")
                dss.Properties.Name('kW')
                dss.Properties.Value(str(gen_kw))
            elif el['type'] == 'Switch':
                # binary switch: open if scale<0.5 else close
                state = 'Open' if scale < 0.5 else 'Close'
                dss.Text.Command(f"Switch.{el['name']}.{state}")
            # extend for other types as needed

    def _compute_reward(self):
        """Compute a global reward: negative total losses minus unserved energy."""
        losses = np.array(dss.Circuit.Losses())[0]  # total losses [kW, kVAr]
        # unserved energy (load mismatch): sum of |P_gen - P_load|
        gen = dss.Circuit.TotalPower()[0]
        load = -dss.Circuit.TotalPower()[1]
        unserved = abs(gen - load)
        # reward: penalize losses and unserved
        return - (losses + unserved)

    def render(self, mode='human'):
        pass
