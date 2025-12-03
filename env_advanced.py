import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

class GeothermalEnvAdvanced(gym.Env):
    """
    Geothermal-aware compute scheduling:
    - Reinforcement Learning agent decides energy source per job
    - Geothermal well has finite extraction rate per timestep (max_rate)
    - If extraction exceeds reservoir_heat, thermal collapse penalty applies
    - Battery stores excess or fallback energy
    - Grid exists as fallback with penalty (non-renewable)
    """

    metadata = {"render_modes": []}

    def __init__(self, episode_length=60, max_rate=50.0, regen_rate=8.0,
                 battery_capacity_kwh=200.0, seed=0):
        super().__init__()
        np.random.seed(seed)
        random.seed(seed)

        self.episode_length = episode_length

        # geothermal well
        self.max_rate = max_rate          # max extractable power per job
        self.regen_rate = regen_rate      # constant heat regeneration
        self.reservoir_heat = max_rate    # starts full

        # battery
        self.battery_capacity = battery_capacity_kwh
        self.battery_kwh = battery_capacity_kwh * 0.3

        # observation: geothermal %, battery %, job %
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3,), dtype=np.float32
        )

        # actions: 0=geothermal 1=battery 2=grid
        self.action_space = spaces.Discrete(3)

        self.job = None
        self.steps = 0

    def _sample_job(self):
        return float(
            np.random.choice([10, 20, 30, 40, 60], p=[0.40, 0.25, 0.20, 0.10, 0.05])
        )

    def _obs(self):
        return np.array([
            min(1.0, self.reservoir_heat / self.max_rate),     # normalized well heat
            self.battery_kwh / self.battery_capacity,          # battery %
            min(1.0, self.job / 60.0)                          # job %
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.reservoir_heat = self.max_rate
        self.battery_kwh = self.battery_capacity * 0.3
        self.steps = 0
        self.job = self._sample_job()
        return self._obs(), {}

    def step(self, action):
        job_kwh = self.job
        reward = 0
        executed = False

        # Action 0: Use Geothermal
        if action == 0:
            if job_kwh <= self.reservoir_heat:
                self.reservoir_heat -= job_kwh
                reward += 4
                executed = True
            else:
                reward -= 15     # overdraw / thermal collapse

        # Action 1: Use Battery
        elif action == 1:
            if job_kwh <= self.battery_kwh:
                self.battery_kwh -= job_kwh
                reward += 2
                executed = True
            else:
                reward -= 8

        # Action 2: Use Grid (penalty)
        elif action == 2:
            reward -= 5
            executed = True

        # geothermal regeneration
        self.reservoir_heat = min(self.max_rate, self.reservoir_heat + self.regen_rate)

        # recharge battery slightly if geothermal well > 80%
        leftover = max(0.0, self.reservoir_heat - self.max_rate * 0.8)
        self.battery_kwh = min(self.battery_capacity, self.battery_kwh + leftover * 0.4)

        self.steps += 1
        done = self.steps >= self.episode_length

        self.job = self._sample_job()

        return self._obs(), reward, done, False, {"executed": executed}
