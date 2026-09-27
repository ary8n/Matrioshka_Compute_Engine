import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CarbonAwareComputeEnv(gym.Env):
    """Simulate carbon-aware scheduling of flexible compute workloads.

    The agent balances workload completion against server capacity, battery
    storage, renewable availability, electricity price, and grid emissions.
    """

    metadata = {"render_modes": []}
    ACTIONS = {
        0: "Run now",
        1: "Defer",
        2: "Run reduced",
        3: "Battery-backed run",
    }

    def __init__(self, episode_length=60, server_capacity=100.0,
                 battery_capacity_kwh=200.0, seed=0):
        super().__init__()
        self.episode_length = episode_length
        self.server_capacity = server_capacity
        self.battery_capacity = battery_capacity_kwh
        self.rng = np.random.default_rng(seed)

        # Renewable %, battery %, free server %, carbon intensity, price,
        # workload demand, priority, and deadline are all normalized to [0, 1].
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.steps = 0
        self.battery_kwh = 0.0
        self.server_load = 0.0
        self.renewable_pct = 0.0
        self.carbon_intensity = 0.0
        self.price = 0.0
        self.job = {}
        self.metrics = {}

    def _sample_job(self):
        return {
            "demand": float(self.rng.choice([10, 20, 30, 40, 60], p=[.4, .25, .2, .1, .05])),
            "priority": int(self.rng.integers(1, 11)),
            "deadline": int(self.rng.integers(2, 8)),
        }

    def _update_grid_conditions(self):
        daylight = np.sin((self.steps % 24) / 24 * np.pi)
        self.renewable_pct = float(np.clip(.25 + .55 * daylight + self.rng.normal(0, .06), 0, 1))
        self.carbon_intensity = float(np.clip(.85 - .65 * self.renewable_pct + self.rng.normal(0, .04), .1, 1))
        self.price = float(np.clip(.35 + .45 * self.carbon_intensity + self.rng.normal(0, .04), .05, 1))

    def _obs(self):
        return np.array([
            self.renewable_pct,
            self.battery_kwh / self.battery_capacity,
            max(0.0, 1.0 - self.server_load / self.server_capacity),
            self.carbon_intensity,
            self.price,
            min(1.0, self.job["demand"] / 60.0),
            self.job["priority"] / 10.0,
            min(1.0, self.job["deadline"] / 8.0),
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.battery_kwh = self.battery_capacity * .5
        self.server_load = 0.0
        self._update_grid_conditions()
        self.job = self._sample_job()
        self.metrics = {"completed": 0, "deferred": 0, "energy_kwh": 0.0,
                        "grid_kwh": 0.0, "carbon": 0.0, "cost": 0.0}
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        demand = self.job["demand"]
        priority = self.job["priority"]
        deadline = self.job["deadline"]
        execution_demand = demand * .6 if action == 2 else demand
        executed = False
        source = "deferred"
        grid_energy = 0.0
        reward = 0.0

        if action in (0, 2) and execution_demand <= self.server_capacity - self.server_load:
            executed = True
            source = "renewable/grid mix"
            self.server_load += execution_demand
            grid_energy = execution_demand * (1 - self.renewable_pct)
            reward += 8.0 if action == 0 else 5.0
        elif action == 3 and demand <= self.battery_kwh and demand <= self.server_capacity - self.server_load:
            executed = True
            source = "battery"
            self.battery_kwh -= demand
            self.server_load += demand
            reward += 6.0
        elif action in (0, 2, 3):
            reward -= 6.0

        if not executed:
            self.metrics["deferred"] += 1
            reward -= priority * .4 if deadline <= 1 else .5
        else:
            self.metrics["completed"] += 1
            self.metrics["energy_kwh"] += execution_demand
            self.metrics["grid_kwh"] += grid_energy
            self.metrics["carbon"] += grid_energy * self.carbon_intensity
            self.metrics["cost"] += grid_energy * self.price
            reward -= grid_energy * self.carbon_intensity * .8
            reward -= grid_energy * self.price * .3

        if action == 1:
            reward += 1.0 if deadline > 2 else -priority * .3

        self.server_load *= .55
        self.battery_kwh = min(self.battery_capacity,
                               self.battery_kwh + self.renewable_pct * 10)
        self.steps += 1
        terminated = self.steps >= self.episode_length
        self._update_grid_conditions()
        self.job = self._sample_job()

        info = {
            "action": self.ACTIONS[action], "source": source, "executed": executed,
            "job_demand": demand, "priority": priority, "deadline": deadline,
            "renewable_pct": self.renewable_pct,
            "carbon_intensity": self.carbon_intensity,
            "grid_kwh": grid_energy, "carbon": self.metrics["carbon"],
            "cost": self.metrics["cost"], "completed": self.metrics["completed"],
        }
        return self._obs(), reward, terminated, False, info


# Backwards-compatible import name for existing callers.
GeothermalEnvAdvanced = CarbonAwareComputeEnv
