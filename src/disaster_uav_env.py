from pathlib import Path
import functools

import numpy as np
import pickle
import geopandas as gpd
from pettingzoo import ParallelEnv
from gymnasium import spaces

from uav_model import KinematicUAV, UAVConfig

PROC_DIR = Path(__file__).parent.parent / "data" / "processed"

K_NEAREST = 5
SERVICE_RADIUS_M = 50.0
STEP_PENALTY = -0.001
CRASH_PENALTY = -5.0
MAX_EPISODE_STEPS = 900


DT = 2.0
SHAPING_COEF = 0.002
DEPOT_RADIUS_M = 200.0


ROTATING_FLEET = True
SWAP_DELAY_STEPS = 30
RETURN_PENALTY = -3.0
BATTERY_LOSS_PENALTY = -10.0


class DisasterUAVEnv(ParallelEnv):
    metadata = {"name": "disaster_uav_v0"}

    def __init__(self, n_agents: int = 3, seed: int = None, demand_file: str = "demand_points_v2.geojson"):
        self.n_agents = n_agents
        self.demand_file = demand_file
        self.possible_agents = [f"uav_{i}" for i in range(n_agents)]
        self.agents = self.possible_agents[:]
        self.rng = np.random.default_rng(seed)

        self._load_data()
        self.uav_config = UAVConfig()


        self.pos_scale = 20000.0

    def _load_data(self):
        with open(PROC_DIR / "road_network.pickle", "rb") as f:
            road_data = pickle.load(f)
        self.depot_xy = np.array(road_data["depot_xy_utm"])

        demand = gpd.read_file(PROC_DIR / self.demand_file).to_crs(road_data["crs"])
        self.demand_xy = np.stack([demand.geometry.x.values, demand.geometry.y.values], axis=1)


        self.demand_weight = demand["weight_normalized"].values.astype(float)
        self.demand_confidence = demand["confidence"].values.astype(float)
        self.n_demand = len(demand)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):


        dim = 6 + K_NEAREST * 5
        return spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def _sample_wind(self):


        speed = float(self.rng.uniform(0.0, self.uav_config.max_wind_speed_mps))
        direction = float(self.rng.uniform(0, 2*np.pi))
        self.wind_base = np.array([speed*np.cos(direction), speed*np.sin(direction), 0.0])

    def _current_wind(self):
        gust = self.rng.normal(0.0, 0.8, size=3); gust[2] = 0.0
        return self.wind_base + gust

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.served = np.zeros(self.n_demand, dtype=bool)
        self._sample_wind()

        self.uavs = {}
        for i, agent in enumerate(self.agents):

            offset = self.rng.normal(0, 5, size=2)
            start_xy = self.depot_xy + offset
            start_pos = np.array([start_xy[0], start_xy[1], 60.0])
            self.uavs[agent] = KinematicUAV(self.uav_config, start_pos, rng=self.rng)

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _get_obs(self, agent):
        uav = self.uavs[agent]
        rel_pos = (uav.state.position[:2] - self.depot_xy) / self.pos_scale
        vel = uav.state.velocity[:2] / self.uav_config.max_speed_mps
        battery = np.array([uav.state.battery_pct / 100.0])
        payload = np.array([uav.state.payload_remaining / self.uav_config.payload_capacity])


        unserved_idx = np.where(~self.served)[0]
        if len(unserved_idx) == 0:
            nearest_info = np.zeros(K_NEAREST * 4, dtype=np.float32)
        else:
            deltas = self.demand_xy[unserved_idx] - uav.state.position[:2]
            dists = np.linalg.norm(deltas, axis=1)
            order = np.argsort(dists)[:K_NEAREST]
            chosen = unserved_idx[order]


            info_list = []
            for idx in chosen:
                delta = self.demand_xy[idx] - uav.state.position[:2]
                dist = np.linalg.norm(delta)
                direction = delta / dist if dist > 1e-6 else np.zeros(2)
                dist_norm = np.log1p(dist) / np.log1p(self.pos_scale)
                w = self.demand_weight[idx]
                c = self.demand_confidence[idx]
                info_list.append([direction[0], direction[1], dist_norm, w, c])
            while len(info_list) < K_NEAREST:
                info_list.append([0.0, 0.0, 0.0, 0.0, 0.0])
            nearest_info = np.array(info_list, dtype=np.float32).flatten()

        obs = np.concatenate([rel_pos, vel, battery, payload, nearest_info]).astype(np.float32)
        return obs

    def get_global_state(self):
        """
        Merkezi kritik için global state -- TÜM olası ajanların (possible_agents)
        gözlemlerinin birleşimi, sabit boyutlu (terminated/truncated ajanlar
        için son bilinen gözlem ya da sıfır kullanılır, boyut hep aynı kalsın).
        """
        parts = []
        for agent in self.possible_agents:
            if agent in self.uavs:
                parts.append(self._get_obs(agent))
            else:
                parts.append(np.zeros(self.observation_space(agent).shape[0], dtype=np.float32))
        return np.concatenate(parts)

    @functools.lru_cache(maxsize=None)
    def global_state_dim(self):
        return self.observation_space(self.possible_agents[0]).shape[0] * len(self.possible_agents)

    def step(self, actions):
        self.step_count += 1
        rewards = {agent: STEP_PENALTY for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        wind_vector = self._current_wind()

        for agent in self.agents:
            uav = self.uavs[agent]
            if uav.state.crashed or uav.state.battery_depleted:
                terminations[agent] = True
                continue


            prev_pos_xy = uav.state.position[:2].copy()
            unserved_idx_before = np.where(~self.served)[0]
            shaping_target_idx = None
            dist_before = None
            if len(unserved_idx_before) > 0:
                deltas_before = self.demand_xy[unserved_idx_before] - uav.state.position[:2]
                dists_before = np.linalg.norm(deltas_before, axis=1)
                nearest_before = np.argmin(dists_before)
                shaping_target_idx = unserved_idx_before[nearest_before]
                dist_before = dists_before[nearest_before]

            action = np.clip(actions[agent], -1.0, 1.0)
            accel_2d = action * self.uav_config.max_accel_mps2
            accel = np.array([accel_2d[0], accel_2d[1], 0.0])

            uav.step(accel, wind_vector=wind_vector, dt=DT)

            if uav.state.crashed:
                rewards[agent] += CRASH_PENALTY
                terminations[agent] = True
                continue
            if uav.state.battery_depleted:
                rewards[agent] += BATTERY_LOSS_PENALTY
                terminations[agent] = True
                continue


            if uav.state.payload_remaining <= 0 or len(unserved_idx_before) == 0:
                d_before = np.linalg.norm(self.depot_xy - prev_pos_xy)
                d_after = np.linalg.norm(self.depot_xy - uav.state.position[:2])
                rewards[agent] += SHAPING_COEF * (d_before - d_after)
                if d_after <= DEPOT_RADIUS_M:
                    uav.state.payload_remaining = self.uav_config.payload_capacity
                    uav.state.battery_pct = 100.0
                continue


            if shaping_target_idx is not None and not self.served[shaping_target_idx]:
                dist_after = np.linalg.norm(self.demand_xy[shaping_target_idx] - uav.state.position[:2])
                rewards[agent] += SHAPING_COEF * (dist_before - dist_after)


            unserved_idx = np.where(~self.served)[0]
            if len(unserved_idx) > 0:
                deltas = self.demand_xy[unserved_idx] - uav.state.position[:2]
                dists = np.linalg.norm(deltas, axis=1)
                closest = np.argmin(dists)
                if dists[closest] <= SERVICE_RADIUS_M:
                    real_idx = unserved_idx[closest]
                    self.served[real_idx] = True


                    rewards[agent] += self.demand_weight[real_idx] * 100.0
                    uav.state.payload_remaining -= 1


        all_served = self.served.all()
        max_steps_reached = self.step_count >= MAX_EPISODE_STEPS
        if all_served or max_steps_reached:
            for agent in self.agents:
                d_home = np.linalg.norm(self.uavs[agent].state.position[:2] - self.depot_xy)
                if d_home > DEPOT_RADIUS_M:
                    rewards[agent] += RETURN_PENALTY * min(1.0, d_home / 10000.0)
                truncations[agent] = True

        observations = {agent: self._get_obs(agent) for agent in self.agents}


        self.agents = [
            a for a in self.agents if not (terminations[a] or truncations[a])
        ]

        infos["__common__"] = {
            "n_served": int(self.served.sum()),
            "n_total": self.n_demand,
            "step": self.step_count,
        }

        return observations, rewards, terminations, truncations, infos
