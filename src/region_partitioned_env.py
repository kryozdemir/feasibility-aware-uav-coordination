"""
TEŞHİS/DÜZELTME v3: Sabit Bölge Görevlendirmesi.

Önceki iki deneme (sabit bonus, dinamik "en az kapsanan bölge" şekillendirmesi)
başarısız oldu -- ya keşif sorunu ya da ödül kararsızlığı yüzünden.

Bu versiyon farklı bir mantık kullanıyor: DEĞERİ dengelemeye çalışmıyor
(zaten imkansız -- bir bölge tüm değerin %87'sini taşıyor), bunun yerine
COĞRAFİ KAPSAMAYI GARANTİ ALTINA ALIYOR: her bölüm başında, 6 bölge 3
UAV'a sabit şekilde (2'şer bölge) dağıtılıyor -- hangi ajanın hangi
bölgeleri alacağı her bölümde rastgele karıştırılıyor (paylaşımlı
politikanın hepsini öğrenebilmesi için).

Bu atama bölüm SÜRESİNCE SABİT kalıyor (önceki dinamik versiyondaki
kararsızlığı önlüyor). Bir ajanın gözlemi ve şekillendirmesi SADECE
kendi bölgesindeki noktalarla sınırlı -- kendi bölgesi tükenirse
serbest kalıyor (tüm noktalara bakabiliyor).
"""

import numpy as np
from gymnasium import spaces
import functools

from disaster_uav_env import (
    DisasterUAVEnv, K_NEAREST, STEP_PENALTY, CRASH_PENALTY,
    MAX_EPISODE_STEPS, DT, SHAPING_COEF, SERVICE_RADIUS_M
)

N_REGIONS = 6


class RegionPartitionedUAVEnv(DisasterUAVEnv):

    def __init__(self, n_agents: int = 3, seed: int = None, region_labels: np.ndarray = None, max_episode_steps: int = None):
        assert region_labels is not None
        self.region_labels = region_labels
        self.n_regions = int(region_labels.max()) + 1
        self.max_episode_steps_override = max_episode_steps or MAX_EPISODE_STEPS
        super().__init__(n_agents=n_agents, seed=seed)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):


        dim = 5 + K_NEAREST * 4 + 1
        return spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.agent_regions = {}
        observations, infos = super().reset(seed=seed, options=options)


        region_order = np.random.permutation(self.n_regions)
        self.agent_regions = {}
        regions_per_agent = self.n_regions // len(self.agents)
        for i, agent in enumerate(self.agents):
            start = i * regions_per_agent
            end = start + regions_per_agent if i < len(self.agents) - 1 else self.n_regions
            self.agent_regions[agent] = set(region_order[start:end].tolist())

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        return observations, infos

    def _candidate_pool(self, agent):
        """Ajanın kendi bölgesindeki servis edilmemiş noktalar; boşsa tüm noktalar (serbest kalır)."""
        unserved_idx = np.where(~self.served)[0]
        if len(unserved_idx) == 0:
            return unserved_idx
        own_regions = self.agent_regions.get(agent, set())
        in_region_mask = np.isin(self.region_labels[unserved_idx], list(own_regions))
        pool = unserved_idx[in_region_mask]
        if len(pool) == 0:
            return unserved_idx
        return pool

    def _get_obs(self, agent):
        uav = self.uavs[agent]
        rel_pos = (uav.state.position[:2] - self.depot_xy) / self.pos_scale
        vel = uav.state.velocity[:2] / self.uav_config.max_speed_mps
        battery = np.array([uav.state.battery_pct / 100.0])

        pool = self._candidate_pool(agent)
        if len(pool) == 0:
            nearest_info = np.zeros(K_NEAREST * 4, dtype=np.float32)
            own_region_value = 0.0
        else:
            deltas = self.demand_xy[pool] - uav.state.position[:2]
            dists = np.linalg.norm(deltas, axis=1)
            order = np.argsort(dists)[:K_NEAREST]
            chosen = pool[order]
            info_list = []
            for idx in chosen:
                dx, dy = (self.demand_xy[idx] - uav.state.position[:2]) / self.pos_scale
                w = self.demand_weight[idx]
                c = self.demand_confidence[idx]
                info_list.append([dx, dy, w, c])
            while len(info_list) < K_NEAREST:
                info_list.append([0.0, 0.0, 0.0, 0.0])
            nearest_info = np.array(info_list, dtype=np.float32).flatten()
            own_region_value = np.sum(self.demand_weight[pool] * self.demand_confidence[pool])

        obs = np.concatenate([rel_pos, vel, battery, nearest_info, [own_region_value]]).astype(np.float32)
        return obs

    def step(self, actions):
        self.step_count += 1
        rewards = {agent: STEP_PENALTY for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        wind_vector = np.zeros(3)

        for agent in self.agents:
            uav = self.uavs[agent]
            if uav.state.crashed or uav.state.battery_depleted:
                terminations[agent] = True
                continue

            pool_before = self._candidate_pool(agent)
            shaping_target_idx = None
            dist_before = None
            if len(pool_before) > 0:
                deltas_before = self.demand_xy[pool_before] - uav.state.position[:2]
                dists_before = np.linalg.norm(deltas_before, axis=1)
                nearest_before = np.argmin(dists_before)
                shaping_target_idx = pool_before[nearest_before]
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
                terminations[agent] = True
                continue

            if shaping_target_idx is not None and not self.served[shaping_target_idx]:
                dist_after = np.linalg.norm(self.demand_xy[shaping_target_idx] - uav.state.position[:2])
                rewards[agent] += SHAPING_COEF * (dist_before - dist_after)
                if dist_after <= SERVICE_RADIUS_M:
                    self.served[shaping_target_idx] = True
                    rewards[agent] += self.demand_weight[shaping_target_idx] * self.demand_confidence[shaping_target_idx] * 100.0

        all_served = self.served.all()
        max_steps_reached = self.step_count >= self.max_episode_steps_override
        if all_served or max_steps_reached:
            for agent in self.agents:
                truncations[agent] = True

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        self.agents = [a for a in self.agents if not (terminations[a] or truncations[a])]

        regions_visited = set()
        served_idx = np.where(self.served)[0]
        for idx in served_idx:
            regions_visited.add(int(self.region_labels[idx]))

        infos["__common__"] = {
            "n_served": int(self.served.sum()),
            "n_total": self.n_demand,
            "step": self.step_count,
            "n_regions_visited": len(regions_visited),
        }
        return observations, rewards, terminations, truncations, infos
