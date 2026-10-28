"""
K4b + kapsama garantisi hibriti.

K4b'nin esnekliğini (worker'a K aday gösterip serbest seçim) koruyorum,
ama gösterilen K adaydan birini her zaman "en az kapsanan bölgedeki en
yakın güvenli nokta" olarak ayırıyorum -- ve o bölgeden bir şey
tamamlayınca küçük bir bonus ekliyorum.

Bunu neden yapıyorum: K4b'yi ilk denediğimde (sadece gözlem düzeltmesiyle)
değer inanılmaz arttı (0.236) ama potential kapsama tam sıfıra düştü --
worker'a uzak/düşük-değerli adayı gösteriyordum ama gitmesi için hiçbir
sebep yoktu. Bu sefer hem görünürlük hem teşvik veriyorum.

İlk region-bonus denemem (region_bonus_env.py) işe yaramamıştı çünkü
ajan hiç o bölgeye gitmediği için bonusun var olduğunu deneyimleyip
öğrenemiyordu (keşif sorunu). Bu sefer aday doğrudan gözlemde (yön+mesafe
olarak) gösteriliyor, yani rastgele keşfe bağımlı değil -- ajan nereye
gideceğini zaten biliyor, sadece gitmeye değer olduğunu öğrenmesi lazım.
"""

import numpy as np

from flexible_fair_env import FlexibleFairUAVEnv
from disaster_uav_env import (
    K_NEAREST, STEP_PENALTY, CRASH_PENALTY, MAX_EPISODE_STEPS, DT,
    SHAPING_COEF, SERVICE_RADIUS_M
)
from ppes import prune_candidates

COVERAGE_BONUS = 3.0


class FlexibleFairCoverageUAVEnv(FlexibleFairUAVEnv):

    def __init__(self, n_agents: int = 3, seed: int = None, odd=None, region_labels: np.ndarray = None):
        super().__init__(n_agents=n_agents, seed=seed, odd=odd)
        self.region_labels = region_labels
        if region_labels is not None:
            n_regions = int(region_labels.max()) + 1
            self.region_total_value = np.zeros(n_regions)
            for r in range(n_regions):
                mask = region_labels == r
                self.region_total_value[r] = np.sum(self.demand_weight[mask] * self.demand_confidence[mask])
            self.dominant_region = int(np.argmax(self.region_total_value))
        else:
            self.region_total_value = None
            self.dominant_region = None

    def _under_covered_regions(self):
        """Şu anki duruma göre az kapsanmış (baskın hariç) bölgeleri döndürür.
        fair_allocation.py'deki medyan mantığıyla aynı -- sabit bir yüzde
        yerine o anki dağılıma göre kendini ayarlıyor."""
        unserved_idx = np.where(~self.served)[0]
        n_regions = len(self.region_total_value)
        region_remaining_value = np.zeros(n_regions)
        for idx in unserved_idx:
            r = self.region_labels[idx]
            region_remaining_value[r] += self.demand_weight[idx] * self.demand_confidence[idx]
        region_coverage = 1.0 - np.divide(
            region_remaining_value, self.region_total_value,
            out=np.ones(n_regions), where=self.region_total_value > 0
        )
        eligible = [r for r in range(n_regions) if r != self.dominant_region]
        if not eligible or region_coverage[eligible].min() >= 0.95:
            return []
        return [r for r in eligible if region_coverage[r] <= np.median(region_coverage[eligible])]

    def _get_obs(self, agent):
        uav = self.uavs[agent]
        rel_pos = (uav.state.position[:2] - self.depot_xy) / self.pos_scale
        vel = uav.state.velocity[:2] / self.uav_config.max_speed_mps
        battery = np.array([uav.state.battery_pct / 100.0])

        unserved_idx = np.array(list(np.where(~self.served)[0]))
        remaining_steps = max(0, MAX_EPISODE_STEPS - self.step_count)
        remaining_time_s = remaining_steps * DT

        info_list = []

        if len(unserved_idx) > 0:
            safe_candidates = prune_candidates(
                uav.state.position[:2], uav.state.battery_pct, unserved_idx.tolist(),
                self.demand_xy, self.depot_xy, self.uav_config, self.odd,
                remaining_time_s=remaining_time_s
            )
            safe_candidates = np.array(safe_candidates) if safe_candidates else np.array([], dtype=int)


            coverage_slot_idx = None
            if self.region_labels is not None and len(safe_candidates) > 0:
                under_covered = self._under_covered_regions()
                if under_covered:
                    mask = np.isin(self.region_labels[safe_candidates], under_covered)
                    region_safe = safe_candidates[mask]
                    if len(region_safe) > 0:
                        d = np.linalg.norm(self.demand_xy[region_safe] - uav.state.position[:2], axis=1)
                        coverage_slot_idx = region_safe[np.argmin(d)]


            n_value_slots = K_NEAREST - (1 if coverage_slot_idx is not None else 0)
            chosen = []
            if len(safe_candidates) > 0:
                dists = np.linalg.norm(self.demand_xy[safe_candidates] - uav.state.position[:2], axis=1)
                values = self.demand_weight[safe_candidates] * self.demand_confidence[safe_candidates]
                density_score = values / (dists + 100.0)
                order = np.argsort(-density_score)
                for idx in safe_candidates[order]:
                    if coverage_slot_idx is not None and idx == coverage_slot_idx:
                        continue
                    chosen.append(idx)
                    if len(chosen) >= n_value_slots:
                        break

            if coverage_slot_idx is not None:
                chosen.append(coverage_slot_idx)

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

        obs = np.concatenate([rel_pos, vel, battery, nearest_info]).astype(np.float32)
        return obs

    def step(self, actions):
        self.step_count += 1
        rewards = {agent: STEP_PENALTY for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        wind_vector = np.zeros(3)


        under_covered_before = self._under_covered_regions() if self.region_labels is not None else []

        for agent in self.agents:
            uav = self.uavs[agent]
            if uav.state.crashed or uav.state.battery_depleted:
                terminations[agent] = True
                continue

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
                terminations[agent] = True
                continue

            if shaping_target_idx is not None and not self.served[shaping_target_idx]:
                dist_after = np.linalg.norm(self.demand_xy[shaping_target_idx] - uav.state.position[:2])
                rewards[agent] += SHAPING_COEF * (dist_before - dist_after)

                if dist_after <= SERVICE_RADIUS_M:
                    self.served[shaping_target_idx] = True
                    rewards[agent] += self.demand_weight[shaping_target_idx] * self.demand_confidence[shaping_target_idx] * 100.0

                    if self.region_labels is not None and self.region_labels[shaping_target_idx] in under_covered_before:
                        rewards[agent] += COVERAGE_BONUS

        all_served = self.served.all()
        max_steps_reached = self.step_count >= MAX_EPISODE_STEPS
        if all_served or max_steps_reached:
            for agent in self.agents:
                truncations[agent] = True

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        self.agents = [a for a in self.agents if not (terminations[a] or truncations[a])]

        infos["__common__"] = {
            "n_served": int(self.served.sum()),
            "n_total": self.n_demand,
            "step": self.step_count,
        }
        return observations, rewards, terminations, truncations, infos
