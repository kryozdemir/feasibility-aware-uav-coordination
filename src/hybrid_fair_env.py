
import functools

import numpy as np
from gymnasium import spaces

from disaster_uav_env import (
    DisasterUAVEnv, K_NEAREST, STEP_PENALTY, CRASH_PENALTY,
    MAX_EPISODE_STEPS, DT, SHAPING_COEF, SERVICE_RADIUS_M
)
from ppes import ODDParams, prune_candidates

COVERAGE_SHAPING_COEF = 0.002
COVERAGE_BONUS = 1.0


class HybridFairUAVEnv(DisasterUAVEnv):

    def __init__(self, n_agents: int = 3, seed: int = None, odd: ODDParams = None,
                 region_labels: np.ndarray = None):
        self.odd = odd or ODDParams()
        self.region_labels = region_labels
        super().__init__(n_agents=n_agents, seed=seed)

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

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        dim = 5 + K_NEAREST * 5
        return spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

    def _get_top_candidates(self, uav, remaining_time_s):
        unserved_idx = np.where(~self.served)[0]
        if len(unserved_idx) == 0:
            return []
        safe = prune_candidates(
            uav.state.position[:2], uav.state.battery_pct, unserved_idx.tolist(),
            self.demand_xy, self.depot_xy, self.uav_config, self.odd,
            remaining_time_s=remaining_time_s
        )
        if not safe:
            return []
        safe = np.array(safe)
        dists = np.linalg.norm(self.demand_xy[safe] - uav.state.position[:2], axis=1)
        values = self.demand_weight[safe] * self.demand_confidence[safe]
        score = values / (dists + 100.0)
        order = np.argsort(-score)[:K_NEAREST]
        return safe[order].tolist()

    def _find_coverage_candidate(self, remaining_time_s):
        if self.region_labels is None:
            return None, None
        unserved_idx = np.where(~self.served)[0]
        if len(unserved_idx) == 0:
            return None, None

        n_regions = len(self.region_total_value)
        region_remaining = np.zeros(n_regions)
        for idx in unserved_idx:
            r = self.region_labels[idx]
            region_remaining[r] += self.demand_weight[idx] * self.demand_confidence[idx]
        region_coverage = 1.0 - np.divide(
            region_remaining, self.region_total_value,
            out=np.ones(n_regions), where=self.region_total_value > 0
        )
        eligible = [r for r in range(n_regions) if r != self.dominant_region]
        if not eligible or region_coverage[eligible].min() >= 0.95:
            return None, None
        under_covered = [r for r in eligible if region_coverage[r] <= np.median(region_coverage[eligible])]
        candidates = unserved_idx[np.isin(self.region_labels[unserved_idx], under_covered)]
        if len(candidates) == 0:
            return None, None

        best_agent, best_dist, best_target = None, None, None
        for agent in self.agents:
            uav = self.uavs[agent]
            safe = prune_candidates(
                uav.state.position[:2], uav.state.battery_pct, candidates.tolist(),
                self.demand_xy, self.depot_xy, self.uav_config, self.odd,
                remaining_time_s=remaining_time_s
            )
            if not safe:
                continue
            d = np.linalg.norm(self.demand_xy[safe] - uav.state.position[:2], axis=1)
            i = int(np.argmin(d))
            if best_dist is None or d[i] < best_dist:
                best_dist, best_agent, best_target = d[i], agent, safe[i]
        return best_agent, best_target

    def _get_obs(self, agent):
        uav = self.uavs[agent]
        rel_pos = (uav.state.position[:2] - self.depot_xy) / self.pos_scale
        vel = uav.state.velocity[:2] / self.uav_config.max_speed_mps
        battery = np.array([uav.state.battery_pct / 100.0])

        remaining_steps = max(0, MAX_EPISODE_STEPS - self.step_count)
        remaining_time_s = remaining_steps * DT

        chosen = self._get_top_candidates(uav, remaining_time_s)

        cov_agent, cov_target = self._find_coverage_candidate(remaining_time_s)
        if cov_agent == agent and cov_target is not None and cov_target not in chosen:
            chosen = chosen[:K_NEAREST - 1] + [cov_target]

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

        obs = np.concatenate([rel_pos, vel, battery, nearest_info]).astype(np.float32)
        return obs

    def step(self, actions):
        self.step_count += 1
        rewards = {agent: STEP_PENALTY for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        wind_vector = np.zeros(3)

        remaining_steps = max(0, MAX_EPISODE_STEPS - self.step_count)
        remaining_time_s = remaining_steps * DT
        cov_agent, cov_target = self._find_coverage_candidate(remaining_time_s)

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

            cov_dist_before = None
            if agent == cov_agent and cov_target is not None and not self.served[cov_target]:
                cov_dist_before = np.linalg.norm(self.demand_xy[cov_target] - uav.state.position[:2])

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

            if cov_dist_before is not None and not self.served[cov_target]:
                cov_dist_after = np.linalg.norm(self.demand_xy[cov_target] - uav.state.position[:2])
                rewards[agent] += COVERAGE_SHAPING_COEF * (cov_dist_before - cov_dist_after)
                if cov_dist_after <= SERVICE_RADIUS_M:
                    self.served[cov_target] = True
                    rewards[agent] += self.demand_weight[cov_target] * self.demand_confidence[cov_target] * 100.0
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
