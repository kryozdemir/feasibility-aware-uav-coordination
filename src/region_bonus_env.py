
import numpy as np

from disaster_uav_env import DisasterUAVEnv, STEP_PENALTY, CRASH_PENALTY, MAX_EPISODE_STEPS, DT, SHAPING_COEF, SERVICE_RADIUS_M

REGION_BONUS = 2.0


class RegionBonusUAVEnv(DisasterUAVEnv):

    def __init__(self, n_agents: int = 3, seed: int = None, region_labels: np.ndarray = None):
        assert region_labels is not None, "region_labels dışarıdan verilmeli (tüm koşullarda tutarlı olsun diye)"
        self.region_labels = region_labels
        self.n_regions = int(region_labels.max()) + 1
        super().__init__(n_agents=n_agents, seed=seed)

    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed=seed, options=options)
        self.region_visited = np.zeros(self.n_regions, dtype=bool)
        return observations, infos

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

            unserved_idx_before = np.where(~self.served)[0]
            shaping_target_idx = None
            dist_before = None
            if len(unserved_idx_before) > 0:


                region_coverage = np.zeros(self.n_regions)
                for r in range(self.n_regions):
                    total_r = (self.region_labels == r).sum()
                    served_r = ((self.region_labels == r) & self.served).sum()
                    region_coverage[r] = served_r / total_r if total_r > 0 else 1.0
                least_covered_region = np.argmin(region_coverage)

                candidates_in_region = unserved_idx_before[self.region_labels[unserved_idx_before] == least_covered_region]
                if len(candidates_in_region) == 0:
                    candidates_in_region = unserved_idx_before

                deltas_before = self.demand_xy[candidates_in_region] - uav.state.position[:2]
                dists_before = np.linalg.norm(deltas_before, axis=1)
                nearest_before = np.argmin(dists_before)
                shaping_target_idx = candidates_in_region[nearest_before]
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


                    region = self.region_labels[shaping_target_idx]
                    if not self.region_visited[region]:
                        self.region_visited[region] = True
                        rewards[agent] += REGION_BONUS

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
            "n_regions_visited": int(self.region_visited.sum()),
        }
        return observations, rewards, terminations, truncations, infos
