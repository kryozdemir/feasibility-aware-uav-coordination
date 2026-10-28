import numpy as np

from disaster_uav_env import (
    DisasterUAVEnv, STEP_PENALTY, CRASH_PENALTY, MAX_EPISODE_STEPS, DT,
    SHAPING_COEF, SERVICE_RADIUS_M
)

COVERAGE_BONUS = 5.0
COVERAGE_SHAPING_COEF = 0.002


class CoverageRewardUAVEnv(DisasterUAVEnv):

    def __init__(self, n_agents: int = 3, seed: int = None, region_labels: np.ndarray = None,
                 coverage_bonus: float = COVERAGE_BONUS, coverage_shaping_coef: float = COVERAGE_SHAPING_COEF):
        assert region_labels is not None
        self.coverage_bonus = coverage_bonus
        self.coverage_shaping_coef = coverage_shaping_coef
        self.region_labels = region_labels
        self.n_regions = int(region_labels.max()) + 1
        super().__init__(n_agents=n_agents, seed=seed)
        self.region_total_value = np.zeros(self.n_regions)
        for r in range(self.n_regions):
            m = region_labels == r
            self.region_total_value[r] = np.sum(self.demand_weight[m] * self.demand_confidence[m])
        self.dominant_region = int(np.argmax(self.region_total_value))

    def _under_covered_points(self):
        """Az kapsanan bölgelerdeki servis edilmemiş noktalar."""
        unserved = np.where(~self.served)[0]
        if len(unserved) == 0:
            return np.array([], dtype=int)
        rem = np.zeros(self.n_regions)
        for idx in unserved:
            rem[self.region_labels[idx]] += self.demand_weight[idx] * self.demand_confidence[idx]
        cov = 1.0 - np.divide(rem, self.region_total_value,
                              out=np.ones(self.n_regions), where=self.region_total_value > 0)
        eligible = [r for r in range(self.n_regions) if r != self.dominant_region]
        if cov[eligible].min() >= 0.95:
            return np.array([], dtype=int)
        under = [r for r in eligible if cov[r] <= np.median(cov[eligible])]
        return unserved[np.isin(self.region_labels[unserved], under)]

    def step(self, actions):
        self.step_count += 1
        rewards = {agent: STEP_PENALTY for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        wind_vector = np.zeros(3)

        under_pts = self._under_covered_points()

        for agent in self.agents:
            uav = self.uavs[agent]
            if uav.state.crashed or uav.state.battery_depleted:
                terminations[agent] = True
                continue

            unserved_before = np.where(~self.served)[0]
            shaping_target = None
            dist_before = None
            if len(unserved_before) > 0:
                d = np.linalg.norm(self.demand_xy[unserved_before] - uav.state.position[:2], axis=1)
                i = int(np.argmin(d))
                shaping_target = unserved_before[i]
                dist_before = d[i]


            cov_target = None
            cov_dist_before = None
            if len(under_pts) > 0:
                d2 = np.linalg.norm(self.demand_xy[under_pts] - uav.state.position[:2], axis=1)
                j = int(np.argmin(d2))
                cov_target = under_pts[j]
                cov_dist_before = d2[j]

            action = np.clip(actions[agent], -1.0, 1.0)
            accel = np.array([action[0], action[1], 0.0]) * self.uav_config.max_accel_mps2
            uav.step(accel, wind_vector=wind_vector, dt=DT)

            if uav.state.crashed:
                rewards[agent] += CRASH_PENALTY
                terminations[agent] = True
                continue
            if uav.state.battery_depleted:
                terminations[agent] = True
                continue

            if shaping_target is not None and not self.served[shaping_target]:
                d_after = np.linalg.norm(self.demand_xy[shaping_target] - uav.state.position[:2])
                rewards[agent] += SHAPING_COEF * (dist_before - d_after)
                if d_after <= SERVICE_RADIUS_M:
                    self.served[shaping_target] = True
                    rewards[agent] += self.demand_weight[shaping_target] * 100.0

            if cov_target is not None and not self.served[cov_target]:
                d_after = np.linalg.norm(self.demand_xy[cov_target] - uav.state.position[:2])
                rewards[agent] += self.coverage_shaping_coef * (cov_dist_before - d_after)
                if d_after <= SERVICE_RADIUS_M:
                    self.served[cov_target] = True
                    rewards[agent] += self.demand_weight[cov_target] * 100.0 + self.coverage_bonus

        all_served = self.served.all()
        if all_served or self.step_count >= MAX_EPISODE_STEPS:
            for agent in self.agents:
                truncations[agent] = True

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        self.agents = [a for a in self.agents if not (terminations[a] or truncations[a])]

        infos["__common__"] = {"n_served": int(self.served.sum()),
                                "n_total": self.n_demand, "step": self.step_count}
        return observations, rewards, terminations, truncations, infos
