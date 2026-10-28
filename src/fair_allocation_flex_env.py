import functools

import numpy as np
from gymnasium import spaces

from disaster_uav_env import (
    SERVICE_RADIUS_M, STEP_PENALTY, CRASH_PENALTY, MAX_EPISODE_STEPS, DT, SHAPING_COEF
)
from fair_allocation_env import FairAllocationUAVEnv
from fair_allocation import BATTERY_CRITICAL_THRESHOLD
from ppes import prune_candidates

N_OPTIONS = 3


class FairAllocationFlexUAVEnv(FairAllocationUAVEnv):

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):

        dim = 5 + N_OPTIONS * 5
        return spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

    def _option_set(self, agent):
        """Manager'ın bu ajana sunduğu aday listesi. İlk sırada ana atama
        (assigned_target), sonrasında aynı profile göre bir sonraki en iyi
        güvenli adaylar."""
        uav = self.uavs[agent]
        primary = self.assigned_target.get(agent)

        remaining_steps = max(0, MAX_EPISODE_STEPS - self.step_count)
        remaining_time_s = remaining_steps * DT

        unserved = np.where(~self.served)[0]
        if len(unserved) == 0:
            return []


        taken = set(v for a, v in self.assigned_target.items() if a != agent and v is not None)
        pool = np.array([i for i in unserved if i not in taken])
        if len(pool) == 0:
            return [primary] if primary is not None else []

        safe = prune_candidates(
            uav.state.position[:2], uav.state.battery_pct, pool.tolist(),
            self.demand_xy, self.depot_xy, self.uav_config, self.odd,
            remaining_time_s=remaining_time_s
        )
        if not safe:
            return [primary] if primary is not None else []

        safe = np.array(safe)
        dists = np.linalg.norm(self.demand_xy[safe] - uav.state.position[:2], axis=1)

        if uav.state.battery_pct < BATTERY_CRITICAL_THRESHOLD:
            from_depot = np.linalg.norm(self.demand_xy[safe] - self.depot_xy, axis=1)
            score = -(dists + from_depot)
        else:
            score = self.demand_weight[safe] / (dists + 100.0)

        order = np.argsort(-score)
        ranked = list(safe[order])

        options = []
        if primary is not None and not self.served[primary]:
            options.append(primary)
        for idx in ranked:
            if len(options) >= N_OPTIONS:
                break
            if idx not in options:
                options.append(int(idx))
        return options

    def _get_obs(self, agent):
        uav = self.uavs[agent]
        rel_pos = (uav.state.position[:2] - self.depot_xy) / self.pos_scale
        vel = uav.state.velocity[:2] / self.uav_config.max_speed_mps
        battery = np.array([uav.state.battery_pct / 100.0])

        options = self._option_set(agent) if hasattr(self, "assigned_target") else []

        info_list = []
        for idx in options[:N_OPTIONS]:
            delta = self.demand_xy[idx] - uav.state.position[:2]
            dist = np.linalg.norm(delta)
            direction = delta / dist if dist > 1e-6 else np.zeros(2)
            dist_norm = np.log1p(dist) / np.log1p(self.pos_scale)
            info_list.append([direction[0], direction[1], dist_norm,
                              self.demand_weight[idx], self.demand_confidence[idx]])
        while len(info_list) < N_OPTIONS:
            info_list.append([0.0, 0.0, 0.0, 0.0, 0.0])

        obs = np.concatenate([rel_pos, vel, battery,
                              np.array(info_list, dtype=np.float32).flatten()]).astype(np.float32)
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


            options = self._option_set(agent)
            dists_before = {}
            for idx in options:
                if not self.served[idx]:
                    dists_before[idx] = np.linalg.norm(self.demand_xy[idx] - uav.state.position[:2])

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


            best_gain = 0.0
            for idx, d_before in dists_before.items():
                if self.served[idx]:
                    continue
                d_after = np.linalg.norm(self.demand_xy[idx] - uav.state.position[:2])
                gain = d_before - d_after
                if gain > best_gain:
                    best_gain = gain
            rewards[agent] += SHAPING_COEF * best_gain

            for idx in list(dists_before.keys()):
                if self.served[idx]:
                    continue
                d_after = np.linalg.norm(self.demand_xy[idx] - uav.state.position[:2])
                if d_after <= SERVICE_RADIUS_M:
                    self.served[idx] = True
                    rewards[agent] += self.demand_weight[idx] * 100.0
                    if self.assigned_target.get(agent) == idx:
                        self.assigned_target[agent] = None
                    break

        if self.step_count % 10 == 0:
            self._run_manager_assignment()

        all_served = self.served.all()
        max_steps_reached = self.step_count >= MAX_EPISODE_STEPS
        if all_served or max_steps_reached:
            for agent in self.agents:
                truncations[agent] = True

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        self.agents = [a for a in self.agents if not (terminations[a] or truncations[a])]
        for a in list(self.assigned_target.keys()):
            if a not in self.agents:
                self.assigned_target.pop(a, None)

        infos["__common__"] = {
            "n_served": int(self.served.sum()),
            "n_total": self.n_demand,
            "step": self.step_count,
        }
        return observations, rewards, terminations, truncations, infos
