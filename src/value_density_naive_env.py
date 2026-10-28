
import numpy as np

from hierarchical_env import NaiveManagerUAVEnv
from disaster_uav_env import MAX_EPISODE_STEPS, DT


class ValueDensityNaiveUAVEnv(NaiveManagerUAVEnv):

    def _run_manager_assignment(self):
        unserved_idx = list(np.where(~self.served)[0])
        for agent in self.agents:
            if self.assigned_target[agent] is not None and self.served[self.assigned_target[agent]]:
                self.assigned_target[agent] = None

        idle_agents = [a for a in self.agents
                       if self.assigned_target[a] is None and self.uavs[a].state.payload_remaining > 0]
        if not idle_agents:
            return

        already_assigned = set(v for v in self.assigned_target.values() if v is not None)
        available = [i for i in unserved_idx if i not in already_assigned]

        remaining_steps = max(0, MAX_EPISODE_STEPS - self.step_count)
        max_reachable_m = remaining_steps * DT * self.uav_config.max_speed_mps * 0.8

        for agent in idle_agents:
            if not available:
                self.assigned_target[agent] = None
                continue
            uav = self.uavs[agent]
            deltas = self.demand_xy[available] - uav.state.position[:2]
            dists = np.linalg.norm(deltas, axis=1)

            reachable_mask = dists <= max_reachable_m
            if reachable_mask.any():
                candidate_pool = np.array(available)[reachable_mask]
                candidate_dists = dists[reachable_mask]
            else:
                candidate_pool = np.array(available)
                candidate_dists = dists


            values = self.demand_weight[candidate_pool]
            score = values / (candidate_dists + 100.0)
            best_i = int(np.argmax(score))
            chosen = int(candidate_pool[best_i])
            available.remove(chosen)
            self.assigned_target[agent] = chosen
