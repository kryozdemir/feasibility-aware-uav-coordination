"""
K2vcp: Ablasyon -- K2vc (value-density + kapsama kurali) + PPES eklenmis
hali. Fair allocation / battery-critical profil DEGISMEDI (hala yok).

Amac: K2vc -> K2vcp gecisi PPES'in SAF etkisini gosterir (baska hicbir
sey degismiyor -- ayni value-density formulu, ayni kapsama kurali,
sadece PPES filtresi eklendi). K2vcp -> K4 gecisi ise geriye kalan
fair/profile etkisini izole eder.

Tam faktoriyel zincir:
K2   : value-density YOK, kapsama YOK, PPES YOK, fair/profil YOK
K2v  : value-density VAR, kapsama YOK, PPES YOK, fair/profil YOK
K2vc : value-density VAR, kapsama VAR, PPES YOK, fair/profil YOK
K2vcp: value-density VAR, kapsama VAR, PPES VAR, fair/profil YOK
K4   : value-density VAR, kapsama VAR, PPES VAR, fair/profil VAR
"""

import numpy as np

from value_density_coverage_env import ValueDensityCoverageUAVEnv
from disaster_uav_env import MAX_EPISODE_STEPS, DT
from ppes import prune_candidates, ODDParams


class ValueDensityCoveragePPESUAVEnv(ValueDensityCoverageUAVEnv):

    def __init__(self, n_agents: int = 3, seed: int = None, region_labels: np.ndarray = None,
                 coverage_agent_fraction=None, odd: ODDParams = None):
        from fair_allocation import COVERAGE_AGENT_FRACTION
        self.odd = odd if odd is not None else ODDParams()
        cf = coverage_agent_fraction if coverage_agent_fraction is not None else COVERAGE_AGENT_FRACTION
        super().__init__(n_agents=n_agents, seed=seed, region_labels=region_labels,
                          coverage_agent_fraction=cf)

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
        if not available:
            for a in idle_agents:
                self.assigned_target[a] = None
            return

        remaining_steps = max(0, MAX_EPISODE_STEPS - self.step_count)
        remaining_time_s = remaining_steps * DT

        avail_arr = np.array(available)
        rem = np.zeros(self.n_regions)
        for idx in avail_arr:
            rem[self.region_labels[idx]] += self.demand_weight[idx] * self.demand_confidence[idx]
        cov = 1.0 - np.divide(rem, self.region_total_value,
                              out=np.ones(self.n_regions), where=self.region_total_value > 0)
        eligible = [r for r in range(self.n_regions) if r != self.dominant_region]
        assigned_here = set()


        if cov[eligible].min() < 0.95:
            under = [r for r in eligible if cov[r] <= np.median(cov[eligible])]
            cands = avail_arr[np.isin(self.region_labels[avail_arr], under)].tolist()
            n_cov = max(1, int(round(len(self.agents) * self.coverage_agent_fraction)))
            for _ in range(n_cov):
                if not cands:
                    break
                best_a, best_score, best_t = None, None, None
                for agent in idle_agents:
                    if agent in assigned_here:
                        continue
                    uav = self.uavs[agent]

                    safe_cands = prune_candidates(
                        uav.state.position[:2], uav.state.battery_pct, cands,
                        self.demand_xy, self.depot_xy, self.uav_config, self.odd,
                        remaining_time_s=remaining_time_s
                    )
                    if not safe_cands:
                        continue
                    p = uav.state.position[:2]
                    d = np.linalg.norm(self.demand_xy[safe_cands] - p, axis=1)
                    values = self.demand_weight[np.array(safe_cands)]
                    score = values / (d + 100.0)
                    i = int(np.argmax(score))
                    if best_score is None or score[i] > best_score:
                        best_score, best_a, best_t = score[i], agent, safe_cands[i]
                if best_a is None:
                    break
                self.assigned_target[best_a] = best_t
                assigned_here.add(best_a)
                available.remove(best_t)
                cands.remove(best_t)


        for agent in idle_agents:
            if agent in assigned_here:
                continue
            if not available:
                self.assigned_target[agent] = None
                continue
            uav = self.uavs[agent]
            safe_avail = prune_candidates(
                uav.state.position[:2], uav.state.battery_pct, available,
                self.demand_xy, self.depot_xy, self.uav_config, self.odd,
                remaining_time_s=remaining_time_s
            )
            pool = safe_avail if safe_avail else available
            deltas = self.demand_xy[pool] - uav.state.position[:2]
            dists = np.linalg.norm(deltas, axis=1)
            values = self.demand_weight[pool]
            score = values / (dists + 100.0)
            best_i = int(np.argmax(score))
            chosen = int(np.array(pool)[best_i])
            available.remove(chosen)
            self.assigned_target[agent] = chosen
