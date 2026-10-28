"""
K2c: Naif manager + kapsama-kritik kural.

Kritik ablasyon testi. K4'un kapsama basarisi gercekten PPES + adil
paylasimdan mi geliyor, yoksa sadece "her turda filonun %20'sini az
kapsanan bolgeye zorla gonder" kuralindan mi?

Bu kosul o kurali naif manager'a ekliyor, ama:
- PPES budamasi YOK (menzil/batarya guvenlik filtresi yok)
- adil paylasim YOK (batarya-kritik/deger-kritik profilleri yok)
- geri kalan her sey K4 ile ayni (rotasyonlu filo, 60dk, 200Wh,
  kapasite 10, depo yaricapi, hedefsiz kalmama) -- bunlari zaten
  NaiveManagerUAVEnv'den miras aliyor

K2c ~ K4 cikarsa: katki kapsama kuralindadir, PPES/adil paylasim
degildir. K2c << K4 cikarsa: PPES + adil paylasim gercekten gerekli.
"""

import numpy as np

from hierarchical_env import NaiveManagerUAVEnv
from disaster_uav_env import MAX_EPISODE_STEPS, DT
from fair_allocation import COVERAGE_AGENT_FRACTION


class NaiveCoverageUAVEnv(NaiveManagerUAVEnv):

    def __init__(self, n_agents: int = 3, seed: int = None, region_labels: np.ndarray = None,
                 demand_file: str = "demand_points_v2.geojson", coverage_agent_fraction: float = COVERAGE_AGENT_FRACTION):
        assert region_labels is not None
        self.region_labels = region_labels
        self.n_regions = int(region_labels.max()) + 1
        self.coverage_agent_fraction = coverage_agent_fraction
        super().__init__(n_agents=n_agents, seed=seed, demand_file=demand_file)
        self.region_total_value = np.zeros(self.n_regions)
        for r in range(self.n_regions):
            m = region_labels == r
            self.region_total_value[r] = np.sum(self.demand_weight[m] * self.demand_confidence[m])
        self.dominant_region = int(np.argmax(self.region_total_value))

    def _run_manager_assignment(self):
        unserved_idx = list(np.where(~self.served)[0])
        for agent in self.agents:
            if self.assigned_target[agent] is not None and self.served[self.assigned_target[agent]]:
                self.assigned_target[agent] = None

        idle_agents = [a for a in self.agents
                       if self.assigned_target[a] is None and self.uavs[a].state.payload_remaining > 0]
        if not idle_agents:
            return

        already = set(v for v in self.assigned_target.values() if v is not None)
        available = [i for i in unserved_idx if i not in already]
        if not available:
            return

        remaining_steps = max(0, MAX_EPISODE_STEPS - self.step_count)
        max_reachable_m = remaining_steps * DT * self.uav_config.max_speed_mps * 0.8


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
            if getattr(self, "_diag_log", None) is not None and cands:
                any_agent = idle_agents[0] if idle_agents else self.agents[0]
                p = self.uavs[any_agent].state.position[:2]
                d = np.linalg.norm(self.demand_xy[cands] - p, axis=1)
                feas = int((d <= max_reachable_m).sum())
                self._diag_log.append({"raw_pool": len(cands), "feasible_pool": feas, "quota": n_cov})
            for _ in range(n_cov):
                if not cands:
                    break
                best_a, best_d, best_t = None, None, None
                for agent in idle_agents:
                    if agent in assigned_here:
                        continue
                    p = self.uavs[agent].state.position[:2]
                    d = np.linalg.norm(self.demand_xy[cands] - p, axis=1)
                    i = int(np.argmin(d))

                    if d[i] > max_reachable_m:
                        continue
                    if best_d is None or d[i] < best_d:
                        best_d, best_a, best_t = d[i], agent, cands[i]
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
                break
            p = self.uavs[agent].state.position[:2]
            d = np.linalg.norm(self.demand_xy[available] - p, axis=1)
            mask = d <= max_reachable_m
            pool = np.array(available)[mask] if mask.any() else np.array(available)
            dists = d[mask] if mask.any() else d
            chosen = int(pool[int(np.argmin(dists))])
            available.remove(chosen)
            self.assigned_target[agent] = chosen
