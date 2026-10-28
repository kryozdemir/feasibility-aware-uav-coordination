"""
Koşul 4: PPES + Adil Paylaşım (adjusted winner / envy-free ailesi, round-robin).

Koşul 3'ün (PPES + naif en-yakın atama) üzerine, atama mantığını
run_fair_allocation() ile değiştiriyor -- artık sadece "en yakın" değil,
ajan-özel öncelik profiline (batarya-kritik / değer-kritik) göre adil
bir sıralı seçim yapılıyor.

YAPIŞKAN (sticky) atama: bir ajanın geçerli, hâlâ güvenli hedefi
KORUNUR -- gereksiz yeniden atama (thrashing) önlenir. İSTİSNA: bir
ajanın öncelik profili DEĞER-KRİTİK'ten BATARYA-KRİTİK'e geçerse
(batarya eşiği aşıldığında), mevcut hedef -- hâlâ teknik olarak güvenli
olsa bile -- terk edilip acil/yakın bir hedefe geçilir. Bu, gerçek bir
güvenlik önceliklendirmesi (thrashing değil) -- ajan artık "değerli ama
uzak" bir hedefin peşinde koşmamalı.
"""

import numpy as np

from disaster_uav_env import MAX_EPISODE_STEPS, DT
from ppes_env import PPESManagerUAVEnv
from ppes import prune_candidates
from fair_allocation import run_fair_allocation, BATTERY_CRITICAL_THRESHOLD


class FairAllocationUAVEnv(PPESManagerUAVEnv):

    def __init__(self, n_agents: int = 3, seed: int = None, odd=None, region_labels: np.ndarray = None,
                 coverage_agent_fraction: float = None):
        from fair_allocation import COVERAGE_AGENT_FRACTION
        self.coverage_agent_fraction = coverage_agent_fraction if coverage_agent_fraction is not None else COVERAGE_AGENT_FRACTION
        super().__init__(n_agents=n_agents, seed=seed, odd=odd)


        self.region_labels = region_labels
        if region_labels is not None:
            n_regions = int(region_labels.max()) + 1
            self.region_total_value = np.zeros(n_regions)
            for r in range(n_regions):
                mask = region_labels == r
                self.region_total_value[r] = np.sum(self.demand_weight[mask] * self.demand_confidence[mask])
        else:
            self.region_total_value = None

    def reset(self, seed=None, options=None):
        self._was_battery_critical = {}
        observations, infos = super().reset(seed=seed, options=options)
        self._was_battery_critical = {
            agent: self.uavs[agent].state.battery_pct < BATTERY_CRITICAL_THRESHOLD
            for agent in self.agents
        }
        return observations, infos

    def _run_manager_assignment(self):
        unserved_idx = list(np.where(~self.served)[0])
        for agent in self.agents:
            if self.assigned_target[agent] is not None and self.served[self.assigned_target[agent]]:
                self.assigned_target[agent] = None

        remaining_steps = max(0, MAX_EPISODE_STEPS - self.step_count)
        remaining_time_s = remaining_steps * DT

        for agent in self.agents:
            uav = self.uavs[agent]
            if self._must_return_home(uav):
                self.assigned_target[agent] = None
                continue
            is_battery_critical_now = uav.state.battery_pct < BATTERY_CRITICAL_THRESHOLD
            just_became_critical = is_battery_critical_now and not self._was_battery_critical.get(agent, False)
            self._was_battery_critical[agent] = is_battery_critical_now

            tgt = self.assigned_target[agent]
            if tgt is None:
                continue

            if just_became_critical:


                self.assigned_target[agent] = None
                continue


            still_safe = prune_candidates(
                uav.state.position[:2], uav.state.battery_pct, [tgt],
                self.demand_xy, self.depot_xy, self.uav_config, self.odd,
                remaining_time_s=remaining_time_s
            )
            if not still_safe:
                self.assigned_target[agent] = None

        idle_agents = [a for a in self.agents
                       if self.assigned_target[a] is None and self.uavs[a].state.payload_remaining > 0
                       and not self._must_return_home(self.uavs[a])]
        if not idle_agents:
            return

        already_assigned = set(v for v in self.assigned_target.values() if v is not None)
        available = [i for i in unserved_idx if i not in already_assigned]
        if not available:
            for a in idle_agents:
                self.assigned_target[a] = None
            return

        new_assignments = run_fair_allocation(
            self.uavs, idle_agents, self.demand_xy, self.demand_weight,
            self.demand_confidence, self.depot_xy, available,
            self.uav_config, self.odd, remaining_time_s=remaining_time_s,
            region_labels=self.region_labels, region_total_value=self.region_total_value,
            coverage_agent_fraction=self.coverage_agent_fraction,
            diagnostic_log=getattr(self, "_diag_log", None)
        )
        self.assigned_target.update(new_assignments)
