"""
Koşul 3: PPES + Naif Manager.

Koşul 2'nin (naif manager) üzerine PPES budama katmanı ekleniyor.
Manager artık TÜM servis edilmemiş talepler arasından değil, PPES'in
"güvenli" (batarya ile gidip-dönülebilir) olarak işaretlediği alt küme
içinden en yakını seçiyor.

Fark: sadece _run_manager_assignment metodu değişiyor -- gözlem,
ödül, fizik hep aynı (NaiveManagerUAVEnv'den miras).
"""

import numpy as np

from hierarchical_env import NaiveManagerUAVEnv
from disaster_uav_env import MAX_EPISODE_STEPS, DT
from ppes import ODDParams, prune_candidates, RETURN_MARGIN


class PPESManagerUAVEnv(NaiveManagerUAVEnv):

    def __init__(self, n_agents: int = 3, seed: int = None, odd: ODDParams = None):
        self.odd = odd or ODDParams()
        super().__init__(n_agents=n_agents, seed=seed)

    def _must_return_home(self, uav) -> bool:
        """Return-to-home kontrolu: kalan batarya, depoya donmek icin
        gerekenden az mi? PPES'in eksik parcasi buydu -- sadece atama
        aninda menzil kontrol ediliyordu, ucus sirasinda kimse "artik
        don" demiyordu, UAV'lar sahada batarya bitirip kayboluyordu.
        Gercek dronelarda standart bir emniyet ozelligi."""
        d_home = np.linalg.norm(uav.state.position[:2] - self.depot_xy)
        power = self.uav_config.power_base_w + self.uav_config.power_drag_coef * self.odd.cruise_speed_mps ** 2

        eff_v = max(1.0, self.odd.cruise_speed_mps - self.odd.wind_speed_mps)
        energy_needed_wh = power * (d_home / eff_v) / 3600.0
        pct_needed = 100.0 * energy_needed_wh / self.uav_config.battery_capacity_wh
        return uav.state.battery_pct <= pct_needed + self.odd.min_battery_reserve_pct

    def _run_manager_assignment(self):
        """
        PPES-budamalı atama -- SADECE görevsiz ajanlara veya mevcut hedefi
        ARTIK GÜVENSİZ hale gelmiş (batarya düştüğü için round-trip
        yapılamayacak) ajanlara yeni atama yapar. Geçerli ve hâlâ güvenli
        bir atama KORUNUR -- gereksiz yeniden atama (thrashing) önlenir.

        NOT: Önceki versiyon her turda TÜM ajanları sıfırdan yeniden
        atıyordu -- bu, bir ajanın bir hedefe yarı yolda giderken sürekli
        başka bir hedefe kaydırılmasına (thrashing) yol açıyordu (300
        adımda 12-15 kez yeniden atama, neredeyse hiç tamamlama olmadan).
        """
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
            tgt = self.assigned_target[agent]
            if tgt is None:
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

        for agent in idle_agents:
            uav = self.uavs[agent]
            safe_candidates = prune_candidates(
                uav.state.position[:2], uav.state.battery_pct, available,
                self.demand_xy, self.depot_xy, self.uav_config, self.odd,
                remaining_time_s=remaining_time_s
            )
            if not safe_candidates:
                self.assigned_target[agent] = None
                continue
            deltas = self.demand_xy[safe_candidates] - uav.state.position[:2]
            dists = np.linalg.norm(deltas, axis=1)
            nearest_i = np.argmin(dists)
            chosen = safe_candidates[nearest_i]
            available.remove(chosen)
            self.assigned_target[agent] = chosen
