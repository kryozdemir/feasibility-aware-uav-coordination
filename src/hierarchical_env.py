"""
Koşul 2: Naif Manager + Worker Hiyerarşisi.

Manager (yavaş döngü, MANAGER_INTERVAL adımda bir çalışır):
    Kural-tabanlı, ÖĞRENMEYEN bir atama yapar -- her UAV'a en yakın
    servis edilmemiş (ve başka bir UAV'a henüz atanmamış) talep noktasını
    atar (açgözlü/greedy en-yakın-komşu ataması).

Worker (hızlı döngü, MAPPO ile öğrenen):
    Artık "hangi talebe gideceğim" kararını kendisi vermiyor -- manager'ın
    atadığı TEK hedefe nasıl en verimli şekilde ulaşacağını öğreniyor.
    Bu, Koşul 1'deki (düz MARL) çok-hedefli seçim problemini, çok daha
    kolay bir "hedefe git" problemine indirgiyor.

Gözlem farkı (Koşul 1'e göre):
    Eskiden: K=5 en yakın servis edilmemiş talebin bilgisi
    Şimdi:   SADECE atanmış tek hedefin bilgisi (dx, dy, weight, confidence)
"""

from pathlib import Path
import functools

import numpy as np
from gymnasium import spaces

from disaster_uav_env import DisasterUAVEnv, SERVICE_RADIUS_M, STEP_PENALTY, CRASH_PENALTY, MAX_EPISODE_STEPS, DT, SHAPING_COEF

MANAGER_INTERVAL = 10
DEPOT_RADIUS_M = 200.0


ROTATING_FLEET = True
SWAP_DELAY_STEPS = 15
RETURN_PENALTY = -3.0
BATTERY_LOSS_PENALTY = -10.0


class NaiveManagerUAVEnv(DisasterUAVEnv):
    """DisasterUAVEnv'i miras alır, sadece gözlem ve manager atama mantığını değiştirir."""

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):


        dim = 6 + 5
        return spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):


        self.assigned_target = {}
        observations, infos = super().reset(seed=seed, options=options)
        self.assigned_target = {agent: None for agent in self.agents}
        self.n_reloads = {agent: 0 for agent in self.agents}
        self._swap_timer = {agent: 0 for agent in self.agents}
        self._run_manager_assignment()
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        return observations, infos

    def _run_manager_assignment(self):
        """
        Naif/açgözlü atama -- SADECE görevsiz (None) ajanlara yeni hedef
        atar. Zaten geçerli bir hedefi olan ajanın ataması KORUNUR.

        NOT: Önceki versiyon her manager turunda TÜM atamaları sıfırdan
        yeniden hesaplıyordu -- bu, bir ajanın bir hedefe yarı yolda
        giderken sürekli başka bir hedefe yönlendirilip (özellikle Koşul
        4'te "daha değerli" bir seçenek belirince) önceki ilerlemesinin
        boşa gitmesine yol açan bir "çırpınma" (thrashing) sorunu
        yaratıyordu -- ölçtük: 300 adımda bir ajan 12-15 kez yeniden
        atanıyordu, neredeyse hiç hedefi TAMAMLAMADAN. Şimdi sadece
        görevsiz kalan ajanlar için yeni atama yapılıyor; mevcut geçerli
        atamalar bozulmuyor.

        Yine de tamamen "yapışkan" (sticky) olmak, ilk atamanın ulaşılamaz
        çıkması riskini taşır -- bunu, adayları seçerken zaten kalan bölüm
        süresine göre erişilebilirlik kontrolü yaparak (basit, PPES'siz bir
        mesafe/süre kontrolü) önlüyoruz.
        """
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
        remaining_time_s = remaining_steps * DT
        max_reachable_m = remaining_time_s * self.uav_config.max_speed_mps * 0.8

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

            nearest_i = np.argmin(candidate_dists)
            chosen = int(candidate_pool[nearest_i])
            available.remove(chosen)
            self.assigned_target[agent] = chosen

    def _get_obs(self, agent):
        uav = self.uavs[agent]
        rel_pos = (uav.state.position[:2] - self.depot_xy) / self.pos_scale
        vel = uav.state.velocity[:2] / self.uav_config.max_speed_mps
        battery = np.array([uav.state.battery_pct / 100.0])
        payload = np.array([uav.state.payload_remaining / self.uav_config.payload_capacity])

        target_idx = self.assigned_target.get(agent)
        if target_idx is None:
            target_info = np.zeros(5, dtype=np.float32)
        else:
            delta = self.demand_xy[target_idx] - uav.state.position[:2]
            dist = np.linalg.norm(delta)

            direction = delta / dist if dist > 1e-6 else np.zeros(2)


            dist_norm = np.log1p(dist) / np.log1p(self.pos_scale)
            w = self.demand_weight[target_idx]
            c = self.demand_confidence[target_idx]
            target_info = np.array([direction[0], direction[1], dist_norm, w, c], dtype=np.float32)

        obs = np.concatenate([rel_pos, vel, battery, payload, target_info]).astype(np.float32)
        return obs

    def step(self, actions):
        self.step_count += 1
        rewards = {agent: STEP_PENALTY for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        wind_vector = self._current_wind()

        for agent in self.agents:
            uav = self.uavs[agent]
            if uav.state.crashed or uav.state.battery_depleted:
                terminations[agent] = True
                continue

            prev_pos_xy = uav.state.position[:2].copy()
            target_idx = self.assigned_target.get(agent)
            dist_before = None
            if target_idx is not None:
                dist_before = np.linalg.norm(self.demand_xy[target_idx] - uav.state.position[:2])

            action = np.clip(actions[agent], -1.0, 1.0)
            accel_2d = action * self.uav_config.max_accel_mps2
            accel = np.array([accel_2d[0], accel_2d[1], 0.0])

            uav.step(accel, wind_vector=wind_vector, dt=DT)

            if uav.state.crashed:
                rewards[agent] += CRASH_PENALTY
                terminations[agent] = True
                continue
            if uav.state.battery_depleted:
                rewards[agent] += BATTERY_LOSS_PENALTY
                terminations[agent] = True
                continue


            if uav.state.payload_remaining <= 0:
                if ROTATING_FLEET:


                    self._swap_timer[agent] = self._swap_timer.get(agent, 0) + 1
                    if self._swap_timer[agent] >= SWAP_DELAY_STEPS:
                        uav.state.position[:2] = self.depot_xy
                        uav.state.velocity[:] = 0.0
                        uav.state.payload_remaining = self.uav_config.payload_capacity
                        uav.state.battery_pct = 100.0
                        self._swap_timer[agent] = 0
                        self.n_reloads[agent] = self.n_reloads.get(agent, 0) + 1
                    continue
                d_depot_before = np.linalg.norm(self.depot_xy - prev_pos_xy)
                d_depot_after = np.linalg.norm(self.depot_xy - uav.state.position[:2])
                rewards[agent] += SHAPING_COEF * (d_depot_before - d_depot_after)
                if d_depot_after <= DEPOT_RADIUS_M:
                    uav.state.payload_remaining = self.uav_config.payload_capacity
                    uav.state.battery_pct = 100.0
                    self.n_reloads[agent] = self.n_reloads.get(agent, 0) + 1
                continue

            if target_idx is None:


                uns = np.where(~self.served)[0]
                if len(uns) > 0:
                    dd = np.linalg.norm(self.demand_xy[uns] - uav.state.position[:2], axis=1)
                    target_idx = int(uns[np.argmin(dd)])
                    dist_before = float(dd.min())

            if target_idx is not None and not self.served[target_idx]:
                dist_after = np.linalg.norm(self.demand_xy[target_idx] - uav.state.position[:2])
                rewards[agent] += SHAPING_COEF * (dist_before - dist_after)

                if dist_after <= SERVICE_RADIUS_M:
                    self.served[target_idx] = True


                    rewards[agent] += self.demand_weight[target_idx] * 100.0
                    uav.state.payload_remaining -= 1
                    self.assigned_target[agent] = None


        if self.step_count % MANAGER_INTERVAL == 0:
            self._run_manager_assignment()

        all_served = self.served.all()
        max_steps_reached = self.step_count >= MAX_EPISODE_STEPS
        if all_served or max_steps_reached:


            for agent in self.agents:
                d_home = np.linalg.norm(self.uavs[agent].state.position[:2] - self.depot_xy)
                if d_home > DEPOT_RADIUS_M:
                    rewards[agent] += RETURN_PENALTY * min(1.0, d_home / 10000.0)
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
