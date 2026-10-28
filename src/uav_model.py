"""
Kinematik nokta-kütle UAV modeli (fizik motoru yok).

Makaledeki 4.4 bölümüne karşılık gelir:
"Kinematik/nokta-kütle bir UAV modeli (hız, rüzgar, batarya, GPS gürültüsü) kurulacak."

Bu modül tek başına (ortamdan bağımsız) test edilebilir -- bir UAV nesnesi
oluşturup, aksiyon uygulayıp, durumun beklendiği gibi evrildiğini görebiliriz.

Durum (state):
    - position: (x, y, z) metre, UTM projeksiyonunda
    - velocity: (vx, vy, vz) m/s
    - battery: 0-100 arası yüzde
    - time: geçen simülasyon süresi (s)

Aksiyon (action):
    - acceleration: (ax, ay, az) m/s^2 -- UAV'ın uygulamak istediği ivme

Dinamik:
    gerçek_hız = uav_hızı + rüzgar_vektörü
    yeni_konum = konum + gerçek_hız * dt
    batarya_tüketimi = f(hız, rüzgar'a karşı harcanan güç, dt)

Gözlem (observation, GPS gürültülü):
    gözlemlenen_konum = gerçek_konum + N(0, gps_noise_std)
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class UAVConfig:
    """UAV'ın fiziksel/operasyonel parametreleri (ODD ile bağlantılı olacak)."""

    max_speed_mps: float = 15.0
    max_accel_mps2: float = 3.0
    max_altitude_m: float = 120.0
    min_altitude_m: float = 20.0


    battery_capacity_wh: float = 200.0

    power_base_w: float = 80.0
    power_drag_coef: float = 0.5

    gps_noise_std_m: float = 2.5


    payload_capacity: int = 10

    max_wind_speed_mps: float = 10.0


@dataclass
class UAVState:
    position: np.ndarray
    velocity: np.ndarray
    battery_pct: float = 100.0
    time_s: float = 0.0
    crashed: bool = False
    battery_depleted: bool = False
    payload_remaining: int = 10

    def copy(self) -> "UAVState":
        return UAVState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            battery_pct=self.battery_pct,
            time_s=self.time_s,
            crashed=self.crashed,
            battery_depleted=self.battery_depleted,
            payload_remaining=self.payload_remaining,
        )


class KinematicUAV:
    """Tek bir UAV'ın kinematik modeli. Ortamdan (PettingZoo) bağımsız."""

    def __init__(self, config: UAVConfig, initial_position: np.ndarray, rng: np.random.Generator = None):
        self.config = config
        self.rng = rng or np.random.default_rng()
        self.state = UAVState(
            position=np.array(initial_position, dtype=float),
            velocity=np.zeros(3, dtype=float),
            payload_remaining=config.payload_capacity,
        )

    def reset(self, initial_position: np.ndarray):
        self.state = UAVState(
            position=np.array(initial_position, dtype=float),
            velocity=np.zeros(3, dtype=float),
            payload_remaining=self.config.payload_capacity,
        )

    def step(self, acceleration: np.ndarray, wind_vector: np.ndarray, dt: float = 1.0) -> UAVState:
        """
        acceleration: (3,) UAV'ın uygulamak istediği ivme komutu (m/s^2)
        wind_vector: (3,) rüzgarın o anki hız vektörü (m/s) -- dışarıdan verilir
        dt: zaman adımı (saniye)
        """
        if self.state.crashed or self.state.battery_depleted:
            return self.state

        acceleration = np.array(acceleration, dtype=float)
        wind_vector = np.array(wind_vector, dtype=float)


        accel_norm = np.linalg.norm(acceleration)
        if accel_norm > self.config.max_accel_mps2:
            acceleration = acceleration / accel_norm * self.config.max_accel_mps2


        new_airspeed = self.state.velocity + acceleration * dt
        speed_norm = np.linalg.norm(new_airspeed)
        if speed_norm > self.config.max_speed_mps:
            new_airspeed = new_airspeed / speed_norm * self.config.max_speed_mps
        self.state.velocity = new_airspeed


        ground_velocity = self.state.velocity + wind_vector


        new_position = self.state.position + ground_velocity * dt


        if new_position[2] < self.config.min_altitude_m or new_position[2] > self.config.max_altitude_m:
            self.state.crashed = True

        self.state.position = new_position


        airspeed_norm = np.linalg.norm(self.state.velocity)
        power_w = self.config.power_base_w + self.config.power_drag_coef * airspeed_norm ** 2
        energy_used_wh = power_w * dt / 3600.0
        battery_pct_used = 100.0 * energy_used_wh / self.config.battery_capacity_wh
        self.state.battery_pct = max(0.0, self.state.battery_pct - battery_pct_used)
        if self.state.battery_pct <= 0.0:
            self.state.battery_depleted = True

        self.state.time_s += dt
        return self.state

    def observe_gps(self) -> np.ndarray:
        """Gürültülü GPS gözlemi döndürür (gerçek konum değil)."""
        noise = self.rng.normal(0.0, self.config.gps_noise_std_m, size=3)
        return self.state.position + noise
