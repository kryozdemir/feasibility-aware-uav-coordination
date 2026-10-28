"""
UAV modelini ortamdan bağımsız test etme.

Senaryo: UAV, (0,0,50) konumundan başlayıp (2000, 1500, 50) hedefine doğru
uçmaya çalışıyor. Basit bir "hedefe git" kontrolcüsü (proportional control)
kullanıyoruz -- bu RL değil, sadece modelin fiziksel olarak doğru davrandığını
görmek için.

Kontrol edilecekler:
1. UAV rüzgarsız ortamda hedefe ulaşabiliyor mu?
2. Rüzgar varken yörünge nasıl sapıyor?
3. Batarya tüketimi mantıklı bir hızda mı azalıyor?
4. GPS gözlemi gerçek konumdan ne kadar sapıyor?
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from uav_model import KinematicUAV, UAVConfig


def simple_goto_controller(state, target, max_accel, wind_vector):
    """
    Orantısal kontrolcü -- rüzgar üçgeni mantığıyla düzeltilmiş.

    Önceki (hatalı) versiyon: istenen YER hızını doğrudan airspeed hatası
    olarak kullanıyordu, yani rüzgarı hiç hesaba katmıyordu. Bu da UAV'ın
    rüzgar olsun olmasın hemen hemen aynı airspeed'e yakınsamasına ve
    dolayısıyla batarya tüketim ORANININ rüzgardan etkilenmemesine yol
    açıyordu -- fiziksel olarak yanlış (gerçek bir UAV, rüzgara karşı aynı
    yer-hızını korumak için daha yüksek airspeed üretmeli, bu da anlık güç
    tüketimini artırmalı).

    Düzeltme: istenen yer-hızını hesapla, sonra rüzgarı çıkararak UAV'ın
    ÜRETMESİ gereken airspeed'i bul (wind triangle), hatayı buna göre hesapla.
    """
    direction = target - state.position
    dist = np.linalg.norm(direction)
    if dist < 1e-3:
        return np.zeros(3)
    desired_speed = min(15.0, dist * 0.5)
    desired_ground_velocity = direction / dist * desired_speed

    desired_airspeed = desired_ground_velocity - wind_vector
    accel = (desired_airspeed - state.velocity) * 0.5
    return accel


def run_scenario(wind_vector, label, rng_seed=42):
    config = UAVConfig()
    rng = np.random.default_rng(rng_seed)
    uav = KinematicUAV(config, initial_position=[0, 0, 50], rng=rng)
    target = np.array([2000.0, 1500.0, 50.0])

    dt = 1.0
    max_steps = 400

    positions = [uav.state.position.copy()]
    gps_positions = [uav.observe_gps()]
    battery_history = [uav.state.battery_pct]
    times = [0.0]

    for step in range(max_steps):
        if uav.state.crashed or uav.state.battery_depleted:
            print(f"  [{label}] Durdu: crashed={uav.state.crashed}, battery_depleted={uav.state.battery_depleted} (adım {step})")
            break
        accel = simple_goto_controller(uav.state, target, config.max_accel_mps2, wind_vector)
        uav.step(accel, wind_vector=wind_vector, dt=dt)

        positions.append(uav.state.position.copy())
        gps_positions.append(uav.observe_gps())
        battery_history.append(uav.state.battery_pct)
        times.append(uav.state.time_s)

        dist_to_target = np.linalg.norm(uav.state.position - target)
        if dist_to_target < 20.0:
            print(f"  [{label}] Hedefe ulaşıldı! Süre: {uav.state.time_s:.0f}s, kalan batarya: %{uav.state.battery_pct:.1f}")
            break
    else:
        print(f"  [{label}] {max_steps} adımda hedefe ulaşılamadı. Son mesafe: {dist_to_target:.0f}m")

    return {
        "positions": np.array(positions),
        "gps_positions": np.array(gps_positions),
        "battery": np.array(battery_history),
        "times": np.array(times),
        "target": target,
    }


def main():
    print("Senaryo 1: Rüzgarsız")
    no_wind = run_scenario(wind_vector=np.array([0.0, 0.0, 0.0]), label="rüzgarsız")

    print("\nSenaryo 2: Sabit yan rüzgar (5 m/s, doğu->batı)")
    with_wind = run_scenario(wind_vector=np.array([-5.0, 0.0, 0.0]), label="rüzgarlı")


    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    ax.plot(no_wind["positions"][:, 0], no_wind["positions"][:, 1], label="Rüzgarsız (gerçek yörünge)", color="blue")
    ax.plot(with_wind["positions"][:, 0], with_wind["positions"][:, 1], label="Rüzgarlı (gerçek yörünge)", color="red")
    ax.scatter(*no_wind["target"][:2], color="green", marker="*", s=200, label="Hedef", zorder=5)
    ax.scatter(0, 0, color="black", marker="o", s=80, label="Başlangıç", zorder=5)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("UAV Yörüngesi: Rüzgar Etkisi")
    ax.legend(); ax.axis("equal")

    ax = axes[1]
    ax.plot(no_wind["times"], no_wind["battery"], label="Rüzgarsız", color="blue", linewidth=2)
    ax.plot(with_wind["times"], with_wind["battery"], label="Rüzgarlı (karşı rüzgar bileşeni)", color="red", linewidth=2)
    ax.set_xlabel("Zaman (s)"); ax.set_ylabel("Batarya (%)")
    ax.set_title("Batarya Tüketimi (düzeltilmiş kontrolcü)")
    ax.legend()

    ax = axes[2]

    n = min(50, len(no_wind["positions"]))
    ax.plot(no_wind["positions"][:n, 0], no_wind["positions"][:n, 1], color="blue", label="Gerçek konum", linewidth=2)
    ax.scatter(no_wind["gps_positions"][:n, 0], no_wind["gps_positions"][:n, 1], color="orange", s=10, alpha=0.6, label="GPS gözlemi (gürültülü)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("GPS Gürültüsü (ilk 50 adım)")
    ax.legend(); ax.axis("equal")

    plt.tight_layout()
    out_path = Path(__file__).parent.parent / "data" / "processed" / "uav_model_test.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nGörsel kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
