
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

RETURN_MARGIN = 0.10


class EntityType(Enum):
    """Ontoloji: sistemdeki varlık tipleri."""
    UAV = auto()
    DEMAND_POINT = auto()
    DEPOT = auto()


@dataclass
class ODDParams:


    wind_speed_mps: float = 3.0
    wind_dir_rad: float = 0.6


    gps_reliable_radius_m: float = 8000.0
    gps_degraded_margin: float = 0.15


    no_fly_zones: tuple = ()
    """Operational Design Domain -- fiziksel/operasyonel güvenlik sınırları."""


    min_battery_reserve_pct: float = 15.0


    cruise_speed_mps: float = 14.0


def compute_safe_max_range_m(battery_pct: float, uav_config, odd: ODDParams,
                              remaining_time_s: float = None) -> float:
    """
    SES: Mevcut batarya durumuna göre UAV'ın GİDİP-DÖNEBİLECEĞİ (round-trip)
    maksimum tek-yön mesafeyi hesaplar.

    Mantık: kullanılabilir batarya = mevcut batarya - güvenlik rezervi.
    Bu enerjiyle, seyir hızında ne kadar süre uçulabilir hesaplanır,
    süre * hız = kat edilebilecek TOPLAM mesafe. Round-trip olduğu için
    (gidip dönmesi gerektiği için) tek-yön güvenli mesafe bunun YARISI.

    remaining_time_s verilirse (bölümün/görevin kalan süresi), menzil
    ayrıca bu süre içinde fiziksel olarak kat edilebilecek mesafeyle de
    sınırlanır -- ODD'nin batarya limiti kadar gerçek bir kısıt: bir UAV
    bataryası yetse bile, GÖREV PENCERESİ (mission window) dolmadan önce
    oraya varamayacaksa, o hedef de güvenli/ulaşılabilir sayılmamalı.
    Bu olmadan, batarya-bazlı menzil (örn. tam şarjda ~29km) bölüm
    uzunluğumuzun fiziksel olarak kat edilebilir mesafesini (örn. ~9km)
    fersah fersah aşabiliyor ve "değer-kritik" profil, hiçbir zaman
    ulaşılamayacak hedefler seçebiliyordu.
    """
    usable_battery_pct = max(0.0, battery_pct - odd.min_battery_reserve_pct)
    usable_energy_wh = usable_battery_pct / 100.0 * uav_config.battery_capacity_wh

    power_w = uav_config.power_base_w + uav_config.power_drag_coef * odd.cruise_speed_mps ** 2
    flight_time_s = usable_energy_wh * 3600.0 / power_w if power_w > 0 else 0.0

    total_distance_m = flight_time_s * odd.cruise_speed_mps
    battery_based_one_way_m = total_distance_m / 2.0

    if remaining_time_s is None:
        return battery_based_one_way_m


    usable_time_s = remaining_time_s * (1.0 - RETURN_MARGIN)


    effective_v = max(1.0, odd.cruise_speed_mps - odd.wind_speed_mps * 0.5)
    time_based_one_way_m = (usable_time_s * effective_v) / 2.0
    return min(battery_based_one_way_m, time_based_one_way_m)


def prune_candidates(uav_position_xy: np.ndarray, battery_pct: float, candidate_indices: list,
                      demand_xy: np.ndarray, depot_xy: np.ndarray, uav_config, odd: ODDParams,
                      remaining_time_s: float = None) -> list:
    """
    Behavior Oracle -> PPES: adaylardan, UAV'ın MEVCUT konumundan gidip
    depoya güvenle dönebileceği (hem batarya rezervi hem de kalan görev
    süresi ihlal edilmeden) olanları döndürür. Bu mesafeyi aşan adaylar
    tamamen elenir -- manager bunları görmez bile.
    """
    if not candidate_indices:
        return []

    safe_range_m = compute_safe_max_range_m(battery_pct, uav_config, odd, remaining_time_s)

    candidate_indices = np.array(candidate_indices)
    to_candidate = np.linalg.norm(demand_xy[candidate_indices] - uav_position_xy, axis=1)
    from_candidate_to_depot = np.linalg.norm(demand_xy[candidate_indices] - depot_xy, axis=1)
    round_trip_dist = to_candidate + from_candidate_to_depot


    safe_mask = round_trip_dist <= 2 * safe_range_m


    dist_from_depot = np.linalg.norm(demand_xy[candidate_indices] - depot_xy, axis=1)
    degraded = dist_from_depot > odd.gps_reliable_radius_m
    if degraded.any():
        stricter = round_trip_dist <= 2 * safe_range_m * (1.0 - odd.gps_degraded_margin)
        safe_mask = np.where(degraded, stricter, safe_mask)


    for (zx, zy, zr) in odd.no_fly_zones:
        d_zone = np.linalg.norm(demand_xy[candidate_indices] - np.array([zx, zy]), axis=1)
        safe_mask &= (d_zone > zr)

    return candidate_indices[safe_mask].tolist()
