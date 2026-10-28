"""
Manager'ın atama mantığı. Öğrenmiyor, tamamen kural tabanlı.

Makaledeki 4.3 bölümü "adjusted winner / envy-free" ve "batarya-kritik,
haberleşme-kritik" profillerinden bahsediyor -- iki noktada netleştirme
gerekiyor:

1. Kullandığım yöntem gerçekte SIRALI SEÇİM (round-robin / picking
   sequence). Adjusted-winner'ın kendisi (Brams & Taylor) 2 taraflı bir
   algoritma, N-ajanlı senaryoya doğrudan uygulanmıyor. Round-robin aynı
   aileden (kıskançlıksızlık hedefleyen) farklı bir yöntem, EF1
   (Envy-Free up to 1 item) garantisi veriyor.

2. İkinci profil "haberleşme-kritik" değil "değer-kritik" -- sistemde
   haberleşme/menzil modellemesi (paket kaybı, röle ağı vs.) hiç yok,
   gerçek bir haberleşme-tabanlı önceliklendirme kuramadım. Onun yerine
   batarya yeterliyken değer/mesafe oranı en yüksek hedefi seçen bir
   profil koydum.

Üç profil (her manager turunda ajanın o anki durumuna göre belirlenir):
- batarya-kritik (batarya < %50): en yakın güvenli hedef, enerji tasarrufu
- değer-kritik (batarya >= %50): değer/mesafe oranı en yüksek hedef
- kapsama-kritik (turda en fazla 1 ajan): az kapsanmış bölgelere zorla gönder

Sıra: bataryası en düşük ajan önce seçiyor (en acil ihtiyaç önce).
"""

import numpy as np

from ppes import ODDParams, prune_candidates

BATTERY_CRITICAL_THRESHOLD = 50.0
COVERAGE_AGENT_FRACTION = 0.2


def run_fair_allocation(uavs: dict, agent_order_pool: list, demand_xy: np.ndarray,
                         demand_weight: np.ndarray, demand_confidence: np.ndarray,
                         depot_xy: np.ndarray, unserved_idx: list, uav_config, odd: ODDParams,
                         remaining_time_s: float = None, region_labels: np.ndarray = None,
                         region_total_value: np.ndarray = None,
                         coverage_agent_fraction: float = COVERAGE_AGENT_FRACTION,
                         diagnostic_log: list = None) -> dict:
    """
    uavs: {agent_name: KinematicUAV}
    agent_order_pool: atama yapılacak ajan isimleri listesi
    unserved_idx: henüz servis edilmemiş talep indeksleri
    remaining_time_s: bölümün kalan süresi (saniye)
    region_labels: her talep noktasının hangi coğrafi bölgeye ait olduğu (sabit)
    region_total_value: her bölgenin TOPLAM (ağırlık x güven) değeri (sabit)

    Dönüş: {agent_name: target_idx veya None}

    YENİ -- KAPSAMA-KRİTİK ÖNCELİK: Normal batarya-kritik/değer-kritik
    seçimden ÖNCE, eğer herhangi bir bölgenin değer-bazlı kapsaması
    COVERAGE_THRESHOLD'un altındaysa, o bölgeye en yakın/en uygun TEK bir
    ajan ZORUNLU olarak yönlendirilir (değeri ne olursa olsun). Bu, saf
    değer-yoğunluğu mantığının düşük-confidence (potansiyel) bölgeleri
    hiçbir zaman seçmediği bulgusuna karşı bir DÜZELTME -- manager'a
    kasıtlı, öğrenmeyen bir "herkese ulaş" tabanı ekliyor. Kalan ajanlar
    hâlâ normal (batarya-kritik/değer-kritik) mantıkla atanıyor, yani
    confirmed bölgeler hâlâ ağırlıklı payı alıyor -- sadece potansiyel
    bölgeler artık asla sıfıra düşmüyor.
    """
    ordered_agents = sorted(agent_order_pool, key=lambda a: uavs[a].state.battery_pct)
    available = list(unserved_idx)
    assignment = {}


    if region_labels is not None and region_total_value is not None and len(available) > 0:
        avail_arr = np.array(available)
        n_regions = len(region_total_value)
        region_remaining_value = np.zeros(n_regions)
        for idx in avail_arr:
            r = region_labels[idx]
            region_remaining_value[r] += demand_weight[idx] * demand_confidence[idx]
        region_coverage = 1.0 - np.divide(
            region_remaining_value, region_total_value,
            out=np.ones(n_regions), where=region_total_value > 0
        )


        dominant_region = int(np.argmax(region_total_value))
        eligible_regions = [r for r in range(n_regions) if r != dominant_region]


        if region_coverage[eligible_regions].min() >= 0.95:
            under_covered_regions = []
        else:
            under_covered_regions = [r for r in eligible_regions if region_coverage[r] <= np.median(region_coverage[eligible_regions])]

        if under_covered_regions:
            region_candidates = avail_arr[np.isin(region_labels[avail_arr], under_covered_regions)]
            if len(region_candidates) > 0:


                if diagnostic_log is not None:
                    any_agent = ordered_agents[0]
                    any_uav = uavs[any_agent]
                    feasible_pool = prune_candidates(
                        any_uav.state.position[:2], any_uav.state.battery_pct, region_candidates.tolist(),
                        demand_xy, depot_xy, uav_config, odd, remaining_time_s=remaining_time_s
                    )
                    diagnostic_log.append({
                        "raw_pool": len(region_candidates),
                        "feasible_pool": len(feasible_pool),
                        "quota": max(1, int(round(len(agent_order_pool) * coverage_agent_fraction))),
                    })


                n_coverage_agents = max(1, int(round(len(agent_order_pool) * coverage_agent_fraction)))
                remaining_candidates = region_candidates.tolist()

                for _ in range(n_coverage_agents):
                    if not remaining_candidates:
                        break
                    best_agent, best_dist, best_target = None, None, None
                    for agent in ordered_agents:
                        if agent in assignment:
                            continue
                        uav = uavs[agent]
                        safe = prune_candidates(
                            uav.state.position[:2], uav.state.battery_pct, remaining_candidates,
                            demand_xy, depot_xy, uav_config, odd, remaining_time_s=remaining_time_s
                        )
                        if not safe:
                            continue
                        d = np.linalg.norm(demand_xy[safe] - uav.state.position[:2], axis=1)
                        i = int(np.argmin(d))
                        if best_dist is None or d[i] < best_dist:
                            best_dist, best_agent, best_target = d[i], agent, safe[i]

                    if best_agent is None:
                        break
                    assignment[best_agent] = best_target
                    available.remove(best_target)
                    remaining_candidates.remove(best_target)


    for agent in ordered_agents:
        if agent in assignment:
            continue

        uav = uavs[agent]

        safe_candidates = prune_candidates(
            uav.state.position[:2], uav.state.battery_pct, available,
            demand_xy, depot_xy, uav_config, odd, remaining_time_s=remaining_time_s
        )

        if not safe_candidates:
            assignment[agent] = None
            continue

        is_battery_critical = uav.state.battery_pct < BATTERY_CRITICAL_THRESHOLD

        if is_battery_critical:
            to_c = np.linalg.norm(demand_xy[safe_candidates] - uav.state.position[:2], axis=1)
            from_c = np.linalg.norm(demand_xy[safe_candidates] - depot_xy, axis=1)
            score = -(to_c + from_c)
        else:


            to_c = np.linalg.norm(demand_xy[safe_candidates] - uav.state.position[:2], axis=1)
            value = demand_weight[safe_candidates]
            score = value / (to_c + 100.0)

        best_i = int(np.argmax(score))
        chosen = safe_candidates[best_i]
        available.remove(chosen)
        assignment[agent] = chosen

    return assignment
