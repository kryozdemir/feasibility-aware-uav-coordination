"""
Koşul 1-4 için kapsamlı değerlendirme: eğitilmiş (deterministik) politikaları
birden fazla bölümde çalıştırıp makaledeki metriklerin bir alt kümesini
hesaplar.

Hesaplanan metrikler:
- Toplam teslim edilen değer (weight x confidence toplamı)
- Teslimat başarı oranı (servis edilen / toplam talep, hem sayı hem değer bazlı)
- Ortalama bölüm başına servis sayısı
- Kat edilen toplam mesafe (ajan başına ortalama, metre)
- Enerji tüketimi (ajan başına ortalama batarya kullanımı, %)
- Çarpışma/batarya-tükenme oranı
- Ajanlar arası adalet endeksi (Jain's index) -- her ajanın ürettiği
  değerin ne kadar eşit dağıldığı
- UAV kullanım yüzdesi (görevli/boşta geçen adım oranı)

NOT: Bu bir "smoke test" değerlendirmesi -- N_EVAL_EPISODES küçük tutuldu
(hesaplama süresi nedeniyle). Makaledeki asıl deneyler için 5+ seed ve
çok daha fazla eval bölümü gerekecek.
"""

import pickle
from pathlib import Path

import numpy as np
import torch

from disaster_uav_env import DisasterUAVEnv
from hierarchical_env import NaiveManagerUAVEnv
from ppes_env import PPESManagerUAVEnv
from fair_allocation_env import FairAllocationUAVEnv
from flexible_fair_env import FlexibleFairUAVEnv
from mappo_networks import Actor

PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
N_EVAL_EPISODES = 15

CONDITIONS = {
    "K1_Duz_MARL": (DisasterUAVEnv, "actor_condition1_long.pt"),
    "K2_Naif_Manager": (NaiveManagerUAVEnv, "actor_condition2_sticky.pt"),
    "K3_PPES": (PPESManagerUAVEnv, "actor_condition3_sticky.pt"),
    "K4_Adil_Paylasim": (FairAllocationUAVEnv, "actor_condition4_density.pt"),
    "K4b_Esnek_Adil": (FlexibleFairUAVEnv, "actor_condition4b_flexible_v2.pt"),
}


def jain_index(values):
    values = np.array(values, dtype=float)
    if values.sum() == 0:
        return 1.0
    n = len(values)
    return (values.sum() ** 2) / (n * np.sum(values ** 2))


def evaluate_condition(env_class, actor_path, n_agents=3, n_episodes=N_EVAL_EPISODES):
    env = env_class(n_agents=n_agents, seed=1000)
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    action_dim = env.action_space(env.possible_agents[0]).shape[0]

    actor = Actor(obs_dim, action_dim)
    actor.load_state_dict(torch.load(PROC_DIR / actor_path, weights_only=True))
    actor.eval()

    results = {
        "n_served": [], "total_value": [], "distance_per_agent": [],
        "energy_used_per_agent": [], "crash_count": [], "battery_depleted_count": [],
        "per_agent_value": [], "active_fraction": [],
    }

    for ep in range(n_episodes):
        obs_dict, _ = env.reset(seed=2000 + ep)
        agent_distance = {a: 0.0 for a in env.possible_agents}
        agent_energy = {a: 0.0 for a in env.possible_agents}
        agent_value = {a: 0.0 for a in env.possible_agents}
        agent_active_steps = {a: 0 for a in env.possible_agents}
        prev_pos = {a: env.uavs[a].state.position[:2].copy() for a in env.possible_agents}
        prev_battery = {a: env.uavs[a].state.battery_pct for a in env.possible_agents}
        crash_count = 0
        battery_depleted_count = 0
        total_steps = 0

        while env.agents:
            actions = {}
            for agent in env.agents:
                obs_t = torch.as_tensor(obs_dict[agent], dtype=torch.float32)
                with torch.no_grad():
                    action, _ = actor.act(obs_t.unsqueeze(0), deterministic=True)
                actions[agent] = np.clip(action.squeeze(0).numpy(), -1.0, 1.0)

                if hasattr(env, "assigned_target") and env.assigned_target.get(agent) is not None:
                    agent_active_steps[agent] += 1
                elif not hasattr(env, "assigned_target"):
                    agent_active_steps[agent] += 1

            served_before = env.served.sum()
            obs_dict, rewards, terminations, truncations, infos = env.step(actions)
            total_steps += 1

            for agent in list(prev_pos.keys()):
                if agent in env.uavs:
                    new_pos = env.uavs[agent].state.position[:2]
                    agent_distance[agent] += np.linalg.norm(new_pos - prev_pos[agent])
                    prev_pos[agent] = new_pos.copy()

                    new_batt = env.uavs[agent].state.battery_pct
                    agent_energy[agent] += max(0.0, prev_battery[agent] - new_batt)
                    prev_battery[agent] = new_batt

                    if env.uavs[agent].state.crashed:
                        crash_count += 1
                    if env.uavs[agent].state.battery_depleted:
                        battery_depleted_count += 1

            for agent, r in rewards.items():
                agent_value[agent] += max(0.0, r)

        results["n_served"].append(int(env.served.sum()))
        results["total_value"].append(sum(env.demand_weight[i] * env.demand_confidence[i]
                                           for i in range(env.n_demand) if env.served[i]))
        results["distance_per_agent"].append(np.mean(list(agent_distance.values())))
        results["energy_used_per_agent"].append(np.mean(list(agent_energy.values())))
        results["crash_count"].append(crash_count)
        results["battery_depleted_count"].append(battery_depleted_count)
        results["per_agent_value"].append(list(agent_value.values()))
        results["active_fraction"].append(np.mean([v / total_steps for v in agent_active_steps.values()]))

    return results, env.n_demand


def main():
    all_results = {}
    for name, (env_class, actor_path) in CONDITIONS.items():
        print(f"Değerlendiriliyor: {name}...")
        results, n_demand = evaluate_condition(env_class, actor_path)
        all_results[name] = results

    print("\n" + "=" * 100)
    print(f"{'Koşul':<20} {'Servis':<10} {'Değer':<10} {'Mesafe(m)':<12} {'Enerji(%)':<12} {'Çarpışma':<10} {'Jain':<8} {'Aktif%':<8}")
    print("=" * 100)
    for name, r in all_results.items():
        n_served = np.mean(r["n_served"])
        total_value = np.mean(r["total_value"])
        dist = np.mean(r["distance_per_agent"])
        energy = np.mean(r["energy_used_per_agent"])
        crash = np.mean(r["crash_count"])
        jain_vals = [jain_index(pv) for pv in r["per_agent_value"]]
        jain = np.mean(jain_vals)
        active = np.mean(r["active_fraction"]) * 100

        print(f"{name:<20} {n_served:<10.1f} {total_value:<10.3f} {dist:<12.0f} {energy:<12.2f} {crash:<10.2f} {jain:<8.3f} {active:<8.1f}")

    with open(PROC_DIR / "full_evaluation_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nDetaylı sonuçlar kaydedildi: {PROC_DIR / 'full_evaluation_results.pkl'}")

    return all_results


if __name__ == "__main__":
    main()
