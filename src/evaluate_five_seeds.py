"""
5-tohum istatistiksel protokolüyle tam metrik karşılaştırması.

Her koşul (K1-K4, K4b) için 5 farklı tohumla eğitilmiş modelin her biri
15 bölümde deterministik olarak değerlendirilir. Sonuçlar önce tohum
içinde (15 bölüm) ortalanır, sonra TOHUMLAR ARASI ortalama±std raporlanır
-- bu, makaledeki "her koşul en az 5 farklı rastgele tohumla koşturulacak,
ortalama±std raporlanacak" protokolüne karşılık gelir.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

from disaster_uav_env import DisasterUAVEnv
from hierarchical_env import NaiveManagerUAVEnv
from ppes_env import PPESManagerUAVEnv
from fair_allocation_env import FairAllocationUAVEnv
from flexible_fair_env import FlexibleFairUAVEnv
from mappo_networks import Actor

SEEDS_DIR = Path(__file__).parent.parent / "data" / "processed" / "seeds"
OUT_DIR = Path(__file__).parent.parent / "data" / "processed"
N_EVAL_EPISODES = 15
SEEDS = [0, 1, 2, 3, 4]
N_REGIONS = 6

import functools

CONDITIONS_TEMPLATE = {
    "K1_Duz_MARL": DisasterUAVEnv,
    "K2_Naif_Manager": NaiveManagerUAVEnv,
    "K3_PPES": PPESManagerUAVEnv,
    "K4_Adil_Paylasim": FairAllocationUAVEnv,
}


def jain_index(values):
    values = np.array(values, dtype=float)
    if values.sum() == 0:
        return 1.0
    n = len(values)
    return (values.sum() ** 2) / (n * np.sum(values ** 2))


def compute_region_labels(env):
    """Talep noktalarını sabit coğrafi kümelere ayırır (tüm koşullar için aynı)."""
    km = KMeans(n_clusters=N_REGIONS, random_state=42, n_init=10)
    labels = km.fit_predict(env.demand_xy)
    return labels


def evaluate_one_seed(env_class, actor_path, region_labels, n_agents=3, n_episodes=N_EVAL_EPISODES, eval_seed_base=2000):
    env = env_class(n_agents=n_agents, seed=1000)
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    action_dim = env.action_space(env.possible_agents[0]).shape[0]

    actor = Actor(obs_dim, action_dim)
    actor.load_state_dict(torch.load(actor_path, weights_only=True))
    actor.eval()


    region_total_value = np.zeros(N_REGIONS)
    for r in range(N_REGIONS):
        mask = region_labels == r
        region_total_value[r] = np.sum(env.demand_weight[mask] * env.demand_confidence[mask])


    is_confirmed = np.array([1.0 if s == "confirmed_destroyed" else 0.0 for s in
                              __import__("geopandas").read_file(OUT_DIR / "demand_points_v2.geojson")["source"]])
    confirmed_total_value = np.sum(env.demand_weight[is_confirmed == 1] * env.demand_confidence[is_confirmed == 1])
    potential_total_value = np.sum(env.demand_weight[is_confirmed == 0] * env.demand_confidence[is_confirmed == 0])

    ep_metrics = {
        "n_served": [], "total_value": [], "distance_per_agent": [],
        "energy_used_per_agent": [], "crash_count": [], "jain": [], "active_fraction": [],
        "region_jain": [], "region_min_coverage": [], "confirmed_coverage": [], "potential_coverage": [],
    }

    for ep in range(n_episodes):
        obs_dict, _ = env.reset(seed=eval_seed_base + ep)
        agent_distance = {a: 0.0 for a in env.possible_agents}
        agent_energy = {a: 0.0 for a in env.possible_agents}
        agent_value = {a: 0.0 for a in env.possible_agents}
        agent_active_steps = {a: 0 for a in env.possible_agents}
        prev_pos = {a: env.uavs[a].state.position[:2].copy() for a in env.possible_agents}
        prev_battery = {a: env.uavs[a].state.battery_pct for a in env.possible_agents}
        crash_count = 0
        total_steps = 0

        while env.agents:
            actions = {}
            for agent in env.agents:
                obs_t = torch.as_tensor(obs_dict[agent], dtype=torch.float32)
                with torch.no_grad():
                    action, _ = actor.act(obs_t.unsqueeze(0), deterministic=True)
                actions[agent] = np.clip(action.squeeze(0).numpy(), -1.0, 1.0)
                if hasattr(env, "assigned_target"):
                    if env.assigned_target.get(agent) is not None:
                        agent_active_steps[agent] += 1
                else:
                    agent_active_steps[agent] += 1

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
            for agent, r in rewards.items():
                agent_value[agent] += max(0.0, r)

        served = env.served
        ep_metrics["n_served"].append(int(served.sum()))
        ep_metrics["total_value"].append(sum(env.demand_weight[i] * env.demand_confidence[i]
                                              for i in range(env.n_demand) if served[i]))
        ep_metrics["distance_per_agent"].append(np.mean(list(agent_distance.values())))
        ep_metrics["energy_used_per_agent"].append(np.mean(list(agent_energy.values())))
        ep_metrics["crash_count"].append(crash_count)
        ep_metrics["jain"].append(jain_index(list(agent_value.values())))
        ep_metrics["active_fraction"].append(np.mean([v / total_steps for v in agent_active_steps.values()]))


        region_served_value = np.zeros(N_REGIONS)
        for r in range(N_REGIONS):
            mask = (region_labels == r) & served
            region_served_value[r] = np.sum(env.demand_weight[mask] * env.demand_confidence[mask])
        region_coverage = np.divide(region_served_value, region_total_value,
                                     out=np.zeros(N_REGIONS), where=region_total_value > 0)
        ep_metrics["region_jain"].append(jain_index(region_coverage))
        ep_metrics["region_min_coverage"].append(region_coverage.min())


        confirmed_served = np.sum(env.demand_weight[(is_confirmed == 1) & served] * env.demand_confidence[(is_confirmed == 1) & served])
        potential_served = np.sum(env.demand_weight[(is_confirmed == 0) & served] * env.demand_confidence[(is_confirmed == 0) & served])
        ep_metrics["confirmed_coverage"].append(confirmed_served / confirmed_total_value if confirmed_total_value > 0 else 0.0)
        ep_metrics["potential_coverage"].append(potential_served / potential_total_value if potential_total_value > 0 else 0.0)

    return {k: np.mean(v) for k, v in ep_metrics.items()}, env.n_demand


def main():
    all_results = {}


    tmp_env = DisasterUAVEnv(n_agents=3, seed=0)
    region_labels = compute_region_labels(tmp_env)
    print(f"Bölge etiketleri hesaplandı: {N_REGIONS} bölge, boyutları: {[int((region_labels==r).sum()) for r in range(N_REGIONS)]}")

    CONDITIONS = dict(CONDITIONS_TEMPLATE)
    CONDITIONS["K4_Adil_Paylasim"] = functools.partial(FairAllocationUAVEnv, region_labels=region_labels)

    for cond_name, env_class in CONDITIONS.items():
        print(f"\n=== {cond_name} ===")
        seed_level_results = []
        for seed in SEEDS:
            actor_path = SEEDS_DIR / f"{cond_name.split('_')[0]}_seed{seed}_actor_final.pt"
            if not actor_path.exists():
                print(f"  UYARI: {actor_path} bulunamadı, atlanıyor.")
                continue
            seed_metrics, n_demand = evaluate_one_seed(env_class, actor_path, region_labels)
            seed_level_results.append(seed_metrics)
            print(f"  seed{seed}: servis={seed_metrics['n_served']:.1f}, değer={seed_metrics['total_value']:.4f}, "
                  f"jain={seed_metrics['jain']:.3f}, bölge_jain={seed_metrics['region_jain']:.3f}, "
                  f"confirmed_kapsama={seed_metrics['confirmed_coverage']:.2f}, potential_kapsama={seed_metrics['potential_coverage']:.3f}")


        agg = {}
        for key in seed_level_results[0].keys():
            vals = [r[key] for r in seed_level_results]
            agg[key] = {"mean": np.mean(vals), "std": np.std(vals), "values": vals}
        all_results[cond_name] = agg


    print("\n" + "=" * 150)
    print(f"{'Koşul':<20} {'Servis':<14} {'Değer':<16} {'Jain(ajan)':<14} {'Jain(bölge)':<14} {'Min bölge kapsama':<18} {'Confirmed kap.':<16} {'Potential kap.':<16}")
    print("=" * 150)
    for cond_name, agg in all_results.items():
        def fmt(key, scale=1.0, prec=3):
            m, s = agg[key]["mean"] * scale, agg[key]["std"] * scale
            return f"{m:.{prec}f}±{s:.{prec}f}"

        print(f"{cond_name:<20} {fmt('n_served',1,1):<14} {fmt('total_value',1,4):<16} "
              f"{fmt('jain',1,3):<14} {fmt('region_jain',1,3):<14} {fmt('region_min_coverage',1,3):<18} "
              f"{fmt('confirmed_coverage',1,3):<16} {fmt('potential_coverage',1,3):<16}")

    with open(OUT_DIR / "five_seed_full_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nKaydedildi: {OUT_DIR / 'five_seed_full_results.pkl'}")

    return all_results


if __name__ == "__main__":
    main()
