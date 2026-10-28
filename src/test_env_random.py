"""
DisasterUAVEnv'i rastgele aksiyonlarla test etme (öğrenme yok).

Amaç: ortam mekaniğinin (fizik, servis kontrolü, ödül, bitiş koşulları)
hatasız çalıştığını doğrulamak. Rastgele ajanlar tabii ki çok az talep
servis edecek -- bu normal, henüz öğrenme yok.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from disaster_uav_env import DisasterUAVEnv


def main():
    env = DisasterUAVEnv(n_agents=3, seed=0)
    observations, infos = env.reset(seed=0)

    print(f"Ajan sayısı: {len(env.agents)}")
    print(f"Toplam talep noktası: {env.n_demand}")
    print(f"Gözlem boyutu: {observations[env.agents[0]].shape}")
    print(f"Aksiyon uzayı: {env.action_space(env.agents[0])}")
    print()

    trajectories = {agent: [env.uavs[agent].state.position[:2].copy()] for agent in env.agents}
    total_rewards = {agent: 0.0 for agent in env.agents}
    n_served_history = []

    step = 0
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)

        for agent in list(trajectories.keys()):
            if agent in env.uavs and not env.uavs[agent].state.crashed:
                trajectories[agent].append(env.uavs[agent].state.position[:2].copy())

        for agent, r in rewards.items():
            total_rewards[agent] += r

        n_served_history.append(infos["__common__"]["n_served"])
        step += 1

        if step % 50 == 0:
            print(f"Adım {step}: {infos['__common__']['n_served']}/{infos['__common__']['n_total']} servis edildi, aktif ajan: {len(env.agents)}")

    print(f"\nBölüm bitti. Toplam adım: {step}")
    print(f"Toplam servis edilen: {n_served_history[-1]} / {env.n_demand}")
    print(f"Toplam ödüller: {total_rewards}")


    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax = axes[0]
    ax.scatter(env.demand_xy[:, 0], env.demand_xy[:, 1], c="lightgray", s=10, label="Servis edilmemiş", zorder=1)
    served_xy = env.demand_xy[env.served]
    if len(served_xy) > 0:
        ax.scatter(served_xy[:, 0], served_xy[:, 1], c="green", s=30, label="Servis edildi", zorder=2)
    ax.scatter(*env.depot_xy, c="blue", s=200, marker="*", label="Depo", zorder=5)

    colors = ["red", "orange", "purple", "brown", "pink"]
    for i, (agent, traj) in enumerate(trajectories.items()):
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], color=colors[i % len(colors)], alpha=0.7, label=f"{agent} yörüngesi")

    ax.set_title(f"Rastgele Ajanlar -- Yörüngeler ({step} adım)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")

    ax = axes[1]
    ax.plot(n_served_history)
    ax.set_xlabel("Adım")
    ax.set_ylabel("Servis edilen talep sayısı")
    ax.set_title("Zaman İçinde Servis İlerlemesi (rastgele politika)")

    plt.tight_layout()
    out_path = Path(__file__).parent.parent / "data" / "processed" / "random_policy_test.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nGörsel kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
