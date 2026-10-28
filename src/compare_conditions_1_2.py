

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from train_mappo import main
from disaster_uav_env import DisasterUAVEnv
from hierarchical_env import NaiveManagerUAVEnv

TOTAL_TIMESTEPS = 300_000

if __name__ == "__main__":
    print("=" * 60)
    print("KOŞUL 1 (Düz MARL) -- Uzun Eğitim")
    print("=" * 60)
    actor1, critic1, returns1, served1, env1 = main(
        total_timesteps=TOTAL_TIMESTEPS, n_agents=3, log_every=10, env_class=DisasterUAVEnv
    )
    torch.save(actor1.state_dict(), Path(__file__).parent.parent / "data" / "processed" / "actor_condition1_long.pt")

    print()
    print("=" * 60)
    print("KOŞUL 2 (Naif Manager) -- Uzun Eğitim")
    print("=" * 60)
    actor2, critic2, returns2, served2, env2 = main(
        total_timesteps=TOTAL_TIMESTEPS, n_agents=3, log_every=10, env_class=NaiveManagerUAVEnv
    )
    torch.save(actor2.state_dict(), Path(__file__).parent.parent / "data" / "processed" / "actor_condition2_long.pt")


    def smooth(x, w=10):
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(smooth(served1), label=f"Koşul 1 (Düz MARL), son 10 ort: {np.mean(served1[-10:]):.1f}")
    axes[0].plot(smooth(served2), label=f"Koşul 2 (Naif Manager), son 10 ort: {np.mean(served2[-10:]):.1f}")
    axes[0].axhline(31, color="gray", linestyle="--", label="Rastgele politika referansı (31)")
    axes[0].set_xlabel("Bölüm (episode, 10-hareketli ortalama)")
    axes[0].set_ylabel("Servis edilen talep sayısı")
    axes[0].set_title(f"Servis Karşılaştırması ({TOTAL_TIMESTEPS:,} adım eğitim)")
    axes[0].legend()

    axes[1].plot(smooth(returns1), label="Koşul 1 (Düz MARL)")
    axes[1].plot(smooth(returns2), label="Koşul 2 (Naif Manager)")
    axes[1].set_xlabel("Bölüm (episode, 10-hareketli ortalama)")
    axes[1].set_ylabel("Toplam ödül")
    axes[1].set_title("Ödül Karşılaştırması")
    axes[1].legend()

    plt.tight_layout()
    out_path = Path(__file__).parent.parent / "data" / "processed" / "condition1_vs_2_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nKarşılaştırma grafiği kaydedildi: {out_path}")

    print(f"\nSON DURUM:")
    print(f"  Koşul 1 (Düz MARL)    -- son 10 bölüm ort. servis: {np.mean(served1[-10:]):.1f} / {env1.n_demand}")
    print(f"  Koşul 2 (Naif Manager) -- son 10 bölüm ort. servis: {np.mean(served2[-10:]):.1f} / {env2.n_demand}")
