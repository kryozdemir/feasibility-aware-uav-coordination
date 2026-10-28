"""Koşul 2: Naif Manager + MAPPO worker eğitimi."""

from pathlib import Path
import torch
import matplotlib.pyplot as plt

from train_mappo import main
from hierarchical_env import NaiveManagerUAVEnv

if __name__ == "__main__":
    actor, critic, returns_hist, served_hist, env = main(
        total_timesteps=100_000, n_agents=3, log_every=5, env_class=NaiveManagerUAVEnv
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(returns_hist)
    axes[0].set_xlabel("Bölüm (episode)"); axes[0].set_ylabel("Toplam ödül")
    axes[0].set_title("Koşul 2 (Naif Manager): Eğitim Sırasında Ödül")

    axes[1].plot(served_hist, label="Koşul 2 (Naif Manager)")
    axes[1].axhline(31, color="gray", linestyle="--", label="Rastgele politika referansı (31)")
    axes[1].set_xlabel("Bölüm (episode)"); axes[1].set_ylabel("Servis edilen talep sayısı")
    axes[1].set_title("Koşul 2: Eğitim Sırasında Servis Sayısı")
    axes[1].legend()

    plt.tight_layout()
    out_path = Path(__file__).parent.parent / "data" / "processed" / "condition2_training_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Grafik kaydedildi: {out_path}")

    torch.save(actor.state_dict(), Path(__file__).parent.parent / "data" / "processed" / "actor_condition2.pt")
    print("Model kaydedildi.")
