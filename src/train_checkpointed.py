"""
Checkpoint destekli MAPPO eğitim sürücüsü.

Tek bir tool-call'un zaman sınırını (~250-260sn) aşan uzun eğitimleri
(450k adım gibi) güvenle tamamlamak için: eğitim, belirli bir duvar-saati
bütçesi içinde çalışır, bütçe dolunca TÜM durumu (actor, critic,
optimizer'lar, ortam gözlemi, epizot geçmişi) diske kaydedip çıkar.
Aynı komutu tekrar çalıştırdığınızda kaldığı yerden (optimizer momentumu
dahil) devam eder -- bu, önceki "sadece actor ağırlığını kaydet, kritiği
sıfırdan başlat" yönteminden daha doğru bir devam ettirme sağlar.

Kullanım (komut satırından):
    python3 train_checkpointed.py <condition> <seed> <total_timesteps>

condition: K1, K2, K3, K4, K4b
"""

import sys
import time
import pickle
from pathlib import Path
import functools

import numpy as np
import torch
import torch.optim as optim
from sklearn.cluster import KMeans

from disaster_uav_env import DisasterUAVEnv
from hierarchical_env import NaiveManagerUAVEnv
from ppes_env import PPESManagerUAVEnv
from fair_allocation_env import FairAllocationUAVEnv
from flexible_fair_env import FlexibleFairUAVEnv
from flexible_fair_coverage_env import FlexibleFairCoverageUAVEnv
from hybrid_fair_env import HybridFairUAVEnv
from region_partitioned_env import RegionPartitionedUAVEnv
from coverage_reward_env import CoverageRewardUAVEnv
from naive_coverage_env import NaiveCoverageUAVEnv
from fair_allocation_flex_env import FairAllocationFlexUAVEnv
from mappo_networks import Actor, CentralizedCritic
from train_mappo import PPOConfig, collect_rollout, ppo_update, DEVICE

CKPT_DIR = Path(__file__).parent.parent / "data" / "processed" / "seeds"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

WALL_CLOCK_BUDGET_S = 190


_tmp_env = DisasterUAVEnv(n_agents=3, seed=0)
_km = KMeans(n_clusters=6, random_state=42, n_init=10)
REGION_LABELS = _km.fit_predict(_tmp_env.demand_xy)

ENV_CLASSES = {
    "K1": DisasterUAVEnv,
    "K2": NaiveManagerUAVEnv,
    "K3": PPESManagerUAVEnv,
    "K4": functools.partial(FairAllocationUAVEnv, region_labels=REGION_LABELS),
    "K4b": functools.partial(FlexibleFairUAVEnv, region_labels=REGION_LABELS),
    "K4c": functools.partial(FlexibleFairCoverageUAVEnv, region_labels=REGION_LABELS),
    "K4h": functools.partial(HybridFairUAVEnv, region_labels=REGION_LABELS),
    "K1rp": functools.partial(RegionPartitionedUAVEnv, region_labels=REGION_LABELS, max_episode_steps=600),
    "K4d": functools.partial(FairAllocationFlexUAVEnv, region_labels=REGION_LABELS),
    "K1c": functools.partial(CoverageRewardUAVEnv, region_labels=REGION_LABELS),
    "K2c": functools.partial(NaiveCoverageUAVEnv, region_labels=REGION_LABELS),
}


def ckpt_path(condition, seed, n_agents=3):
    suffix = "" if n_agents == 3 else f"_n{n_agents}"
    return CKPT_DIR / f"{condition}{suffix}_seed{seed}_checkpoint.pt"


def hist_path(condition, seed, n_agents=3):
    suffix = "" if n_agents == 3 else f"_n{n_agents}"
    return CKPT_DIR / f"{condition}{suffix}_seed{seed}_hist.pkl"


def final_actor_path(condition, seed, n_agents=3):
    suffix = "" if n_agents == 3 else f"_n{n_agents}"
    return CKPT_DIR / f"{condition}{suffix}_seed{seed}_actor_final.pt"


def train_condition_seed(condition: str, seed: int, total_timesteps: int, n_agents: int = 3, log_every: int = 15):
    env_class = ENV_CLASSES[condition]
    cfg = PPOConfig()

    cp_path = ckpt_path(condition, seed, n_agents)
    h_path = hist_path(condition, seed, n_agents)

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = env_class(n_agents=n_agents, seed=seed)

    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    action_dim = env.action_space(env.possible_agents[0]).shape[0]
    state_dim = env.global_state_dim()

    actor = Actor(obs_dim, action_dim).to(DEVICE)
    critic = CentralizedCritic(state_dim).to(DEVICE)
    actor_opt = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    steps_done = 0
    all_episode_returns = []
    all_episode_n_served = []

    if cp_path.exists():
        ckpt = torch.load(cp_path, weights_only=False)
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        actor_opt.load_state_dict(ckpt["actor_opt"])
        critic_opt.load_state_dict(ckpt["critic_opt"])
        steps_done = ckpt["steps_done"]
        torch.manual_seed(seed + 1000 + steps_done)
        obs_dict, _ = env.reset(seed=seed)


        with open(h_path, "rb") as f:
            prev_hist = pickle.load(f)
        all_episode_returns = prev_hist["returns"]
        all_episode_n_served = prev_hist["served"]
        print(f"[{condition} seed{seed}] Checkpoint yüklendi: {steps_done} adımdan devam ediliyor.")
    else:
        obs_dict, _ = env.reset(seed=seed)

    state = env.get_global_state()

    t0 = time.time()
    update_count = 0
    while steps_done < total_timesteps:
        if time.time() - t0 > WALL_CLOCK_BUDGET_S:
            print(f"[{condition} seed{seed}] Zaman bütçesi doldu, checkpoint kaydediliyor. steps_done={steps_done}")
            break

        buffer, obs_dict, state, bootstrap_value, still_active, ep_returns, ep_served = collect_rollout(
            env, actor, critic, cfg, obs_dict, state
        )
        actor_loss, critic_loss, entropy = ppo_update(
            buffer, actor, critic, actor_opt, critic_opt, cfg, bootstrap_value, still_active
        )
        steps_done += cfg.n_steps
        update_count += 1

        all_episode_returns.extend(ep_returns)
        all_episode_n_served.extend(ep_served)

        if update_count % log_every == 0:
            recent_served = all_episode_n_served[-10:] if all_episode_n_served else [0]
            recent_returns = all_episode_returns[-10:] if all_episode_returns else [0]
            print(f"[{condition} seed{seed}] steps={steps_done}/{total_timesteps} | "
                  f"son10 servis: {np.mean(recent_served):.1f} | son10 ödül: {np.mean(recent_returns):.2f} | "
                  f"entropy: {entropy:.3f} | süre: {time.time()-t0:.0f}s")


            torch.save({
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "actor_opt": actor_opt.state_dict(),
                "critic_opt": critic_opt.state_dict(),
                "steps_done": steps_done,
            }, cp_path)
            with open(h_path, "wb") as f:
                pickle.dump({"returns": all_episode_returns, "served": all_episode_n_served, "n_demand": env.n_demand}, f)


    torch.save({
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_opt": actor_opt.state_dict(),
        "critic_opt": critic_opt.state_dict(),
        "steps_done": steps_done,
    }, cp_path)
    with open(h_path, "wb") as f:
        pickle.dump({"returns": all_episode_returns, "served": all_episode_n_served, "n_demand": env.n_demand}, f)

    done = steps_done >= total_timesteps
    if done:
        torch.save(actor.state_dict(), final_actor_path(condition, seed, n_agents))
        print(f"[{condition} seed{seed}] TAMAMLANDI ({steps_done} adım).")
    return done, steps_done


if __name__ == "__main__":
    condition = sys.argv[1]
    seed = int(sys.argv[2])
    total_timesteps = int(sys.argv[3])
    n_agents = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    done, steps_done = train_condition_seed(condition, seed, total_timesteps, n_agents=n_agents)
    print(f"RESULT done={done} steps_done={steps_done}")
