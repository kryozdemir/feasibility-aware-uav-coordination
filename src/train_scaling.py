"""H5/H6 dogrulamasi: tutarli fizikle (rotasyonlu, 200Wh, kapasite 10, 30dk)
coklu filo buyuklugunde K1 (koordinasyonsuz) vs K4 (koordinasyonlu) egitimi."""
import sys, time, pickle
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from sklearn.cluster import KMeans

from disaster_uav_env import DisasterUAVEnv
from fair_allocation_env import FairAllocationUAVEnv
from hierarchical_env import NaiveManagerUAVEnv
from ppes_env import PPESManagerUAVEnv
from mappo_networks import Actor, CentralizedCritic
from train_mappo import PPOConfig, collect_rollout, ppo_update, DEVICE

CKPT_DIR = Path(__file__).parent.parent / "data" / "processed" / "seeds"
WALL_CLOCK_BUDGET_S = 190

_tmp = DisasterUAVEnv(n_agents=3, seed=0)
_km = KMeans(n_clusters=6, random_state=42, n_init=10).fit(_tmp.demand_xy)
REGION_LABELS = _km.labels_


def make_env(cond, n_agents, seed):
    if cond == "K1":
        return DisasterUAVEnv(n_agents=n_agents, seed=seed)
    elif cond == "K2":
        return NaiveManagerUAVEnv(n_agents=n_agents, seed=seed)
    elif cond == "K3":
        return PPESManagerUAVEnv(n_agents=n_agents, seed=seed)
    elif cond == "K4":
        return FairAllocationUAVEnv(n_agents=n_agents, seed=seed, region_labels=REGION_LABELS)
    raise ValueError(cond)


def train(cond, n_agents, seed, total_timesteps, log_every=15):
    cfg = PPOConfig()
    tag = f"{cond}_SCALE_n{n_agents}_seed{seed}"
    cp_path = CKPT_DIR / f"{tag}_checkpoint.pt"
    h_path = CKPT_DIR / f"{tag}_hist.pkl"

    torch.manual_seed(seed); np.random.seed(seed)
    env = make_env(cond, n_agents, seed)
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    action_dim = env.action_space(env.possible_agents[0]).shape[0]
    state_dim = env.global_state_dim()

    actor = Actor(obs_dim, action_dim).to(DEVICE)
    critic = CentralizedCritic(state_dim).to(DEVICE)
    actor_opt = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    steps_done = 0
    all_returns, all_served = [], []

    if cp_path.exists():
        ckpt = torch.load(cp_path, weights_only=False)
        actor.load_state_dict(ckpt["actor"]); critic.load_state_dict(ckpt["critic"])
        actor_opt.load_state_dict(ckpt["actor_opt"]); critic_opt.load_state_dict(ckpt["critic_opt"])
        steps_done = ckpt["steps_done"]
        obs_dict, _ = env.reset(seed=seed)
        with open(h_path, "rb") as f: prev = pickle.load(f)
        all_returns, all_served = prev["returns"], prev["served"]
        print(f"[{tag}] Checkpoint yuklendi: {steps_done} adimdan devam")
    else:
        obs_dict, _ = env.reset(seed=seed)

    state = env.get_global_state()
    t0 = time.time(); update_count = 0

    while steps_done < total_timesteps:
        if time.time() - t0 > WALL_CLOCK_BUDGET_S:
            print(f"[{tag}] Zaman budu doldu. steps_done={steps_done}")
            break
        buffer, obs_dict, state, bootstrap_value, still_active, ep_returns, ep_served = collect_rollout(
            env, actor, critic, cfg, obs_dict, state
        )
        actor_loss, critic_loss, entropy = ppo_update(
            buffer, actor, critic, actor_opt, critic_opt, cfg, bootstrap_value, still_active
        )
        steps_done += cfg.n_steps; update_count += 1
        all_returns.extend(ep_returns); all_served.extend(ep_served)

        if update_count % log_every == 0:
            print(f"[{tag}] steps={steps_done}/{total_timesteps} | son10 servis={np.mean(all_served[-10:]):.1f} | "
                  f"entropy={entropy:.3f} | sure={time.time()-t0:.0f}s")
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                        "actor_opt": actor_opt.state_dict(), "critic_opt": critic_opt.state_dict(),
                        "steps_done": steps_done}, cp_path)
            with open(h_path, "wb") as f:
                pickle.dump({"returns": all_returns, "served": all_served}, f)

    done = steps_done >= total_timesteps
    if done:
        torch.save(actor.state_dict(), CKPT_DIR / f"{tag}_actor_final.pt")
        print(f"[{tag}] TAMAMLANDI ({steps_done} adim)")
    return done, steps_done


if __name__ == "__main__":
    cond, n_agents, seed, total = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    done, steps = train(cond, n_agents, seed, total)
    print(f"RESULT done={done} steps_done={steps}")
