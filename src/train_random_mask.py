
import sys, time, pickle
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim

from disaster_uav_env import DisasterUAVEnv
from hierarchical_env import NaiveManagerUAVEnv
from mappo_networks import Actor, CentralizedCritic
from train_mappo import PPOConfig, collect_rollout, ppo_update, DEVICE

CKPT_DIR = Path(__file__).parent.parent / "data" / "processed" / "seeds"
WALL_CLOCK_BUDGET_S = 190
DEMAND_FILE = "demand_points_random_mask.geojson"

ENV_CLASSES = {"K1": DisasterUAVEnv, "K2": NaiveManagerUAVEnv}


def train(condition, seed, total_timesteps, n_agents=10, log_every=15):
    env_class = ENV_CLASSES[condition]
    cfg = PPOConfig()
    tag = f"{condition}_RANDMASK_n{n_agents}_seed{seed}"
    cp_path = CKPT_DIR / f"{tag}_checkpoint.pt"
    h_path = CKPT_DIR / f"{tag}_hist.pkl"

    torch.manual_seed(seed); np.random.seed(seed)
    env = env_class(n_agents=n_agents, seed=seed, demand_file=DEMAND_FILE)

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
            print(f"[{tag}] Zaman budu doldu, kaydediliyor. steps_done={steps_done}")
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
            rs = all_served[-10:] if all_served else [0]
            rr = all_returns[-10:] if all_returns else [0]
            print(f"[{tag}] steps={steps_done}/{total_timesteps} | son10 servis={np.mean(rs):.1f} | "
                  f"son10 odul={np.mean(rr):.2f} | entropy={entropy:.3f} | sure={time.time()-t0:.0f}s")
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
    cond, seed, total = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    n_agents = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    done, steps = train(cond, seed, total, n_agents)
    print(f"RESULT done={done} steps_done={steps}")
