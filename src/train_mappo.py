
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from disaster_uav_env import DisasterUAVEnv
from mappo_networks import Actor, CentralizedCritic

DEVICE = torch.device("cpu")


@dataclass
class PPOConfig:
    n_steps: int = 2048
    n_epochs: int = 10
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    max_grad_norm: float = 0.5


class RolloutBuffer:
    """
    Ajan-bazlı ayrı diziler tutuyor -- GAE hesabı zamansal olarak sıralı
    TEK bir ajanın ödül/değer akışı üzerinde doğru çalışır. Farklı ajanların
    verilerini düz bir diziye karıştırmak (önceki hatalı versiyon) GAE'nin
    zamansal varsayımını bozar ve anlamsız avantajlar üretir.
    """

    def __init__(self, possible_agents):
        self.possible_agents = possible_agents
        self.data = {a: {"obs": [], "actions": [], "logps": [], "rewards": [],
                          "dones": [], "values": [], "global_states": []}
                     for a in possible_agents}

    def add(self, agent, obs, action, logp, reward, done, value, global_state):
        d = self.data[agent]
        d["obs"].append(obs)
        d["actions"].append(action)
        d["logps"].append(logp)
        d["rewards"].append(reward)
        d["dones"].append(done)
        d["values"].append(value)
        d["global_states"].append(global_state)


def compute_gae(rewards, values, dones, next_value, gamma, lam):
    """
    rewards, values, dones: (T,) tek bir ajan/akış için
    next_value: rollout sonrasındaki (bootstrap) değer tahmini
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_v = next_value if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def collect_rollout(env, actor, critic, cfg: PPOConfig, obs_dict, state):
    buffer = RolloutBuffer(env.possible_agents)
    episode_returns = []
    episode_n_served = []
    current_ep_reward = {a: 0.0 for a in env.possible_agents}


    steps_collected = 0
    while steps_collected < cfg.n_steps:
        actions_dict = {}
        logps_dict = {}
        obs_this_step = {}
        agent_order = list(obs_dict.keys())

        for agent in agent_order:
            obs_t = torch.as_tensor(obs_dict[agent], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                action, logp = actor.act(obs_t.unsqueeze(0))
            action_np = action.squeeze(0).numpy()
            actions_dict[agent] = np.clip(action_np, -1.0, 1.0)
            logps_dict[agent] = logp.item()
            obs_this_step[agent] = obs_dict[agent]

        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
            value = critic(state_t.unsqueeze(0)).item()

        next_obs_dict, rewards, terminations, truncations, infos = env.step(actions_dict)

        for agent in agent_order:
            done = terminations[agent] or truncations[agent]
            buffer.add(
                agent,
                obs_this_step[agent],
                actions_dict[agent],
                logps_dict[agent],
                rewards[agent],
                float(done),
                value,
                state,
            )
            current_ep_reward[agent] += rewards[agent]
            steps_collected += 1

        state = env.get_global_state()

        episode_done = len(env.agents) == 0
        if episode_done:
            episode_returns.append(sum(current_ep_reward.values()))
            episode_n_served.append(infos.get("__common__", {}).get("n_served", 0))
            current_ep_reward = {a: 0.0 for a in env.possible_agents}
            next_obs_dict, _ = env.reset()
            state = env.get_global_state()

        obs_dict = next_obs_dict


    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
        bootstrap_value = critic(state_t.unsqueeze(0)).item()
    still_active = set(obs_dict.keys())

    return buffer, obs_dict, state, bootstrap_value, still_active, episode_returns, episode_n_served


def ppo_update(buffer: RolloutBuffer, actor, critic, actor_opt, critic_opt, cfg: PPOConfig,
                bootstrap_value: float, still_active: set):


    all_obs, all_actions, all_logps = [], [], []
    all_advantages, all_returns, all_global_states = [], [], []

    for agent in buffer.possible_agents:
        d = buffer.data[agent]
        if len(d["rewards"]) == 0:
            continue
        rewards = np.array(d["rewards"], dtype=np.float32)
        values = np.array(d["values"], dtype=np.float32)
        dones = np.array(d["dones"], dtype=np.float32)

        agent_bootstrap = bootstrap_value if agent in still_active else 0.0
        advantages, returns = compute_gae(rewards, values, dones, agent_bootstrap, cfg.gamma, cfg.gae_lambda)

        all_obs.extend(d["obs"])
        all_actions.extend(d["actions"])
        all_logps.extend(d["logps"])
        all_global_states.extend(d["global_states"])
        all_advantages.append(advantages)
        all_returns.append(returns)

    advantages = np.concatenate(all_advantages)
    returns = np.concatenate(all_returns)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs = torch.as_tensor(np.array(all_obs), dtype=torch.float32, device=DEVICE)
    actions = torch.as_tensor(np.array(all_actions), dtype=torch.float32, device=DEVICE)
    old_logps = torch.as_tensor(np.array(all_logps), dtype=torch.float32, device=DEVICE)
    global_states = torch.as_tensor(np.array(all_global_states), dtype=torch.float32, device=DEVICE)
    advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=DEVICE)
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)

    n = len(obs)
    idxs = np.arange(n)

    actor_losses, critic_losses, entropies = [], [], []

    for epoch in range(cfg.n_epochs):
        np.random.shuffle(idxs)
        for start in range(0, n, cfg.batch_size):
            batch_idx = idxs[start:start + cfg.batch_size]

            logp, entropy = actor.evaluate_actions(obs[batch_idx], actions[batch_idx])
            ratio = torch.exp(logp - old_logps[batch_idx])

            adv_b = advantages_t[batch_idx]
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv_b
            actor_loss = -torch.min(surr1, surr2).mean() - cfg.entropy_coef * entropy.mean()

            value_pred = critic(global_states[batch_idx])
            critic_loss = ((value_pred - returns_t[batch_idx]) ** 2).mean()

            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            actor_opt.step()

            critic_opt.zero_grad()
            (critic_loss * cfg.value_coef).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
            critic_opt.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy.mean().item())

    return np.mean(actor_losses), np.mean(critic_losses), np.mean(entropies)


def main(total_timesteps: int = 100_000, n_agents: int = 3, log_every: int = 5, env_class=None,
         random_baseline: float = 31.0, resume_actor_path=None, seed: int = 0):
    if env_class is None:
        env_class = DisasterUAVEnv
    cfg = PPOConfig()


    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_class(n_agents=n_agents, seed=seed)

    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    action_dim = env.action_space(env.possible_agents[0]).shape[0]
    state_dim = env.global_state_dim()

    actor = Actor(obs_dim, action_dim).to(DEVICE)
    if resume_actor_path is not None:
        actor.load_state_dict(torch.load(resume_actor_path, weights_only=True))
        print(f"Aktör ağırlıkları yüklendi: {resume_actor_path} (kaldığı yerden devam)")
    critic = CentralizedCritic(state_dim).to(DEVICE)
    actor_opt = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    obs_dict, _ = env.reset(seed=seed)
    state = env.get_global_state()

    n_updates = total_timesteps // cfg.n_steps
    print(f"Toplam güncelleme sayısı: {n_updates} (her biri {cfg.n_steps} adım)")

    all_episode_returns = []
    all_episode_n_served = []
    update_log = []

    t0 = time.time()
    for update in range(n_updates):
        buffer, obs_dict, state, bootstrap_value, still_active, ep_returns, ep_served = collect_rollout(
            env, actor, critic, cfg, obs_dict, state
        )
        actor_loss, critic_loss, entropy = ppo_update(
            buffer, actor, critic, actor_opt, critic_opt, cfg, bootstrap_value, still_active
        )

        all_episode_returns.extend(ep_returns)
        all_episode_n_served.extend(ep_served)

        if update % log_every == 0 or update == n_updates - 1:
            recent_served = all_episode_n_served[-10:] if all_episode_n_served else [0]
            recent_returns = all_episode_returns[-10:] if all_episode_returns else [0]
            elapsed = time.time() - t0
            print(f"[Update {update+1}/{n_updates}] "
                  f"episodes so far: {len(all_episode_returns)} | "
                  f"son 10 ep. ort. servis: {np.mean(recent_served):.1f}/{env.n_demand} | "
                  f"son 10 ep. ort. ödül: {np.mean(recent_returns):.2f} | "
                  f"actor_loss: {actor_loss:.4f} | critic_loss: {critic_loss:.4f} | "
                  f"entropy: {entropy:.3f} | süre: {elapsed:.0f}s")

    return actor, critic, all_episode_returns, all_episode_n_served, env


if __name__ == "__main__":
    actor, critic, returns_hist, served_hist, env = main(total_timesteps=100_000, n_agents=3)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(returns_hist)
    axes[0].set_xlabel("Bölüm (episode)"); axes[0].set_ylabel("Toplam ödül")
    axes[0].set_title("Eğitim Sırasında Ödül")

    axes[1].plot(served_hist)
    axes[1].axhline(31, color="gray", linestyle="--", label="Rastgele politika referansı (31)")
    axes[1].set_xlabel("Bölüm (episode)"); axes[1].set_ylabel("Servis edilen talep sayısı")
    axes[1].set_title("Eğitim Sırasında Servis Sayısı")
    axes[1].legend()

    plt.tight_layout()
    out_path = Path(__file__).parent.parent / "data" / "processed" / "mappo_training_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Grafik kaydedildi: {out_path}")

    torch.save(actor.state_dict(), Path(__file__).parent.parent / "data" / "processed" / "actor_condition1.pt")
    print("Model kaydedildi.")
