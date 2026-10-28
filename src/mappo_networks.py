
import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(64, 64)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden_sizes, action_dim])


        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

    def forward(self, obs):
        mean = self.net(obs)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def act(self, obs, deterministic=False):
        dist = self.forward(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        logp = dist.log_prob(action).sum(axis=-1)
        return action, logp

    def evaluate_actions(self, obs, actions):
        dist = self.forward(obs)
        logp = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return logp, entropy


class CentralizedCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_sizes=(128, 128)):
        super().__init__()
        self.net = mlp([state_dim, *hidden_sizes, 1])

    def forward(self, global_state):
        return self.net(global_state).squeeze(-1)
