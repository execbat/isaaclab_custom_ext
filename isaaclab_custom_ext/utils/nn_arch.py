import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, n_obs, n_actions,
                 log_std_init=-1.2, log_std_min=-4.0, log_std_max=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 512), nn.ELU(),
            nn.Linear(512, 256),   nn.ELU(),
            nn.Linear(256, 128),   nn.ELU(),
        )
        self.mu_head = nn.Linear(128, n_actions)
        self.ls_head = nn.Linear(128, n_actions)      # state-dependent log-std
        self.log_std_bias = nn.Parameter(torch.full((n_actions,), log_std_init))
        self.register_buffer("ls_min", torch.tensor(log_std_min))
        self.register_buffer("ls_max", torch.tensor(log_std_max))

        # init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.4) 
                nn.init.zeros_(m.bias)
        nn.init.uniform_(self.mu_head.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.ls_head.weight)
        nn.init.zeros_(self.ls_head.bias)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu_head(h)                     
        raw = self.log_std_bias + self.ls_head(h) # (-inf, +inf)
        log_std = self.ls_min + 0.5*(self.ls_max - self.ls_min) * (torch.tanh(raw) + 1.0)
        std = torch.exp(log_std)
        return mu, std


class Critic(nn.Module):
    def __init__(self, n_obs):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_obs, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)
