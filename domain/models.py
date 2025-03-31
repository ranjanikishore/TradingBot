import torch
import torch.nn as nn
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.__net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.__log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state: torch.Tensor) -> Normal:
        mean = self.__net(state)
        mean = torch.tanh(mean) * 1.0
        std = self.__log_std.exp().expand_as(mean)
        return Normal(mean, std)

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.__net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.__net(torch.cat([state, action], 1))