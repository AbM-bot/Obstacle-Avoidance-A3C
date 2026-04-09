import torch
import torch.nn as nn
import torch.nn.functional as F

class A3C(nn.Module):
    def __init__(self, state_size=6, action_size=4):
        super(A3C, self).__init__()

        self.fc1 = nn.Linear(state_size, 128)
        self.actor = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value