import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, n_state, n_action):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(n_state, 64)
        self.fc2 = nn.Linear(64, 64)
        self.policy = nn.Linear(64, n_action)
        self.value = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.policy(x))
        value = self.value(x)
        return policy, value