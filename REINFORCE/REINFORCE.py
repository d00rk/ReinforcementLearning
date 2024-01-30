import gym
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Policy Network
class REINFORCE(nn.Module):
    def __init__(self, n_states, n_actions):
        super(REINFORCE, self).__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, n_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x)
    
class REINFORCEAgent:
    def __init__(self, n_states, n_actions, lr, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = lr
        self.gamma = gamma
        self.policy = REINFORCE(n_states, n_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.rewards = []
        self.log_probs = []
        
    def get_action(self, state):
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob
    
    def remember(self, reward, log_prob):
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        
    def reset(self):
        self.rewards.clear()
        self.log_probs.clear()
            
    def train(self):
        R = 0
        loss = []
        returns = deque()
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.appendleft(R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-3)
        for log_prob, R in zip(self.log_probs, returns):
            loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        loss = torch.cat(loss).sum()
        loss.backward()
        self.optimizer.step()
        self.reset()
        
        return loss.item()