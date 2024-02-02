import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque

class A2C(nn.Module):
    def __init__(self, n_states, n_actions):
        super(A2C, self).__init__()
        self.fc1 = nn.Linear(n_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        policy = F.softmax(self.actor(x))
        critic = self.critic(x)
        return policy, critic
        
class A2Cagent:
    def __init__(self, n_states, n_actions, lr, gamma, action_ratio):
        self.gamma = gamma
        self.action_ratio = action_ratio
        
        self.A2Cnet = A2C(n_states, n_actions)
        self.optimizer = optim.Adam(self.A2Cnet.parameters(), lr=lr)
        
        self.states = []
        self.action_lps = []
        self.advantages = []
        self.y_is = []
        
    def get_action(self, state):
        probs, _ = self.A2Cnet(state)
        dist = Categorical(probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action, action_log_prob
    
    def get_q_value(self, state):
        _, q = self.A2Cnet(state)
        return q
        
    def remember(self, state, action_log_prob, advantage, y_i):
        self.states.append(state)
        self.action_lps.append(action_log_prob)
        self.advantages.append(advantage)
        self.y_is.append(y_i)
        
    def clear(self):
        self.states.clear()
        self.action_lps.clear()
        self.advantages.clear()
        self.y_is.clear()
    
    def advantage_td_target(self, reward, q, next_q, done):
        q = q.detach()
        next_q = next_q.detach()
        reward = torch.tensor([[reward]])
        target = reward + self.gamma * next_q * (1 - done)
        advantage = target - q
        return advantage, target
    
    def train(self):
        states = torch.cat(self.states)
        action_lps = torch.cat(self.action_lps)
        advantages = torch.cat(self.advantages)
        y_is = torch.cat(self.y_is)
        
        self.clear()
        
        _, predict = self.A2Cnet(states)
        critic_loss = torch.mean((predict - y_is)**2)
        critic_loss.requires_grad_(True)     
        actor_loss = -1.*(torch.mean(advantages*action_lps))
        actor_loss.requires_grad_(True)
        loss = self.action_ratio * actor_loss + critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        