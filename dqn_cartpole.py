# Libraries
import gym
import math
import random
import matplotlib.pyplot as plt
from collections import deque
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# openai gym CartPole environment
env = gym.make("CartPole-v1")
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

# Hyperparameters
BATCH_SIZE = 64
EPISODES = 200
GAMMA = 0.99
EPS = 0.9
EPS_DECAY = 200
EPS_END = 0.05
LR = 0.0001

# Q-Network
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
# Agent
class DQNAgent:
    def __init__(self):
        self.model = DQN(n_observations, n_actions)
        self.target = DQN(n_observations, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.steps_done = 0
        self.memory = deque(maxlen=10000)
        
        self.target.load_state_dict(self.model.state_dict())
        
    def memorize(self, state, action, reward, next_state):
        self.memory.append((state, action, torch.FloatTensor([reward]), torch.FloatTensor([next_state])))
        
    def act(self, state):
        eps_threshold = EPS_END + (EPS - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if random.random() > eps_threshold:
            return self.model(state).data.max(1)[1].view(1, 1)
        else:
            return torch.LongTensor([[random.randrange(n_actions)]])
        
    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
        
        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)
        
        q_value = self.model(states).gather(1, actions)
        max_next_q = self.model(next_states).detach().max(1)[0]
        
        expected_q = rewards + (GAMMA * max_next_q)
        
        loss = F.mse_loss(q_value.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.target.load_state_dict(self.model.state_dict())
        

agent = DQNAgent()
scores = []

for episode in range(EPISODES):
    state = env.reset()
    steps = 0
    for t in count():
        env.render()
        state = torch.FloatTensor([state])
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action.item())
        if done:
            reward = -100
        agent.memorize(state, action, reward, next_state)
        agent.learn()
        state = next_state
        steps += 1
        
        if done:
            print(f"episode {episode+1} | score: {steps}")
            scores.append(steps)
            break

env.close()
     
plt.plot(scores)
plt.show()