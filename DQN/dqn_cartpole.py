# Libraries
import gym
import math
import random
import numpy as np
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

# device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 128
EPISODES = 500
GAMMA = 0.99
EPS = 0.9
EPS_DECAY = 0.99
EPS_END = 0.05
LR = 0.0001

# Q-Network
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)
    
# DQN Agent
class DQNAgent:
    def __init__(self, n_observations, n_actions):
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.model = DQN(n_observations, n_actions)
        self.target = DQN(n_observations, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.steps_done = 0
        self.memory = deque(maxlen=2000)
        
        self.target.load_state_dict(self.model.state_dict())
        
    def memorize(self, state, action, reward, next_state):
        self.memory.append((state, action, torch.FloatTensor([reward]), torch.FloatTensor([next_state])))
        
    def choose_action(self, state):
        eps_threshold = EPS_END + (EPS - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        
        if random.random() >= eps_threshold:
            return self.model(state).data.max(1)[1].view(1, 1)
        else:
            return torch.LongTensor([[random.randrange(self.n_actions)]])
    
    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0
        
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
        
        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)
        
        q_value = self.model(states).gather(1, actions)
        max_next_q = self.target(next_states).detach().max(1)[0]
        
        expected_q = rewards + (GAMMA * max_next_q)
        
        loss = F.mse_loss(q_value.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.trace_func = trace_func
        self.counter = 0
        self.loss_min = None
        self.stop = False
        
    def __call__(self, loss, model, file_name):
        if loss == 0.:
            return
        
        if self.loss_min is None:
            self.loss_min = loss
            self.save_checkpoint(loss, model, file_name)
        elif loss > self.loss_min + self.delta:
            self.counter += 1
            self.trace_func(f"Early Stopping Counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.loss_min = loss
            self.counter = 0
            self.save_checkpoint(loss, model, file_name)
            
    def save_checkpoint(self, loss, model, file_name):
        if self.verbose:
            self.trace_func(f"loss decreased {self.loss_min:.4f} --> {loss:.4f}. Saving model...")
        torch.save(model.state_dict(), file_name)
        self.loss_min = loss
        
### Train ###
agent = DQNAgent(n_observations, n_actions)
earlystop = EarlyStopping(patience=100, verbose=True, delta=0)
scores = []
avg_losses = []

for episode in range(EPISODES):
    if earlystop.stop:
        break
    
    score = 0
    avg_loss = 0.0
    state = env.reset()
    
    for t in count():
        env.render()
        
        state = torch.FloatTensor([state])
        
        # select action
        action = agent.choose_action(state)
        # act
        next_state, reward, done, info = env.step(action.item())
        # get reward
        reward = -50 if done else reward

        # store transition 
        agent.memorize(state, action, reward, next_state)
        
        loss = agent.learn()
        state = next_state
        score += 1
        avg_loss += loss
    
        if done:
            scores.append(score)
            avg_loss = avg_loss / (t+1)
            avg_losses.append(round(avg_loss, 3))
            agent.update_target()
            earlystop(avg_loss, agent.model, f"/Users/user/Desktop/rl/best.pth")
            print(f"episode: {episode+1} | score: {score} | average loss: {avg_loss:.2f}")
            break
        
env.close()       

plt.plot(avg_losses)
plt.xlabel('episode')
plt.ylabel('loss')
plt.show()