# Libraries
import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

# openai gym CartPole environment
env = gym.make("CartPole-v1")

# device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPS = 0.9
EPS_DECAY = 0.999
LR = 0.0001
TAU = 0.005

# Q-Network
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 32)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, n_actions)
        
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# Replay Buffer
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
# number of actions
n_actions = env.action_space.n
# state
observation = env.reset()
# number of observations
n_observations = len(observation)

# networks
policy_net = DQN(n_observations, n_actions).to(device)  # Q-network which has to be optimized
target_net = DQN(n_observations, n_actions).to(device)  # Q-network used to calculate target
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.RMSprop(policy_net.parameters(), lr=LR)
memory = ReplayMemory(10000)

# select action with decaying-epsilon greedy
def get_action(state):
    sample = random.random()
    eps = max(0.01, EPS*EPS_DECAY)
    if sample > eps:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

# optimize Q-network
def optimize_net():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    # compute Q(s, t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    # compute Expectation of Q
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    # Huber loss
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
        
### Train ###
EPISODES = 300
    
scores = []
    
for episode in range(EPISODES):
    score = 0
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        env.render()
        
        # select action
        action = get_action(state)
        # act
        observation, reward, done, info = env.step(action.item())
        # get reward
        reward = reward if not done else -100
        reward = torch.tensor([reward], device=device)

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
        # store transition 
        memory.push(state, action, next_state, reward)
        
        state = next_state
        
        # optimize Q-network
        optimize_net()
        
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        
        # update target network
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        
        score += reward
        
        if done:
            scores.append(score)
            print(f"episode: {episode+1} | score: {score} | timesteps: {t}")
            break 
env.close()       
print('Complete')

plt.plot(scores)
plt.axis('off')
plt.xlabel('episode')
plt.ylabel('score')
plt.show()