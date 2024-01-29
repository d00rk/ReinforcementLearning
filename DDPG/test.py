import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# mu network
class Actor(nn.Module):
    def __init__(self, n_states, n_actions, max_action):
        super(Actor, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.max_action = max_action
        self.fc1 = nn.Linear(n_states, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, n_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.max_action
        return x

# Q network
class Critic(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Critic, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.fc1 = nn.Linear(n_states + n_actions, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, n_actions)
        
    def forward(self, s, a):
        out = F.leaky_relu(self.fc1(torch.cat([s, a], 1)))
        out = F.leaky_relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
# environment
env = gym.make('MountainCarContinuous-v0')
n_states = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
max_action = env.action_space.high[0]

best_actor = Actor(n_states, n_actions, max_action)
best_actor.load_state_dict(torch.load("/Users/user/Desktop/rl/DDPG/ddpg_best_actor.pth"))

best_critic = Critic(n_states, n_actions)
best_critic.load_state_dict(torch.load("/Users/user/Desktop/rl/DDPG/ddpg_best_critic.pth"))

for i in range(5):
    score = 0
    time = 0
    state = env.reset()
    
    while True:
        env.render()
        state = state.astype(np.float64)
        state = torch.FloatTensor(state).view(1, 2)
        action = best_actor(state)
        action = action.detach().cpu()
        next_states, rewards, done, info = env.step(action)
        time += 1
        car_pos, car_vel = next_states[0], next_states[1]
        rewards = 100 if car_pos >= 0.45 else rewards
        
        state = next_states
        score += rewards
        
        if done:
            print(f"score : {score} | time: {time}")
            break
        
env.close()