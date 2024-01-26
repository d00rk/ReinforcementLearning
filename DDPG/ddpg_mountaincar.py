# Libraries
import gym
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# environment
env = gym.make("MountainCarContinuous-v0")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
max_action = env.action_space.high[0]

# Hyperparameters
EPISODES = 1000
BATCH_SIZE = 128
CRITIC_LR = 0.0001
ACTOR_LR = 0.001
GAMMA = 0.99
TAU = 0.1

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
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0.001)
        
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
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0.001)
        
    def forward(self, s, a):
        out = F.leaky_relu(self.fc1(torch.cat([s, a], 1)))
        out = F.leaky_relu(self.fc2(out))
        out = self.fc3(out)
        return out

# exploration noise
class OrnsteinUhlenbeckNoise:
    def __init__(self, theta=0.05, mu=0, sigma=0.25, dim=1):
        self.theta = theta
        self.mu = mu * np.ones(dim)
        self.sigma = sigma
        self.dim = dim
        
    def reset(self):
        self.state = copy.copy(self.mu)
        
    def __call__(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self. state = x + dx
        return self.state

# Agent
class DDPGAgent:
    def __init__(self, n_states, n_actions, low_bound, high_bound):
        self.n_states = n_states
        self.n_actions = n_actions
        self.noise = OrnsteinUhlenbeckNoise(dim=self.n_actions) # exploration noise
        self.action_low = low_bound
        self.action_high = high_bound
        
        self.memory = deque(maxlen=5000)    # replay buffer
        
        self.actor = Actor(n_states, n_actions, max_action) # mu
        self.target_actor = Actor(n_states, n_actions, max_action)  # mu'
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=ACTOR_LR)
        
        self.critic = Critic(n_states, n_actions)   # Q
        self.target_critic = Critic(n_states, n_actions)    # Q'
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        
        # initialize actor network, critic network
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
    
    # select action    
    def choose_action(self, states):
        action = self.actor(states)    
        noise = self.noise()
        action = action.detach()
        action = action + noise
        action = action.type(torch.FloatTensor)
        return action
    
    # store transition
    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, torch.FloatTensor(np.array([reward])), torch.FloatTensor(np.array([next_state])), done))
    
    # train network
    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return np.Inf, np.Inf
        
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, done = zip(*batch)

        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards).view(BATCH_SIZE, 1)
        next_states = torch.cat(next_states)
        done = torch.FloatTensor(done).view(BATCH_SIZE, 1)

        with torch.no_grad():
            actor_target = self.target_actor(next_states)
            critic_target = rewards + GAMMA * (1 - done) * self.target_critic(next_states, actor_target)
        
        critic_q = self.critic(states, actions)
        
        # train critic network
        critic_loss = F.mse_loss(critic_target, critic_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # train actor network
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return critic_loss, actor_loss
        
    # soft update
    def update_target(self):
        actor_state_dict = self.actor.state_dict()
        target_actor_state_dict = self.target_actor.state_dict()
        critic_state_dict = self.critic.state_dict()
        target_critic_state_dict = self.target_critic.state_dict()
        
        for key in actor_state_dict:
            actor_state_dict[key] = TAU *  actor_state_dict[key] + (1 - TAU) * target_actor_state_dict[key]

        for key in critic_state_dict:
            critic_state_dict[key] = TAU * critic_state_dict[key] + (1 - TAU) * target_critic_state_dict[key]
        
        self.target_actor.load_state_dict(actor_state_dict)
        self.target_critic.load_state_dict(critic_state_dict)
        
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.trace_func = trace_func
        self.counter = 0
        self.max_reward = None
        self.stop = False
        
    def __call__(self, reward, actor, critic, actor_name, critic_name):
        if self.max_reward is None:
            self.max_reward = reward
            self.save_checkpoint(reward, actor, critic, actor_name, critic_name)
        elif self.max_reward + self.delta > reward:
            self.counter += 1
            self.trace_func(f"Early Stopping Counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.counter = 0
            self.save_checkpoint(reward, actor, critic, actor_name, critic_name)
            
    def save_checkpoint(self, reward, actor, critic, actor_name, critic_name):
        if self.verbose:
            self.trace_func(f"reward increased {self.max_reward:.3f} --> {reward:.3f}. Saving model...")
        torch.save(actor.state_dict(), actor_name)
        torch.save(critic.state_dict(), critic_name)
        self.max_reward = reward
            
            
agent = DDPGAgent(n_states, n_actions, -max_action, max_action)
es = EarlyStopping(patience=10000, verbose=True, delta=0)
scores = []
times = []
sum_of_time = 0

for episode in range(EPISODES):
    score = 0
    time = 0
    state = env.reset()
    agent.noise.reset()
    while True:
        env.render()
        state = torch.FloatTensor(state).view(1, 2)
        action = agent.choose_action(state)
        next_states, rewards, done, info = env.step(action)
        time += 1
        sum_of_time += 1
        car_pos, car_vel = next_states[0], next_states[1]
        rewards = 100 if car_pos >= 0.45 else rewards
        
        agent.memorize(state, action, rewards, next_states, done)
        
        critic_loss, actor_loss = agent.learn()
        
        state = next_states
        score += rewards
        
        writer.add_scalar("Critic_Loss/train", critic_loss, sum_of_time)
        writer.add_scalar("Actor_Loss/train", actor_loss, sum_of_time)

        if done:
            agent.update_target()
            scores.append(score)
            times.append(time)
            pre_time = time
            print(f"episode : {episode+1} | score : {score:.4f} | time : {time}")
            es(score, agent.actor, agent.critic, "/Users/user/Desktop/rl/DDPG/ddpg_best_actor.pth", "/Users/user/Desktop/rl/DDPG/ddpg_best_critic.pth")
            break      
env.close()
writer.close()

plt.subplot(1, 2, 1)
plt.plot(scores)
plt.xlabel("episode")
plt.ylabel("score")

plt.subplot(1, 2, 2)
plt.plot(times)
plt.xlabel("episode")
plt.ylabel("time")

plt.show()