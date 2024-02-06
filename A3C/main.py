import gym
import torch
import torch.nn as nn
import torch.utils as utils
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical

import numpy as np
import matplotlib.pyplot as plt
import time

from A3C import ActorCritic

# Hyperparameters
env_name = "LunarLander-v2"
num_processes = 4
seed = 42
MAX_EPISODES = 2000
MAX_STEPS = 20000
LR = 0.001
GAMMA = 0.95
ACTION_RATIO = 0.2

def train(rank, global_actor_critic):
    env = gym.make(env_name)
    
    env.seed(seed+rank)
    np.random.seed(seed+rank)
    torch.manual_seed(seed+rank)
    
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    
    local_actor_critic = ActorCritic(n_state, n_action)
    
    optimizer = optim.Adam(global_actor_critic.parameters(), lr=LR)
    
    episode = 0
        
    while True:
        local_actor_critic.load_state_dict(global_actor_critic.state_dict())
        
        state = env.reset()
        done = False
        step = 1
        batch = []
        
        state = torch.from_numpy(state).float().view(1, -1)
        
        while not done:
            #env.render()
            
            probs, value = local_actor_critic(state)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            next_state, reward, done, _ = env.step(action.item())
            reward = 0.0 if done else reward
            next_state = torch.from_numpy(next_state).float().view(1, -1)
            
            step += 1
            
            if step > MAX_STEPS:
                done = True
                reward = value
                
            batch.append([reward, log_prob, value])
            state = next_state
            
        R = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        epi_reward = 0.0
        for i in reversed(range(len(batch))):
            epi_reward += batch[i][0]
            R = GAMMA * R + batch[i][0]
            advantage = R - batch[i][2]
            value_loss = value_loss + advantage ** 2
            policy_loss = policy_loss - batch[i][1] * advantage
            
        optimizer.zero_grad()
        loss = ACTION_RATIO * policy_loss + value_loss    
        loss.backward()
        
        nn.utils.clip_grad_norm_(local_actor_critic.parameters(), 50)
        
        for local_param, global_param in zip(local_actor_critic.parameters(), global_actor_critic.parameters()):
            global_param.grad = local_param.grad
            
        optimizer.step()
        
        local_actor_critic.load_state_dict(global_actor_critic.state_dict())
        
        print(f"process {rank} | episode: {episode} | reward {epi_reward:.4f}")
        
        episode += 1
        
        if episode > MAX_EPISODES:
            break
        
    print(f"process {rank} finished.")
    env.close()
    
    
if __name__ == "__main__":
    mp.set_start_method("spawn")
    
    env = gym.make(env_name)
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n

    model = ActorCritic(n_state, n_action)
    model.share_memory()
    
    env.close()
    
    processes = []
    for i in range(num_processes):
        p = mp.Process(target=train, args=(i, model, ))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()