import gym
import numpy as np
import torch
from torch.distributions import Categorical
from REINFORCE import REINFORCE

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    policy = REINFORCE(n_states, n_actions)
    policy.load_state_dict(torch.load("/Users/user/Desktop/rl/REINFORCE/REINFORCE_best.pth"))
    
    for i in range(5):
        state = env.reset()
        score = 0
        while True:
            env.render()
            state = torch.from_numpy(state).float().view(1, -1)
            probs = policy(state)
            probs = probs.detach()
            m = Categorical(probs)
            action = m.sample()
            next_state, reward, done, info = env.step(action.item())
            state = next_state
            score += 1
            if done:
                print(f"score : {score}")
                break