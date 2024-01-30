import gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from REINFORCE import REINFORCEAgent

# Hyperparameters
LR = 0.001
GAMMA = 0.99
EPISODES = 1000

writer = SummaryWriter()
scores = []

class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=False, trace_func=print):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.trace_func = trace_func
        self.min_loss = None
        self.counter = 0
        self.stop = False
        
    def __call__(self, loss, model, file_name):
        if self.min_loss is None:
            self.save_checkpoint(loss, model, file_name)
        elif loss > self.min_loss + self.delta:
            self.counter += 1
            self.trace_func(f"Early Stopping Counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.save_checkpoint(loss, model, file_name)
            
    def save_checkpoint(self, loss, model, file_name):
        if self.verbose:
            self.trace_func(f"loss decreased {self.loss_min:.4f} --> {loss:.4f}. Saving model...")
        self.min_loss = loss
        self.counter = 0
        torch.save(model.state_dict(), file_name)
        

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    agent = REINFORCEAgent(n_states, n_actions, LR, GAMMA)  
    es = EarlyStopping(patience=1000, delta=0)
    
    for episode in range(EPISODES):
        if es.stop:
            break
        state = env.reset()
        score = 0
        while True:
            env.render()
            state = torch.from_numpy(state).float().view(1, -1)
            action, log_prob = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(reward, log_prob)
            state = next_state
            score += 1
            if done:
                break
            
        loss = agent.train()
        es(loss, agent.policy, "/Users/user/Desktop/rl/REINFORCE/REINFORCE_best.pth")
        writer.add_scalar("Loss/train", loss, episode)
        scores.append(score)
        print(f"Episode {episode+1} | score {score}")
        
env.close()
writer.close()             

plt.plot(scores)
plt.xlabel("episode")
plt.ylabel("score")
plt.show()