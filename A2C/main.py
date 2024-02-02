import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from A2C import A2Cagent

# Hyperparameters
env_name = "CartPole-v1"
EPISODES = 2000
LR = 0.005
GAMMA = 0.95
ACTION_RATIO = 0.2

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
            self.min_loss = loss
            self.save_checkpoint(loss, model, file_name)
        elif self.min_loss < loss + self.delta:
            self.counter += 1
            self.trace_func(f"Early Stopping Counter {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.save_checkpoint(loss, model, file_name)
        
    def save_checkpoint(self, loss, model, file_name):
        if self.verbose:
            self.trace_func(f"Loss Decreased {self.min_loss:.3f} --> {loss:.3f}. Saving model...")
        torch.save(model.state_dict(), file_name)
        self.counter = 0
        self.min_loss = loss
            
writer = SummaryWriter()
es = EarlyStopping(patience=200, delta=0, verbose=True)

if __name__ == "__main__":
    env = gym.make(env_name)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = A2Cagent(n_states, n_actions, LR, GAMMA, ACTION_RATIO)
    
    for episode in range(EPISODES):
        if es.stop:
            print(f"minimum loss: {es.min_loss:.4f}")
            break
        
        state = env.reset()
        score = 0
        while True:
            env.render()
            
            state = torch.from_numpy(state).float().view(1, -1)
            
            action, action_log_prob = agent.get_action(state)
            
            next_state, reward, done, info = env.step(action.item())
            
            reward = -50 if done else reward
            
            q_value = agent.get_q_value(state)
            next_q_value = agent.get_q_value(torch.from_numpy(next_state).float().view(1, -1))
            
            advantage, y_i = agent.advantage_td_target(reward, q_value, next_q_value, done)
            
            agent.remember(state, action_log_prob, advantage, y_i)
            
            state = next_state
            score += 1
            
            if done:
                loss = agent.train()
                writer.add_scalar("Loss/train", loss, episode+1)
                writer.add_scalar("Score/train", score, episode+1)
                print(f"Episode: {episode+1} | score: {score}")
                es(loss, agent.A2Cnet, "/Users/user/Desktop/rl/A2C/A2C_best.pth")
                break
            
    env.close()
    writer.close()