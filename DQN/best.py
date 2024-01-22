import gym
import torch
import torch.nn as nn
import torch.optim as potim
import torch.nn.functional as F
from itertools import count

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
    
best_model = DQN(4, 2)
best_model.load_state_dict(torch.load("/Users/user/Desktop/rl/best.pth"))
env = gym.make("CartPole-v1")

for i in range(5):
    state = env.reset()
    score = 0
    
    for t in count():
        env.render()
        
        state = torch.FloatTensor([state])
        action = best_model(state).data.max(1)[1].view(1, 1)
        next_state, reward, done, info = env.step(action.item())
        
        state = next_state
        score += 1
        
        if done:
            print(f"score: {score}")
            break
        
env.close()