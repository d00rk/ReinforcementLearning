import gym
import torch
from torch.distributions import Categorical
from A2C import A2C

env_name = "CartPole-v1"

if __name__ == "__main__":
    env = gym.make(env_name)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    A2Cnet = A2C(n_states, n_actions)
    A2Cnet.load_state_dict(torch.load("/Users/user/Desktop/rl/A2C/A2C_best.pth"))

    for i in range(5):
        state = env.reset()
        score = 0
        while True:
            env.render()
            state = torch.from_numpy(state).float().view(1, -1)
            probs, value = A2Cnet(state)
            probs.detach()
            value.detach()
            probs = Categorical(probs)
            action = probs.sample()
            next_state, reward, done, info = env.step(action.item())
            state = next_state
            score += 1
            if done:
                print(f"score : {score}")
                break