''' python version==3.10.8
    gym version==0.17.3'''
    
import gym
from time import sleep

env = gym.make('CartPole-v1')

for i in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        sleep(0.03)
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print(f"{i}th episode finished after {t} timesteps")
            break

env.close()