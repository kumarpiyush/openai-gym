import gym
import time
import random

print
env = gym.make("Pendulum-v0")
env.reset()

inl=1
while True :
    action = env.action_space.sample()
    print(action,inl)
    inl+=1
    print(env.step([random.choice([2])]))
    env.render()
    a=input()
