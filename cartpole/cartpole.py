import gym
import time
import numpy as np

SHOW_VIS = False
TOTAL_EPISODES = 200
ANG_MAR = 0.02

def play_episode(env) :
    obv = env.reset()
    cart_pos, pole_angle = obv[0], obv[2]

    total_reward = 0

    while True :
        if pole_angle < -ANG_MAR : action = 0
        elif pole_angle > 0 and pole_angle < ANG_MAR : action = 0
        else : action = 1

        obv, reward, done, info = env.step(action)
        cart_pos, pole_angle = obv[0], obv[2]
        total_reward += reward

        # env.render()

        if done : break

        # time.sleep(.1)

    return total_reward

env = gym.make("CartPole-v0")

rewards = []

for ep in range(TOTAL_EPISODES) :
    total_reward = play_episode(env)
    rewards.append(total_reward)

print(f"Mean Reward: {np.mean(rewards)}")

env.close()
