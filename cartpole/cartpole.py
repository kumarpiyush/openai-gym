import gym
import time
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

from players import Player, RandomPlayer, DecisionTreePlayer
from interaction import Observation, Action, convert_event_sequence_to_training_set

class Constants :
    training_episodes = 2000
    testing_episodes = 200


def play_episode(env, player, show_viz=False, log_events=False) :
    logs = None
    if log_events : logs = []

    obv = env.reset()
    state = Observation(obv)
    if log_events : logs.append(state)

    total_reward = 0

    while True :
        action = player.play(env, state)
        obv, reward, done, info = env.step(action)
        state = Observation(obv)
        total_reward += reward

        if log_events :
            logs.append(Action(action))
            logs.append(state)

        if done : break

        if show_viz :
            env.render()
            #time.sleep(.1)

    return total_reward, logs

env = gym.make("CartPole-v0")


train_logs = []
random_player = RandomPlayer()

total_reward = 0.0
for ep in range(Constants.training_episodes) :
    reward, event_log = play_episode(env, random_player, log_events=True)
    train_logs.extend(event_log)
    total_reward += reward

logging.info("Mean reward with random player: {}".format(total_reward/Constants.training_episodes))

training_data, training_labels = convert_event_sequence_to_training_set(train_logs)
logging.info("Training set size: {}".format(len(training_data)))

dt_player = DecisionTreePlayer()
dt_player.fit(training_data, training_labels)
logging.info("Trained DT player")

total_reward = 0.0
for ep in range(Constants.testing_episodes) :
    reward, event_log = play_episode(env, dt_player)
    total_reward += reward

logging.info("Mean reward with decision tree player: {}".format(total_reward/Constants.training_episodes))

env.close()
