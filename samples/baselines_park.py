## This is course material for Introduction to Modern Artificial Intelligence
## Example code: baselines_park.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

# To run this code, install two additional Python modules:
# 1. stable_baselines: a Reinforcement Learning package 
# https://stable-baselines.readthedocs.io/en/master/guide/install.html
# 2. highway_env: https://github.com/eleurent/highway-env
# 
# Further note, stable_baselines only compatible with Tensorflow v1 up to 1.15
# Do not run this code with Tensorflow 2.0 or above

import gym
import highway_env
import numpy as np

from stable_baselines import HER, SAC, DDPG, TD3
from stable_baselines.ddpg import NormalActionNoise

env = gym.make("parking-v0")

# Create 4 artificial transitions per real transition
n_sampled_goal = 4

# SAC hyperparams:
model = HER('MlpPolicy', env, SAC, n_sampled_goal=n_sampled_goal,
            goal_selection_strategy='future',
            verbose=1, buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95, batch_size=256,
            policy_kwargs=dict(layers=[256, 256, 256]))

# DDPG Hyperparams:
# NOTE: it works even without action noise
# n_actions = env.action_space.shape[0]
# noise_std = 0.2
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))
# model = HER('MlpPolicy', env, DDPG, n_sampled_goal=n_sampled_goal,
#             goal_selection_strategy='future',
#             verbose=1, buffer_size=int(1e6),
#             actor_lr=1e-3, critic_lr=1e-3, action_noise=action_noise,
#             gamma=0.95, batch_size=256,
#             policy_kwargs=dict(layers=[256, 256, 256]))

import os
path = os.path.dirname(os.path.abspath(__file__))
model_file_name = path + '/her_sac_highway'
LOAD_PRETRAINED = True
if LOAD_PRETRAINED:
  # Load saved model
  model = HER.load(model_file_name, env=env)
else:
  model.learn(int(2e5))
  model.save(model_file_name)

obs = env.reset()

# Evaluate the agent
episode_reward = 0
for _ in range(1000):
  action, _ = model.predict(obs)
  obs, reward, done, info = env.step(action)
  env.render()
  episode_reward += reward
  if done or info.get('is_success', False):
    print("Reward:", episode_reward, "Success?", info.get('is_success', False))
    episode_reward = 0.0
    obs = env.reset()