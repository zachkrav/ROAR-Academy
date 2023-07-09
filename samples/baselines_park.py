## This is course material for Introduction to Modern Artificial Intelligence
## Example code: baselines_park.py
## Author: Allen Y. Yang
##
## (c) Copyright 2023. Intelligent Racing Inc. Not permitted for commercial use

import gymnasium as gym
import highway_env
import numpy as np

from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3

env = gym.make("parking-v0")

# Create 4 artificial transitions per real transition
n_sampled_goal = 4

# SAC hyperparams:
model = SAC('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer, 
            replay_buffer_kwargs=dict(
              n_sampled_goal=n_sampled_goal,
              goal_selection_strategy='future',
            ),
            verbose=1, buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95, batch_size=256,
            policy_kwargs=dict(net_arch=[256, 256, 256])
            )

import os
path = os.path.dirname(os.path.abspath(__file__))
model_file_name = path + '/her_sac_highway'
LOAD_PRETRAINED = False
if LOAD_PRETRAINED:
  # Load saved model
  model = SAC.load(model_file_name, env=env)
else:
  model.learn(int(2e5))
  model.save(model_file_name)

obs, info = env.reset()

# Evaluate the agent
episode_reward = 0
for _ in range(1000):
  action, _ = model.predict(obs)
  obs, reward, done, truncated, info = env.step(action)
  env.render('human')
  episode_reward += reward
  if done or truncated or info.get('is_success', False):
    print("Reward:", episode_reward, "Success?", info.get('is_success', False))
    episode_reward = 0.0
    obs = env.reset()