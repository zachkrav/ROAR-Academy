## This is course material for Introduction to Modern Artificial Intelligence
## Example code: baselines_breakout.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

# To run this code, install stable_baselines: a Reinforcement Learning package 
# https://stable-baselines.readthedocs.io/en/master/guide/install.html
# 
# Further note, stable_baselines only compatible with Tensorflow v1 up to 1.15
# Do not run this code with Tensorflow 2.0 or above

import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend" # If you have AMD GPU and have installed PlaidML library
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import A2C

# There already exists an environment generator that will make and wrap atari environments correctly.
# We use 16 parallel processes
env = make_atari_env('BreakoutNoFrameskip-v4', num_env=16, seed=0)
# Stack 4 frames
env = VecFrameStack(env, n_stack=4)

model = A2C(CnnPolicy, env, lr_schedule='constant', verbose=1)

path = os.path.dirname(os.path.abspath(__file__))
model_file_name = path + '/breakout_a2c'
LOAD_PRETRAINED = False
if LOAD_PRETRAINED:
    # Load saved model
    model = A2C.load(model_file_name, lr_schedule='constant', verbose=1)
else:
    model.learn(total_timesteps=int(5e6))
    model.save(model_file_name )

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