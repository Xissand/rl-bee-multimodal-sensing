import numpy as np
import gymnasium as gym
from bee import BeeWorld
import torch.nn as nn

from stable_baselines3 import TD3
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)


gym.register(
    id="BeeWorld",
    entry_point=BeeWorld,
    max_episode_steps=1000,
)

env = gym.make("BeeWorld", render_mode="human", max_episode_steps=1000)
env.reset()

model = TD3.load("test2.zip")

obs, _ = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, term, dones, info = env.step(action)
    if dones:
        obs, _ = env.reset()

# env.close()
# model.save("test2")
