import numpy as np
import gymnasium as gym
from bee import BeeWorld
import torch.nn as nn

from gymnasium.wrappers.record_video import RecordVideo

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

env = gym.make("BeeWorld", render_mode="rgb_array", max_episode_steps=1000)
env = RecordVideo(env, "videos", episode_trigger=lambda x: x % 25 == 0)
env.reset()

n_actions = 2
action_noise = NormalActionNoise(mean=np.array([0.0, 0.0]), sigma=np.array([0.1, 0.05]))

policy_kwargs = {
    "net_arch": [400, 300],  # Specify the number of hidden units per layer
    "activation_fn": nn.ReLU,  # Specify the activation function
}

model = TD3(
    "MultiInputPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    policy_kwargs=policy_kwargs,
    learning_rate=5e-4,
    tensorboard_log="./logs/",
)
model.learn(total_timesteps=250_000, log_interval=10)
model.save("test_pre")

# env.set_goal_size(1.0)
# env.reset()
# model.learn(total_timesteps=100_000, log_interval=10)
# model.save("test_post")
