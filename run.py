import numpy as np
import gymnasium as gym
from bee import BeeWorld
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

gym.register(
    id="BeeWorld",
    entry_point=BeeWorld,
    max_episode_steps=1000,
)

# mode = "train"
mode = "run"

if mode == "train":
    # env = gym.make("BeeWorld", render_mode="rgb_array")
    env = make_vec_env(
        "BeeWorld",
        n_envs=8,
        env_kwargs={"render_mode": "rgb_array"},
    )
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        # learning_rate=1e-3,
    )
    model.learn(total_timesteps=1_000_000, progress_bar=True)
    model.save("bee.model")

if mode == "run":
    model = PPO.load("bee.model")
    env = gym.make("BeeWorld", render_mode="human")

    obs, info = env.reset()

    done = False
    while not done:
        action, _s = model.predict(obs)
        obs, _, _, done, _ = env.step(action)
        print(obs)

    env.close()

# env.step((0.0, 0.0))
# while not done:
#    A = np.random.uniform(-0.5, 0.5)
#    theta = np.random.uniform(-0.5, 0.5)
#    observation, reward, terminated, done, info = env.step((A, theta))
#
# env.close()
