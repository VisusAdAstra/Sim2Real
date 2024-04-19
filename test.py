import gymnasium as gym
import numpy as np
import roboverse

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG, HerReplayBuffer
from sb3_contrib import TQC
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback


model = TQC.load("data/tqc2/tqc_model_135000_steps")
env = roboverse.make("Widow250PickPlace-v2",
                         gui=True,
                         observation_mode="pixels",
                         transpose_image=False)
model.set_env(env)
env = model.get_env()

obs = env.reset()
print("start render")
for i in range(int(1e4)):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")
env.close()