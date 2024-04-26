import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import time
import numpy as np
from roboverse.policies import policies
from stable_baselines3.common.utils import set_random_seed


def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = roboverse.make(env_id,
                         gui=False,
                         observation_mode="pixels",
                         transpose_image=False)
        #env = TimeFeatureWrapper(env)
        #env.reset(seed=seed + rank)
        env.reset()
        return env
    set_random_seed(seed)
    return _init


def collect_data(env, model, policy, target, num_trajectories=100, num_timesteps=30):
    policy_class = policies[policy]
    policy = policy_class(env)
    num_success = 0
    num_saved = 0
    accept_trajectory_key = target
    noise = 0.1
    EPSILON = 0.1

    while num_saved < num_trajectories:
        num_saved += 1
        num_steps = 1e6
        rewards = []
        env.reset()
        policy.reset()
        time.sleep(0.1)
        success = False
        for j in range(num_timesteps):
            action, agent_info = policy.get_action()

            # In case we need to pad actions by 1 for easier realNVP modelling 
            env_action_dim = env.action_space.shape[0]
            #if env_action_dim - action.shape[0] == 1:
            #    action = np.append(action, 0)
            action += np.random.normal(scale=noise, size=(env_action_dim,))
            action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
            observation = env.get_observation()
            observation["image"] = np.transpose(observation["image"], (2, 0, 1))
            next_observation, reward, done, info = env.step(action)
            next_observation["image"] = np.transpose(next_observation["image"], (2, 0, 1))
            rewards.append(reward)
            success = sum(rewards) > 70
            model.replay_buffer.add(observation, next_observation, action, reward, np.array([done]), [{}])

            if success and num_steps > 1e3: #info[accept_trajectory_key]
                num_steps = j

            if success and j > 23: #info[accept_trajectory_key]
                break
            if done or agent_info['done']:
                break

        if success: #info[accept_trajectory_key]
            PRINT = False
            if PRINT:
                print("num_timesteps: ", num_steps, rewards)
                #print(observation["image"].shape)
                #print(next_observation["image"].shape)
            num_success += 1
        if num_saved%100 == 0:
            print(f"num_trajectories: {num_saved} success rate: {num_success/num_saved} Reward: {sum(rewards)}")

    print("success rate: {}".format(num_success / (num_saved)))


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
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback


env = roboverse.make("Widow250PickPlace-v2",
                         gui=False,
                         observation_mode="pixels",
                         transpose_image=False)
#env = TimeFeatureWrapper(env)
#env = DummyVecEnv([make_env("Widow250PickPlace-v1", i) for i in range(4)])
seed = 2
obs = env.reset()

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path=f"./data/seed_{seed}/",
  name_prefix="tqc_model",
  save_replay_buffer=False,
  save_vecnormalize=False,
)

model = TQC(env=env, batch_size=2048, buffer_size=1_000_000, gamma=0.95, learning_rate=0.001, policy='MultiInputPolicy',
             policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
             replay_buffer_class=HerReplayBuffer,
             replay_buffer_kwargs=dict(goal_selection_strategy='future', n_sampled_goal=4),
             tau=0.05, learning_starts=200, verbose=1)

#model = TQC.load("data/tqc")
#model.set_env(env)
COLLECT=False
if COLLECT:
    collect_data(env, model, "pickplace", "place_success_target", 10000, 35)
    model.save_replay_buffer(f"data/seed_{seed}/tqc_expert_pick_place")
else:
    print("load_replay_buffer")
    model.load_replay_buffer(f"data/seed_1/tqc_expert_pick_place")

print("start pre-training from buffer only")
model.learn(total_timesteps=0, callback=checkpoint_callback, log_interval=5, tb_log_name="exp", reset_num_timesteps = False, progress_bar=True)
model.train(gradient_steps=20000)

print("start learning")
model.learn(total_timesteps=500_000, callback=checkpoint_callback, log_interval=5, tb_log_name="exp", reset_num_timesteps = False, progress_bar=True)

print("load_replay_buffer")
model.load_replay_buffer(f"data/seed_1/tqc_expert_pick_place")

model.learn(total_timesteps=500_000, callback=checkpoint_callback, log_interval=5, tb_log_name="exp", reset_num_timesteps = False, progress_bar=True)
model.save(f"data/seed_{seed}/tqc_pick_place")
print("finish learning")


