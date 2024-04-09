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
    num_attempts = 0
    accept_trajectory_key = target
    noise = 0.1
    EPSILON = 0.1

    while num_saved < num_trajectories:
        num_attempts += 1
        num_steps = 1e6
        rewards = []
        env.reset()
        policy.reset()
        time.sleep(0.1)
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
            if num_steps <1e3:
                reward = 0
            rewards.append(reward)
            model.replay_buffer.add(observation, next_observation, action, reward, np.array([done]), [{}])

            if info[accept_trajectory_key] and num_steps > 1e3:
                num_steps = j

            if info[accept_trajectory_key] and j > 16:
                break
            if done or agent_info['done']:
                break

        if info[accept_trajectory_key]:
            PRINT = False
            if PRINT:
                print("num_timesteps: ", num_steps, rewards)
                print(observation["image"].shape)
                print(next_observation["image"].shape)
            num_success += 1
            num_saved += 1
        if num_saved%100 == 0:
            print(f"num_trajectories: {num_saved} success rate: {num_success/num_attempts} Reward: {sum(rewards)}")

    print("success rate: {}".format(num_success / (num_attempts)))


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


env = roboverse.make("Widow250PickPlace-v1",
                         gui=False,
                         observation_mode="pixels",
                         transpose_image=False)
#env = TimeFeatureWrapper(env)
#env = DummyVecEnv([make_env("Widow250PickPlace-v1", i) for i in range(4)])
obs = env.reset()

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path="./data/tqc/",
  name_prefix="tqc_model",
  save_replay_buffer=False,
  save_vecnormalize=False,
)

model = TQC(env=env, batch_size=2048, buffer_size=200000, gamma=0.95, learning_rate=0.001, policy='MultiInputPolicy',
             policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
             replay_buffer_class=HerReplayBuffer,
             replay_buffer_kwargs=dict(goal_selection_strategy='future', n_sampled_goal=4),
             tau=0.05, learning_starts=0, verbose=1)

COLLECT=True
if COLLECT:
    collect_data(env, model, "grasp", "grasp_success_target", 5000, 30)
    model.save_replay_buffer(f"data/tqc_expert_grasp")
else:
    print("load_replay_buffer")
    model.load_replay_buffer(f"data/tqc_expert_grasp")

# print("start pre-training from buffer only")
# model.learn(total_timesteps=0, log_interval=5, tb_log_name="exp", reset_num_timesteps = False, progress_bar=True)
# model.train(gradient_steps=10000)

print("start learning")
model.learn(total_timesteps=20000, callback=checkpoint_callback, log_interval=5, tb_log_name="exp", reset_num_timesteps = False, progress_bar=True)
model.save("data/tqc")
model.save_replay_buffer(f"data/tqc_expert_grasp{i+1}")

print("finish learning")