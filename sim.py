import numpy as np
import roboverse
from roboverse.policies import policies


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
        num_steps = -1
        rewards = []
        env.reset()
        policy.reset()
        for j in range(num_timesteps):
            action, agent_info = policy.get_action()

            # In case we need to pad actions by 1 for easier realNVP modelling 
            env_action_dim = env.action_space.shape[0]
            #if env_action_dim - action.shape[0] == 1:
            #    action = np.append(action, 0)
            action += np.random.normal(scale=noise, size=(env_action_dim,))
            action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
            observation = env.get_observation()
            next_observation, reward, done, info = env.step(action)
            if not info[accept_trajectory_key]:
                reward += 0.99**(num_timesteps-j)/10
            rewards.append(reward)
            model.replay_buffer.add(observation, next_observation, action, reward, done, [{}])

            if info[accept_trajectory_key] and num_steps < 0:
                num_steps = j

            if info[accept_trajectory_key] and j > 20:
                break
            if done or agent_info['done']:
                break

        if info[accept_trajectory_key]:
            if True:
                print("num_timesteps: ", num_steps)
                #print(traj["observations"])
            num_success += 1
            num_saved += 1
        print(f"num_trajectories: {num_saved} success rate: {num_success/num_attempts} Reward: {sum(rewards)}")

    print("success rate: {}".format(num_success / (num_attempts)))


import gymnasium as gym
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


#env = gym.make("Pendulum-v1", render_mode="rgb_array")
env = roboverse.make("Widow250PickPlace-v1",
                         gui=False,
                         transpose_image=False)
obs = env.reset()

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MultiInputPolicy", env, buffer_size=100000, verbose=1, learning_starts=0) #, action_noise=action_noise
# collect_data(env, model, "grasp", "grasp_success_target", 100, 30)
# model.save_replay_buffer("data/td3_expert_grasp1")
# collect_data(env, model, "grasp", "grasp_success_target", 100, 30)
# model.save_replay_buffer("data/td3_expert_grasp2")

print("start learning")
for i in range(4):
    model.replay_buffer.reset()
    model.load_replay_buffer(f"data/td3_expert_grasp{i%2+1}")
    model.learn(total_timesteps=2000, log_interval=10, progress_bar=True)
print("finish learning")
model.save("data/td3")
vec_env = model.get_env()

#del model # remove to demonstrate saving and loading
#model = TD3.load("data/td3")

obs = vec_env.reset()
print("start render")
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")