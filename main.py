import copy

import numpy as np
import torch

from agent import RNNAgent
from replay_buffer import ReplayBuffer
from t_maze_env import TMazeEnv

# Parameters

SEED = 0
LENGTH = 5
MAX_TIMESTEP = 2000
START_TIMESTEPS = 50
EVAL_FREQ = 100
MAX_MEMORY = 10 ** 4
RNN_MEMORY_DIM = 3

env = TMazeEnv(length=LENGTH)

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(obs_dim, action_dim)

policy = RNNAgent(
    obs_dim=obs_dim,
    action_dim=action_dim,
    memory_dim=RNN_MEMORY_DIM,
)


def eval_policy(policy, length, eval_episodes=10):
    eval_policy = copy.deepcopy(policy)

    eval_env = TMazeEnv(length=length)
    eval_env.seed(SEED + 100)

    avg_reward = 0.

    for _ in range(eval_episodes):
        eval_policy.reset()
        obs, done = eval_env.reset(), False

        while not done:
            action = eval_policy.select_action(obs)

            obs, reward, done, _ = eval_env.step(action)

            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


# Set seeds
env.seed(SEED)
env.action_space.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

replay_buffer = ReplayBuffer(obs_dim=obs_dim, action_dim=action_dim, max_size=MAX_MEMORY)

evaluations = [eval_policy(policy=policy, length=LENGTH)]

episode_timesteps = 0
episode_reward = 0
episode_num = 0

policy.reset()

obs, done = env.reset(), False

for t in range(MAX_TIMESTEP):

    episode_timesteps += 1

    if t < START_TIMESTEPS:
        action = env.action_space.sample()

    else:

        # action = policy.select_action(obs)
        action = env.action_space.sample()

    next_obs, reward, done, info = env.step(action)
    # print(next_obs, reward, done, info, action)

    replay_buffer.add(obs, action, reward, done)

    obs = next_obs

    episode_reward += reward

    if t >= START_TIMESTEPS:
        policy.train(replay_buffer)

    if done:
        print(
            f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")

        # Reset
        policy.reset()
        obs, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    if (t + 1) % EVAL_FREQ == 0:
        eval_score = eval_policy(policy=policy, length=LENGTH)
        evaluations.append(eval_score)
