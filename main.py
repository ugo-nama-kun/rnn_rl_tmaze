import copy

import numpy as np
import torch

from agent import RNNAgent
from replay_buffer import ReplayBuffer
from t_maze_env import TMazeEnv

import wandb

wandb.init(project="rl_rnn", entity="ugo-nama-kun")

# Parameters

SEED = np.random.randint(100000)
LENGTH = 5
MAX_TIMESTEP = 10**6
START_TIMESTEPS = 10000
EVAL_FREQ = 1000
MAX_MEMORY = 10 ** 4
RNN_MEMORY_DIM = 100

env = TMazeEnv(length=LENGTH)

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(obs_dim, action_dim)

policy = RNNAgent(
    obs_dim=obs_dim,
    action_dim=action_dim,
    memory_dim=RNN_MEMORY_DIM,
)


def eval_policy(policy, length, step, eval_episodes=10, len_limit=500):
    eval_policy = RNNAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        memory_dim=RNN_MEMORY_DIM,
    )

    eval_policy.actor.load_state_dict(policy.actor.state_dict())
    eval_policy.encoder.load_state_dict(policy.encoder.state_dict())

    eval_env = TMazeEnv(length=length)
    eval_env.seed(SEED + 100)

    avg_reward = 0.
    avg_len = 0.

    for _ in range(eval_episodes):
        eval_policy.reset()
        obs, done = eval_env.reset(), False
        t = 0

        while not done:
            action = eval_policy.select_action(obs)

            obs, reward, done, _ = eval_env.step(action)

            avg_reward += reward
            avg_len += 1
            t += 1

            if t == len_limit:
                break

        #print("done.")

    avg_reward /= eval_episodes
    avg_len /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, length: {avg_len: .3f}")
    print("---------------------------------------")

    wandb.log({
        "average reward": avg_reward,
        "average episode len": avg_len,
    }, step=step)
    return avg_reward


# Set seeds
env.seed(SEED)
env.action_space.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

replay_buffer = ReplayBuffer(obs_dim=obs_dim, action_dim=action_dim, max_size=MAX_MEMORY)

evaluations = [eval_policy(policy=policy, length=LENGTH, step=0)]

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

        action = policy.select_action(obs)

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
        eval_score = eval_policy(policy=policy, length=LENGTH, step=t+1)
        evaluations.append(eval_score)
