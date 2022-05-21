
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, max_size):
        self.max_size = max_size

        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((max_size, obs_dim))
        self.action = np.zeros((max_size), dtype=int)
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, obs, action, reward, done):
        # (o, ) -> a -> (o', r, done)

        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, n_traj: int, length: int):
        ind = np.random.randint(0, self.size - length, size=n_traj)

        obss = np.array([self.obs[i:i+length+1] for i in ind])
        obss = torch.FloatTensor(obss).to(device)

        actions = np.array([self.action[i:i+length+1] for i in ind], dtype=np.int8)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)

        rewards = np.array([self.reward[i:i+length] for i in ind])
        rewards = torch.FloatTensor(rewards).to(device)

        not_dones = np.array([self.not_done[i:i+length] for i in ind])
        not_dones = torch.FloatTensor(not_dones).to(device)

        return obss, actions, rewards, not_dones


if __name__ == '__main__':
    rb = ReplayBuffer(obs_dim=2, action_dim=1, max_size=20)

    for t in range(50):
        obs = np.random.rand(2)
        action = np.random.randint(4)
        reward = np.random.rand()
        done = np.random.rand() < 0.05

        rb.add(obs, action, reward, done)

        if t > 10:
            traj = rb.sample(n_traj=3, length=5)
            for sample in traj:
                print(sample)

            print("---")
