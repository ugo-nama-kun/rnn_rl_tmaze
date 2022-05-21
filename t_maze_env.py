from typing import Tuple

import numpy as np
import gym


"""
There are four actions. up-down-left-right
At initial state 0, the agent observes 011 (up-goal) or 110 (down-goal).
at the corridor 101 and 010 at the T-junction.
          6
0 1 2 3 4 5
          7
"""


class TMazeEnv(gym.Env):

    def __init__(self, length):
        super(TMazeEnv, self).__init__()

        self.length = length
        self.n_state = length + 2

        # State Reser
        self.switch = np.random.randint(2)  # 0 for up, 1 for bottom
        self.position = 0

        self.action_space = gym.spaces.Discrete(n=4)
        self.observation_space = gym.spaces.Box(low=np.zeros(3, dtype=np.float32),
                                                high=np.ones(3, dtype=np.float32))

    def reset(self, seed=None, **kwargs):

        super(TMazeEnv, self).reset(seed=seed)

        self.switch = np.random.randint(2)  # 0 for up, 1 for bottom
        self.position = 0

        return self.get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        action:
        0: up
        1: down
        2: left
        3: right

                  6
        0 1 2 3 4 5
                  7
        """

        reward = -0.1
        done = False

        if self.position == 0:

            if action == 2:
                self.position += 1

        elif self.position < self.length:

            if action == 2:
                self.position += 1

            elif action == 3:
                self.position -= 1

        elif self.position == self.length:

            if action == 0:
                self.position += 1

                if self.switch == 0:
                    reward = 4.0

                done = True

            elif action == 1:
                self.position += 2

                if self.switch == 1:
                    reward = 4.0

                done = True

        else:
            print(self.position, self.switch, action, done)
            raise ValueError("something went wrong!")


        info = {
            "position": self.position,
            "switch": self.switch
        }

        return self.get_obs(), reward, done, info

    def get_obs(self):
        obs = np.zeros(3, dtype=np.float32)

        if self.position == 0:

            if self.switch == 0:
                # 011
                obs[1] = 1
                obs[2] = 1

            else:
                # 110
                obs[0] = 1
                obs[1] = 1

        elif self.position == self.length:
            # 010
            obs[1] = 1

        elif self.position > self.length:
            # 111 (terminal)
            obs += 1

        else:
            # 101
            obs[0] = 1
            obs[2] = 1

        return obs

    def render(self, mode="human"):
        pass


if __name__ == '__main__':
    env = TMazeEnv(length=5)

    print(env.action_space.shape)
    print(env.observation_space.shape)

    obs = env.reset()
    done = False

    while not done:
        a = env.action_space.sample()

        obs, r, done, info = env.step(a)

        print(obs, r, done, info, a)

    print("finish.")
