import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.distributions import Categorical

from replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNNEncoder(nn.Module):
    def __init__(self, obs_dim, action_dim, memory_dim):
        super(RNNEncoder, self).__init__()

        self.action_dim = action_dim

        self.oa_dim = obs_dim + action_dim

        self.memory_dim = memory_dim

        self.embed = nn.Linear(in_features=self.oa_dim, out_features=memory_dim)

        self.rnn_cell = nn.GRUCell(input_size=memory_dim, hidden_size=memory_dim, bias=True)

    def forward(self, obs, action, memory):
        embed_action = F.one_hot(action, self.action_dim)

        oa = torch.cat([obs, embed_action], 1)

        memory = self.rnn_cell(self.embed(oa), memory)

        return memory

    def get_initial_memory(self, n_batch=1, trainable=False):
        if trainable:
            return Parameter(torch.randn((n_batch, self.memory_dim), dtype=torch.float32), requires_grad=True)

        return torch.randn((n_batch, self.memory_dim), dtype=torch.float32)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim

        self.l1 = nn.Linear(state_dim, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(100, action_dim)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        dist = Categorical(logits=self.l3(a))
        return dist

    def loq_prob(self, state, action):
        dist = self.forward(state)

        log_prob = dist.log_prob(action)

        return log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(100, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 100)
        self.l5 = nn.Linear(100, 100)
        self.l6 = nn.Linear(100, 1)

    def forward(self, state, action):
        embed_action = F.one_hot(action, self.action_dim)

        sa = torch.cat([state, embed_action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        embed_action = F.one_hot(action, self.action_dim)

        sa = torch.cat([state, embed_action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class RNNAgent(object):

    def __init__(self,
                 obs_dim,
                 action_dim,
                 memory_dim,
                 discount=0.99,
                 tau=0.01,
                 lr=1e-6,
                 ):

        self.encoder = RNNEncoder(obs_dim=obs_dim, action_dim=action_dim, memory_dim=memory_dim).to(device)

        self.actor = Actor(state_dim=memory_dim, action_dim=action_dim).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim=memory_dim, action_dim=action_dim).to(device)

        self.critic_target = copy.deepcopy(self.critic)

        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3 * lr)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.mem_dim = memory_dim

        self.discount = discount
        self.tau = tau
        self.lr = lr

        self.total_it = 0

        self.memory = None

    def reset(self, trainable=False):
        self.memory = self.encoder.get_initial_memory(trainable=trainable)

        # TODO: fix this. ugly.
        self.prev_action = torch.tensor([np.random.randint(self.action_dim)])

    def select_action(self, obs):
        assert self.memory is not None, "Agent should be reset before run."

        obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)

        self.memory = self.encoder(obs, self.prev_action, self.memory)

        prob = self.actor(self.memory)

        # print(f"prob: {prob.probs.detach().numpy()}")

        action = prob.sample() if np.random.rand() > 0.1 else np.random.randint(self.action_dim)

        self.prev_action = torch.tensor([action])

        return action

    def train(self, replay_buffer: ReplayBuffer, n_traj=64, length=5):
        obss, actions, rewards, not_dones = replay_buffer.sample(n_traj=n_traj, length=length)

        # update critic
        # TODO make multi-step Q
        n_step = length

        # init_state = self.encoder.get_initial_memory(n_batch=n_traj, trainable=True)
        # init_state_opt = torch.optim.Adam([init_state], lr=self.lr)
        #
        # state = self.encoder(obss[:, 0], actions[:, 0], init_state)
        #
        # current_q1, current_q2 = self.critic(state, actions[:, 0])
        #
        # with torch.no_grad():
        #
        #
        #     for t in range(n_step - 1):
        #         state = self.encoder(obss[:, t + 1], actions[:, t + 1], state)
        #
        #     reward_sum = torch.zeros(n_traj, 1)
        #     for t in reversed(range(n_step)):
        #         reward_sum = self.discount * not_dones[:, t] * reward_sum + rewards[:, t]
        #
        #     target_q1, target_q2 = self.critic_target(state, actions[:, n_step - 1])
        #
        #     target_q = torch.min(target_q1, target_q2)
        #
        #     # TODO correct multi-step target
        #     target = reward_sum + not_dones.prod(axis=1) * (self.discount ** n_step) * target_q
        #
        # critic_loss = F.mse_loss(current_q1, target)/2. + F.mse_loss(current_q2, target)/2.
        #
        # self.critic_opt.zero_grad()
        # init_state_opt.zero_grad()
        #
        # critic_loss.backward()
        #
        # nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        # nn.utils.clip_grad_norm_(init_state, 0.5)
        #
        # self.critic_opt.step()
        # init_state_opt.step()

        # print(f"critic loss: {critic_loss.item()}")

        # Update policy
        log_prob = torch.zeros(n_traj)

        with torch.no_grad():
            not_done = torch.ones_like(not_dones)
            for t in range(n_step - 1):
                not_done[:, t + 1] = not_done[:, t] * not_dones[:, t]

        init_state = self.encoder.get_initial_memory(n_batch=n_traj, trainable=False)
        # init_state_opt = torch.optim.Adam([init_state], lr=self.lr)

        state = init_state
        for t in range(2):
            state = self.encoder(obss[:, t], actions[:, t], state)

            log_prob += not_done[:, t].reshape(n_traj) * self.actor.loq_prob(state, actions[:, t])

        log_prob /= 2.

        # Get importance weights
        with torch.no_grad():
            for t in range(n_step - 1):
                 state = self.encoder(obss[:, t + 1], actions[:, t + 1], state)

            q = self.critic.Q1(state, actions[:, n_step - 1])

            reward_sum = torch.zeros(n_traj, 1)
            for t in reversed(range(n_step)):
                reward_sum = self.discount * not_dones[:, t] * reward_sum + rewards[:, t]

            reward_sum += self.discount ** n_step * not_dones[:, -1] * q

            # TODO correct multi-step weight
            w = torch.tanh(reward_sum)

        actor_loss = torch.mean(-w * log_prob)

        # print(f"actor loss: {actor_loss.item()}")

        self.actor_opt.zero_grad()
        #init_state_opt.zero_grad()

        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        #nn.utils.clip_grad_norm_(init_state, 0.5)

        self.actor_opt.step()
        # init_state_opt.step()


        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # print(f"critic loss: {critic_loss.item(): .3f}, actor loss: {actor_loss.item(): .3f}, entropy: {-log_prob.mean().item()}")


if __name__ == '__main__':
    policy = RNNAgent(
        obs_dim=3,
        action_dim=4,
        memory_dim=3,
    )

    rb = ReplayBuffer(obs_dim=3, action_dim=4, max_size=50)
    for t in range(50):
        obs = np.random.rand(3)
        action = np.random.randint(4)
        reward = np.random.rand()
        done = np.random.rand() < 0.05

        rb.add(obs, action, reward, done)

    policy.train(rb)
