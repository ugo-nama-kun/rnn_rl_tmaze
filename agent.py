import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNNEncoder(nn.Module):
    def __init__(self, obs_dim, memory_dim):
        super(RNNEncoder, self).__init__()

        self.obs_dim = obs_dim

        self.memory_dim = memory_dim

        self.embed = nn.Linear(in_features=obs_dim, out_features=memory_dim)

        self.rnn_cell = nn.GRUCell(input_size=memory_dim, hidden_size=memory_dim, bias=True)

    def forward(self, obs, memory):
        memory = self.rnn_cell(self.embed(obs), memory)

        return memory

    def get_initial_memory(self, trainable=False):
        if trainable:
            return Parameter(torch.zeros((1, self.memory_dim), dtype=torch.float32), requires_grad=True)

        return torch.zeros((1, self.memory_dim), dtype=torch.float32)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.action_dim = action_dim

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        p = self.softmax(self.l3(a))
        return p


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

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
                 tau=0.005
                 ):

        self.encoder = RNNEncoder(obs_dim=obs_dim, memory_dim=memory_dim).to(device)

        self.actor = Actor(state_dim=memory_dim, action_dim=action_dim).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim=memory_dim, action_dim=action_dim).to(device)

        self.critic_target = copy.deepcopy(self.critic)

        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau

        self.total_it = 0

        self.memory = None

    def reset(self, trainable=False):
        self.memory = self.encoder.get_initial_memory(trainable)

    def select_action(self, obs):
        assert self.memory is not None, "Agent should be reset before run."

        obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)

        self.memory = self.encoder(obs, self.memory)

        prob = self.actor(self.memory).detach().cpu().data.numpy().flatten()

        return np.random.choice(self.action_dim, 1, replace=False, p=prob)

    def train(self, replay_buffer: ReplayBuffer):
        traj_list = replay_buffer.sample(n_traj=3, length=5)

        # TODO: Add Reinforce-based training rule





