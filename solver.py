import math

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset

from rl_with_attention import AttentionTSP
from rl_with_rnn import RNNTSP


class Solver(nn.Module):
    def __init__(self):
        super(Solver, self).__init__()

    def reward(self, sample_solution):
        """
        Args:
            sample_solution seq_len of [batch_size]
            torch.LongTensor [batch_size x seq_len x 2]
        """
        #여기 다시 한 번 확인

        batch_size, seq_len, _ = sample_solution.size()

        tour_len = Variable(torch.zeros([batch_size]))
        if isinstance(sample_solution, torch.cuda.FloatTensor):
            tour_len = tour_len.cuda()
        for i in range(seq_len - 1):
            tour_len += torch.norm(sample_solution[:, i, :] - sample_solution[:, i + 1, :], dim=-1)

        tour_len += torch.norm(sample_solution[:, seq_len - 1, :] - sample_solution[:, 0, :], dim=-1)

        return tour_len

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, input_size, seq_len]
        """
        probs, actions = self.actor(inputs)
        R = self.reward(inputs.gather(1, actions.unsqueeze(2).repeat(1, 1, 2)))

        return R, probs, actions


class solver_RNN(Solver):
    def __init__(self,
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration):
        super(solver_RNN, self).__init__()

        self.actor = RNNTSP(embedding_size,
                                hidden_size,
                                seq_len,
                                n_glimpses,
                                tanh_exploration)

class solver_Attention(Solver):
    def __init__(self,
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration):
        super(solver_Attention, self).__init__()

        self.actor = AttentionTSP(embedding_size,
                                  hidden_size,
                                  seq_len)
