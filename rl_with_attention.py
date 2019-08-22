from math import sqrt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical

class Glimpse(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_head):
        super(Glimpse, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.single_dim = hidden_size // n_head
        self.c_div = 1.0 / sqrt(single_dim)
        # Why should we take bias?

        self.W_q = nn.Linear(self.input_size, self.hidden_size)
        self.W_k = nn.Linear(self.input_size, self.hidden_size)
        self.W_v = nn.Linear(self.input_size, self.hidden_size)
        self.W_out = nn.Linear(self.hidden_size, self.input_size)

        # No dropout or No Batch/Layernorm as mentioned at Wouter's paper

    def forward(self, query, target, mask=None):
        """
        Parameters
        ----------
        query : FloatTensor with shape [batch_size x input_size]
        target : FloatTensor with shape [batch_size x seq_len x input_size]
        mask : BoolTensor with shape [batch_size x input_size]
        if any
        """
        batch_size, seq_len, _ = target.shape

        q_c = self.W_q(query).reshape(batch_size, self.n_head, self.single_dim)
        k = self.W_k(target).reshape(batch_size, seq_len, self.n_head, self.single_dim).permute(0, 2, 1, 3).contiguous()
        v = self.W_v(target).reshape(batch_size, seq_len, self.n_head, self.single_dim).permute(0, 2, 1, 3).contiguous()
        qk = torch.einsum("ijl,ijkl->ijk", [q_c, k]) * self.c_div

        if mask is not None:
            _mask = mask.unsqueeze(1).repeat(1, self.n_head, 1)
            qk[_mask] = -100000.0

        alpha = torch.softmax(qk, -1)
        print(alpha.shape, v.shape)
        h = torch.einsum("ijk,ijkl->ijl", alpha, v)

        if n_head == 1:
            ret = h.reshape(batch_size, -1)
            return alpha.squeeze(1), ret
        else:
            ret = self.W_out(h.reshape(batch_size, -1))
            return alpha, ret


class AttentionNet(nn.Module):
    pass

class Solver():
    pass

if __name__ == "__main__":
    # Glimpse Test
    input_size = 16
    batch_size = 13
    target_size = 17
    hidden_size = 32
    n_head = 4

    single_dim = hidden_size // n_head
    query = torch.FloatTensor(batch_size, input_size).uniform_(0, 1)
    target = torch.FloatTensor(batch_size, target_size, input_size).uniform_(0, 1)
    mask = None
    g = Glimpse(input_size, hidden_size, n_head)
    _, ret = g(query, target, mask)
    print(ret.shape)

    class MultiHeadAttention(nn.Module):
        def __init__(self, heads, d_model):
            super().__init__()

            self.d_model = d_model
            self.d_k = d_model // heads
            self.h = heads

            self.q_linear = nn.Linear(d_model, d_model)
            self.v_linear = nn.Linear(d_model, d_model)
            self.k_linear = nn.Linear(d_model, d_model)
            self.out = nn.Linear(d_model, d_model)

        def forward(self, q, k, v, mask=None):

            bs = q.size(0)

            # perform linear operation and split into h heads

            k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
            v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

            # transpose to get dimensions bs * h * sl * d_model

            k = k.transpose(1,2)
            q = q.transpose(1,2)
            v = v.transpose(1,2)
            # calculate attention using function we will define next
            softmax, scores = attention(q, k, v, self.d_k, mask)

            # concatenate heads and put through final linear layer
            concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)

            output = softmax, self.out(concat)

            return output

    def attention(q, k, v, d_k, mask=None):

        scores = torch.matmul(q, k.transpose(-2, -1)) /  sqrt(d_k)
        scores = F.softmax(scores, dim=-1)


        output = torch.matmul(scores, v)
        return scores, output

    input_size = 16
    batch_size = 3
    target_size = 16
    hidden_size = 16
    n_head = 2
    single_dim = hidden_size // n_head
    query = torch.FloatTensor(batch_size, input_size).uniform_(0, 1)
    target = torch.FloatTensor(batch_size, target_size, input_size).uniform_(0, 1)
    g = Glimpse(input_size, hidden_size, n_head)

    g_sft, g_ret = g(query, target)

    mh = MultiHeadAttention(2, 16)#heads, d_model
    mh.q_linear = g.W_q
    mh.v_linear = g.W_v
    mh.k_linear = g.W_k
    mh.out = g.W_out
    mh_sft, mh_ret = mh(query.unsqueeze(1), target, target)
    mh_sft = mh_sft.squeeze(2)
    mh_ret = mh_ret.squeeze(1)
    print("diff between softmaxes", torch.sum(torch.abs(mh_sft - g_sft)))
    print("diff between art", torch.sum(torch.abs(mh_ret - g_ret)))
    print("둘의 차이가 충분히 작으면 구현이 잘 된 것...! >_<")