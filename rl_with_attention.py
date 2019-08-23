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


class GraphEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(GraphEmbedding, self).__init__()
        self.embedding_size = embedding_size

        self.embedding = nn.Linear(input_size, embedding_size)
    def forward(self, inputs):
        return self.embedding(inputs)


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
        self.c_div = 1.0 / sqrt(self.single_dim)
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
        #print(alpha.shape, v.shape)
        h = torch.einsum("ijk,ijkl->ijl", alpha, v)

        if self.n_head == 1:
            ret = h.reshape(batch_size, -1)
            return alpha.squeeze(1), ret
        else:
            ret = self.W_out(h.reshape(batch_size, -1))
            return alpha, ret

class Pointer(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_head,
        C=10):
        super(Pointer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.single_dim = hidden_size // n_head
        self.c_div = 1.0 / sqrt(self.single_dim)
        self.C = C
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
        qk = self.C * torch.tanh(qk)
        if mask is not None:
            _mask = mask.unsqueeze(1).repeat(1, self.n_head, 1)
            qk[_mask] = -100000.0

        alpha = torch.softmax(qk, dim=-1)
        #print(alpha.shape, v.shape)
        h = torch.einsum("ijk,ijkl->ijl", alpha, v)

        if self.n_head == 1:
            ret = h.reshape(batch_size, -1)
            return alpha.squeeze(1), ret
        else:
            ret = self.W_out(h.reshape(batch_size, -1))
            return alpha, ret

class skip_connection(nn.Module):
    def __init__(self, module):
        super(skip_connection, self).__init__()
        self.module = module
    def forward(x):
        return x + self.module(x)

class att_layer(nn.Module):
    def __init__(self, embed_dim, n_heads, feed_forward_hidden=512, bn=False):
        super(att_layer, self).__init__()
        self.mha = torch.nn.MultiheadAttention(embed_dim, n_heads)
        self.embed = nn.Sequential(nn.Linear(embed_dim, feed_forward_hidden), nn.ReLU(), nn.Linear(feed_forward_hidden, embed_dim))

    def forward(self, x):
        #I don't know why, but multiheadattention in pytorch starts with (target_seq_length, batch_size, embedding_size).
        # thus we permute order first. https://pytorch.org/docs/stable/nn.html#multiheadattention
        x = x.permute(1, 0, 2)
        _1 = x + self.mha(x, x, x)[0]
        _1 = _1.permute(1, 0, 2)
        _2 = _1 + self.embed(_1)
        return _1

class AttentionModule(nn.Sequential):
    def __init__(self, embed_dim, n_heads, feed_forward_hidden=512, bn=False):
        super(AttentionModule, self).__init__(
            att_layer(embed_dim, n_heads, feed_forward_hidden, bn),
            att_layer(embed_dim, n_heads, feed_forward_hidden, bn),
        )

class AttentionTSP(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 seq_len,
                 n_head=4,
                 tanh_exploration=10):
        super(AttentionTSP, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.n_head = n_head
        self.tanh_exploration = self.C = tanh_exploration

        self.embedding = GraphEmbedding(2, embedding_size)
        self.mha = AttentionModule(embedding_size, n_head)

        self.init_W = nn.Parameter(torch.Tensor(2 * self.embedding_size))
        self.init_W.data.uniform_(-1, 1)
        self.glimpse = Glimpse(self.embedding_size, self.hidden_size, self.n_head)
        self.pointer = Pointer(self.embedding_size, self.hidden_size, 1, self.tanh_exploration)

        self.h_context_embed = nn.Linear(self.embedding_size, self.embedding_size)
        self.v_weight_embed = nn.Linear(self.embedding_size * 2, self.embedding_size)

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size x seq_len x 2]
        """
        batch_size = inputs.shape[0]

        embedded = self.embedding(inputs)
        h = self.mha(embedded)
        h_mean = h.mean(dim=1)

        h_bar = self.h_context_embed(h_mean)
        h_rest = self.v_weight_embed(self.init_W)
        query = h_bar + h_rest

        #init query
        prev_chosen_indices = []
        prev_chosen_logprobs = []
        first_chosen_hs = None
        mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool)

        for index in range(self.seq_len):
            _, n_query = self.glimpse(query, h, mask)
            prob, _ = self.pointer(n_query, h, mask)
            cat = Categorical(prob)
            chosen = cat.sample()
            logprobs = cat.log_prob(chosen)
            prev_chosen_indices.append(chosen)
            prev_chosen_logprobs.append(logprobs)
            #print(chosen.shape)
            #print(batch_size)
            mask[[i for i in range(batch_size)] , chosen] = True
            cc = chosen.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.embedding_size)
            if first_chosen_hs is None:
                first_chosen_hs = h.gather(1, cc).squeeze(1)
            chosen_hs = h.gather(1, cc).squeeze(1)
            h_rest = self.v_weight_embed(torch.cat([first_chosen_hs, chosen_hs], dim=-1))
            query = h_bar + h_rest

        return torch.stack(prev_chosen_logprobs, 1), torch.stack(prev_chosen_indices, 1)

class solver_Attention(nn.Module):
    def __init__(self,
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration):
        super(solver_Attention, self).__init__()

        self.actor = AttentionTSP(
                embedding_size,
                hidden_size,
                seq_len)

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



def glimpse_test():
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
    #print(ret.shape)

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


if __name__ == "__main__":
    print("None!")