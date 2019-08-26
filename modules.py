from math import sqrt

import torch
import torch.nn as nn


# Linear Embedding
class GraphEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(GraphEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Linear(input_size, embedding_size)

    def forward(self, inputs):
        return self.embedding(inputs)


# Glimpse using Dot-product attention
class Glimpse(nn.Module):
    def __init__(self,
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


# Pointer using Dot-product attention
class Pointer(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_head,
                 C=10):

        super(Pointer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.C = C

        self.W_q = nn.Linear(self.input_size, self.hidden_size)
        self.W_k = nn.Linear(self.input_size, self.hidden_size)
        self.W_v = nn.Linear(self.input_size, self.hidden_size)

    def forward(self, query, target, mask=None):
        """
        Parameters
        ----------
        query : FloatTensor [batch_size x input_size]
        target : FloatTensor [batch_size x seq_len x input_size]
        mask : BoolTensor [batch_size x seq_len]
        """
        batch_size, seq_len, _ = target.shape

        q_c = self.W_q(query)  # batch_size x hidden_size
        k = self.W_k(target)  # batch_size x seq_len x hidden_size
        v = self.W_v(target)  # batch_size x seq_len x hidden_size
        qk = torch.einsum("ik,ijk->ij", [q_c, k])  # batch_size x seq_len
        qk = self.C * torch.tanh(qk)

        if mask is not None:
            _mask = mask.clone()
            qk[_mask] = -100000.0

        alpha = torch.softmax(qk, dim=-1)
        ret = torch.einsum("ij,ijk->ij", [alpha, v])

        return alpha, ret


# Attention/Pointer module using Bahanadu Attention
class Attention(nn.Module):
    def __init__(self, hidden_size, C=10):
        super(Attention, self).__init__()
        self.C = C
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, 1)

    def forward(self, query, target):
        """
        Args:
            query: [batch_size x hidden_size]
            target:   [batch_size x seq_len x hidden_size]
        """

        batch_size, seq_len, _ = target.shape
        query = self.W_q(query).unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size x seq_len x hidden_size]
        target = self.W_k(target)  # [batch_size x seq_len x hidden_size]
        logits = self.W_v(torch.tanh(query + target)).squeeze(-1)
        logits = self.C * torch.tanh(logits)
        return target, logits
