import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical

class Attention(nn.Module):
    def __init__(self, hidden_size, C=10, name='Bahdanau'):
        super(Attention, self).__init__()

        self.C = C
        self.name = name
        self.W_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_ref = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, ref):
        """
        Args:
            query: [batch_size x hidden_size]
            ref:   [batch_size x seq_len x hidden_size]
        """

        batch_size = ref.size(0)
        seq_len    = ref.size(1)
        query = self.W_query(query).unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size x seq_len x hidden_size]
        ref = self.W_ref(ref)  # [batch_size x seq_len x hidden_size]
        logits = self.V(torch.tanh(query + ref)).squeeze(-1)
        logits = self.C * torch.tanh(logits)
        return ref, logits

class PointerNet(nn.Module):
    def __init__(self,
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration):
        super(PointerNet, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size    = hidden_size
        self.n_glimpses     = n_glimpses
        self.seq_len        = seq_len


        self.embedding = GraphEmbedding(2, embedding_size)
        self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.pointer = Attention(hidden_size, C=tanh_exploration)
        self.glimpse = Attention(hidden_size)

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

    def apply_mask_to_logits(self, logits, mask, idxs):
        batch_size = logits.size(0)
        clone_mask = mask.clone()
        if idxs is not None:
            clone_mask[[i for i in range(batch_size)], idxs.data] = 1
            logits[clone_mask] = -np.inf
        return logits, clone_mask

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size x seq_len x 2]
        """
        batch_size = inputs.size(0)
        seq_len    = inputs.size(1)
        assert seq_len == self.seq_len

        embedded = self.embedding(inputs)
        encoder_outputs, (hidden, context) = self.encoder(embedded)


        prev_probs = []
        prev_idxs = []
        try:
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        except:
            mask = torch.zeros(batch_size, seq_len, dtype=torch.uint8)
        idxs = None
        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)

        for i in range(seq_len):
            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))

            query = hidden.squeeze(0)
            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                logits, mask = self.apply_mask_to_logits(logits, mask, idxs)

                query = torch.matmul(ref.transpose(-1, -2), F.softmax(logits, dim=-1).unsqueeze(-1)).squeeze(-1)

            _, logits = self.pointer(query, encoder_outputs)
            logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
            probs = F.softmax(logits, dim=-1)
            cat = Categorical(probs)

            idxs = cat.sample()
            log_probs = cat.log_prob(idxs)
            dec_input = embedded.gather(1, idxs[:, None, None].repeat(1, 1, self.hidden_size)).squeeze(1)
            prev_probs.append(log_probs)
            prev_idxs.append(idxs)

        return torch.stack( prev_probs, 1), torch.stack(prev_idxs, 1)


class solver_RNN(nn.Module):
    def __init__(self,
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration):
        super(solver_RNN, self).__init__()

        self.actor = PointerNet(
                embedding_size,
                hidden_size,
                seq_len,
                n_glimpses,
                tanh_exploration)

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
        inpts = inputs.gather(1, actions.unsqueeze(2).repeat(1, 1, 2))
        R = self.reward(inputs.gather(1, actions.unsqueeze(2).repeat(1, 1, 2)))

        return R, probs, actions

class GraphEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(GraphEmbedding, self).__init__()
        self.embedding_size = embedding_size

        self.embedding = nn.Linear(input_size, embedding_size)
    def forward(self, inputs):
        return self.embedding(inputs)

def test():
    pass

if __name__ == "__main__":
    test()
