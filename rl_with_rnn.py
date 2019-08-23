import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from modules import Attention, GraphEmbedding


class PointerNet(nn.Module):
    def __init__(self,
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration):
        super(PointerNet, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_glimpses = n_glimpses
        self.seq_len = seq_len

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
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        embedded = self.embedding(inputs)
        encoder_outputs, (hidden, context) = self.encoder(embedded)

        prev_chosen_logprobs = []
        preb_chosen_indices = []
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        idxs = None
        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)

        for i in range(seq_len):
            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))

            query = hidden.squeeze(0)
            for j in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
                query = torch.matmul(ref.transpose(-1, -2), F.softmax(logits, dim=-1).unsqueeze(-1)).squeeze(-1)

            _, logits = self.pointer(query, encoder_outputs)
            logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
            probs = F.softmax(logits, dim=-1)
            cat = Categorical(probs)

            idxs = cat.sample()
            log_probs = cat.log_prob(idxs)
            decoder_input = embedded.gather(1, idxs[:, None, None].repeat(1, 1, self.hidden_size)).squeeze(1)
            prev_chosen_logprobs.append(log_probs)
            preb_chosen_indices.append(idxs)

        return torch.stack( prev_chosen_logprobs, 1), torch.stack(preb_chosen_indices, 1)
