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


from IPython.display import clear_output
from tqdm import tqdm
import matplotlib.pyplot as plt

from tsp_heuristic import get_ref_reward
from rl_with_rnn import solver_RNN
from tsp import *

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_type", type=str, default="rnn")
parser.add_argument("--seq_len", type=int, default=20)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--num_tr_dataset", type=int, default=10000)
parser.add_argument("--num_te_dataset", type=int, default=500)
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--grad_clip", type=float, default=1.5)
parser.add_argument("--use_cuda", type=bool, default=False)
parser.add_argument("--beta", type=float, default=0.9)
args = parser.parse_args()

if __name__ =="__main__":
    if args.use_cuda:
        use_pin_memory = True
    else:
        use_pin_memory = False
    train_dataset = TSPDataset(args.seq_len, args.num_tr_dataset)
    test_dataset = TSPDataset(args.seq_len, args.num_te_dataset)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_pin_memory)

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_pin_memory)
    eval_loader = DataLoader(test_dataset, batch_size=args.num_te_dataset, shuffle=False)

    # Calculating heuristics
    heuristic_distance = torch.zeros(args.num_te_dataset)
    for i, pointset in tqdm(test_dataset):
        heuristic_distance[i] = get_ref_reward(pointset)
    if args.model_type == "rnn":
        model = solver_RNN(
            args.embedding_size,
            args.hidden_size,
            args.seq_len,
            2, 10)

    if args.use_cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=3.0 * 1e-4)

    # Train loop
    moving_avg = torch.zeros(args.num_tr_dataset)
    if args.use_cuda:
        moving_avg = moving_avg.cuda()

    #generating first baseline
    for (indices, sample_batch) in tqdm(train_data_loader):
        if args.use_cuda:
            sample_batch = sample_batch.cuda()
        rewards, _, _ = model(sample_batch)
        moving_avg[indices] = rewards

    #Training
    model.train()
    for epoch in range(args.num_epochs):
        for batch_idx, (indices, sample_batch) in enumerate(train_data_loader):
            if args.use_cuda:
                sample_batch.cuda()
            rewards, log_probs, action = model(sample_batch)
            moving_avg[indices] = moving_avg[indices] * args.beta + rewards * (1.0 - args.beta)
            advantage = rewards - moving_avg[indices]
            log_probs = torch.sum(log_probs, dim=-1)
            log_probs[log_probs < -100] = -100
            loss = (advantage * log_probs).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if batch_idx % 100 == 0:
                model.eval()
                ret = []
                for i, batch in eval_loader:
                    if args.use_cuda:
                        batch = pointset.cuda()
                    R, _, _ = model(batch)
                print("[at epoch %d, %d'th batch]RL model generates %0.2f time worse solution than heuristics" %(
                        epoch,
                        batch_idx,
                        (R / heuristic_distance).mean().detach().numpy()))

                model.train()
