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

from combrl import *
from tsp import *
class TrainModel:
    def __init__(self, model, train_dataset, val_dataset, batch_size=128, threshold=None, max_grad_norm=2.):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset   = val_dataset
        self.batch_size = batch_size
        self.threshold = threshold
        self.num_tr_data = len(train_dataset)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        self.val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        self.actor_optim   = optim.Adam(model.actor.parameters(), lr=1e-4)
        self.max_grad_norm = max_grad_norm

        self.train_tour = []
        self.val_tour   = []

        self.epochs = 0

    def train_and_validate(self, n_epochs):
        critic_exp_mvg_avg = torch.zeros(self.num_tr_data).cuda()

        #calculate initial baseline
        print("Caclutating initial Baseline!")

        self.model.train()
        for batch_id, (indices, sample_batch) in tqdm(enumerate(self.train_loader)):

            inputs = sample_batch.cuda()
            R, probs, actions = self.model(inputs)
            critic_exp_mvg_avg[indices] = R

        print("begin training!")
        for epoch in range(n_epochs):
            for batch_id, (indices, sample_batch) in enumerate(self.train_loader):
                self.model.train()
                inputs = sample_batch.cuda()
                R, probs, actions = self.model(inputs)

                critic_exp_mvg_avg[indices] = (critic_exp_mvg_avg[indices] * beta) + ((1. - beta) * R)

                advantage = R - critic_exp_mvg_avg[indices]
                logprobs = torch.sum(probs, dim=-1)
                logprobs[logprobs < -100] = -100
                reinforce = advantage * logprobs
                actor_loss = reinforce.mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), float(self.max_grad_norm))

                self.actor_optim.step()

                critic_exp_mvg_avg = critic_exp_mvg_avg.detach()
                self.train_tour.append(R.mean().data.detach())

                if batch_id % 10 == 0 and (batch_id> 0):
                    print(self.train_tour[-1].cpu().detach())

                if batch_id % 100 == 0:
                    self.model.eval()
                    for _, val_batch in self.val_loader:
                        inputs = val_batch.cuda()

                        R, probs, actions = self.model(inputs)
                        self.val_tour.append(R.mean().data)

            if self.threshold and self.train_tour[-1] < self.threshold:
                print("EARLY STOPPAGE!")
                break

            self.epochs += 1

    def plot(self, epoch):
        clear_output(True)
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.title('train tour length: epoch %s reward %s' % (epoch, self.train_tour[-1] if len(self.train_tour) else 'collecting'))
        plt.plot(self.train_tour)
        plt.grid()
        plt.subplot(132)
        plt.title('val tour length: epoch %s reward %s' % (epoch, self.val_tour[-1] if len(self.val_tour) else 'collecting'))
        plt.plot(self.val_tour)
        plt.grid()
        plt.show()

embedding_size = 128
hidden_size    = 128
n_glimpses = 1
tanh_exploration = 10
use_tanh = True

beta = 0.8
max_grad_norm = 2.

train_size = 10000
val_size = 10000

train_20_dataset = TSPDataset(20, train_size)
val_20_dataset   = TSPDataset(20, val_size)

tsp_20_model = CombinatorialRL(
        embedding_size,
        hidden_size,
        20,
        n_glimpses,
        tanh_exploration,
        use_tanh).cuda()

tsp_20_train = TrainModel(tsp_20_model,
                        train_20_dataset,
                        val_20_dataset,
                        threshold=3.99)

tsp_20_train.train_and_validate(100)