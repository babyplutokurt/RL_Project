from collections import defaultdict
import os
import pickle
import random
import requests
import time
import tqdm
from utils import (EvalDataset, OUNoise, Prioritized_Buffer, get_beta,
                   preprocess_data, to_np, hit_metric, dcg_metric)
from IPython.core.debugger import set_trace
import numpy as np
import pandas as pd
from pytorch_ranger import Ranger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
from torch.utils.tensorboard import SummaryWriter

data_dir = "data"
rating = "ml-1m.train.rating"

params = {
    'batch_size': 512,
    'embedding_dim': 8,
    'hidden_dim': 16,
    'N': 5,  # memory size for state_repr
    'ou_noise': False,

    'value_lr': 1e-5,
    'value_decay': 1e-4,
    'policy_lr': 1e-5,
    'policy_decay': 1e-6,
    'state_repr_lr': 1e-5,
    'state_repr_decay': 1e-3,
    'log_dir': 'logs/final/',
    'gamma': 0.8,
    'min_value': -10,
    'max_value': 10,
    'soft_tau': 1e-3,

    'buffer_size': 1000000
}
if not os.path.isdir('./data'):
    os.mkdir('./data')

file_path = os.path.join(data_dir, rating)
if os.path.exists(file_path):
    print("Skip loading " + file_path)

(train_data, train_matrix, test_data, test_matrix,
 user_num, item_num, appropriate_users) = preprocess_data(data_dir, rating)

class Actor_DRR(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

        self.initialize()

    def initialize(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

    def forward(self, state):
        return self.layers(state)

    def get_action(self, user, memory, state_repr,
                   action_emb,
                   items=torch.tensor([i for i in range(item_num)]),
                   return_scores=False
                   ):
        state = state_repr(user, memory)
        scores = torch.bmm(state_repr.item_embeddings(items).unsqueeze(0),
                           action_emb.T.unsqueeze(0)).squeeze(0)
        if return_scores:
            return scores, torch.gather(items, 0, scores.argmax(0))
        else:
            return torch.gather(items, 0, scores.argmax(0))


class Critic_DRR(nn.Module):
    def __init__(self, state_repr_dim, action_emb_dim, hidden_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_repr_dim + action_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.initialize()

    def initialize(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.layers(x)
        return x


class State_Repr_Module(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim, hidden_dim):
        super().__init__()
        self.user_embeddings = nn.Embedding(user_num, embedding_dim)
        self.item_embeddings = nn.Embedding(item_num + 1, embedding_dim, padding_idx=int(item_num))
        self.drr_ave = torch.nn.Conv1d(in_channels=params['N'], out_channels=1, kernel_size=1)

        self.initialize()

    def initialize(self):
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        self.item_embeddings.weight.data[-1].zero_()
        nn.init.uniform_(self.drr_ave.weight)
        self.drr_ave.bias.data.zero_()

    def forward(self, user, memory):
        user_embedding = self.user_embeddings(user.long())

        item_embeddings = self.item_embeddings(memory.long())
        drr_ave = self.drr_ave(item_embeddings).squeeze(1)

        return torch.cat((user_embedding, user_embedding * drr_ave, drr_ave), 1)

