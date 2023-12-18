from collections import defaultdict
import os
import random
import time
import tqdm
from collections import defaultdict
import os
import pickle
import random
import requests
import time
import tqdm
from IPython.core.debugger import set_trace
import numpy as np
import pandas as pd
from pytorch_ranger import Ranger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as td

class Env():
    def __init__(self, user_item_matrix):
        self.matrix = user_item_matrix
        self.item_count = item_num
        self.memory = np.ones([user_num, params['N']]) * item_num
        # memory is initialized as [item_num] * N for each user
        # it is padding indexes in state_repr and will result in zero embeddings

    def reset(self, user_id):
        self.user_id = user_id
        self.viewed_items = []
        self.related_items = np.argwhere(self.matrix[self.user_id] > 0)[:, 1]
        self.num_rele = len(self.related_items)
        self.nonrelated_items = np.random.choice(
            list(set(range(self.item_count)) - set(self.related_items)), self.num_rele)
        self.available_items = np.zeros(self.num_rele * 2)
        self.available_items[::2] = self.related_items
        self.available_items[1::2] = self.nonrelated_items

        return torch.tensor([self.user_id]), torch.tensor(self.memory[[self.user_id], :])

    def step(self, action, action_emb=None, buffer=None):
        initial_user = self.user_id
        initial_memory = self.memory[[initial_user], :]

        reward = float(to_np(action)[0] in self.related_items)
        self.viewed_items.append(to_np(action)[0])
        if reward:
            if len(action) == 1:
                self.memory[self.user_id] = np.concatenate((self.memory[self.user_id][1:], [action]))

            else:
                self.memory[self.user_id] = list(self.memory[self.user_id][1:]) + [action[0]]

        if len(self.viewed_items) == len(self.related_items):
            done = 1
        else:
            done = 0

        if buffer is not None:
            buffer.push(np.array([initial_user]), np.array(initial_memory), to_np(action_emb)[0],
                        np.array([reward]), np.array([self.user_id]), self.memory[[self.user_id], :],
                        np.array([reward]))

        return torch.tensor([self.user_id]), torch.tensor(self.memory[[self.user_id], :]), reward, done