import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


import torch
import numpy as np
from collections import deque


class ReplayBuffer(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.buffer_capacity = args.buffer_capacity
        self.current_size = 0
        self.count = 0
        self.buffer = {'state': np.zeros((self.buffer_capacity, 6, 1000, 1000)),
                       'action': np.zeros((self.buffer_capacity, 2)), # [row_idx, col_idx]
                       'reward': np.zeros(self.buffer_capacity),
                       'next_state': np.zeros((self.buffer_capacity, 6, 1000, 1000)),
                       'terminal': np.zeros(self.buffer_capacity),
                       }

        self.device = args.device

    def store_transition(self, state, action, reward, next_state, terminal, done):
        self.buffer['state'][self.count] = state
        self.buffer['action'][self.count] = action
        self.buffer['reward'][self.count] = reward
        self.buffer['next_state'][self.count] = next_state
        self.buffer['terminal'][self.count] = terminal
        self.count = (self.count + 1) % self.buffer_capacity  # When the 'count' reaches buffer_capacity, it will be reset to 0.
        self.current_size = min(self.current_size + 1, self.buffer_capacity)

    def sample(self, total_steps):
        index = np.random.randint(0, self.current_size, size=self.batch_size)
        batch = {}
        for key in self.buffer.keys():  # numpy->tensor
            if key == 'action':
                # batch[key] = torch.tensor(self.buffer[key][index], dtype=torch.long)
                batch[key] = torch.LongTensor(self.buffer[key][index]).to(self.device)

            else:
                # batch[key] = torch.tensor(self.buffer[key][index], dtype=torch.float32)
                batch[key] = torch.FloatTensor(self.buffer[key][index]).to(self.device)


        return batch, None, None

