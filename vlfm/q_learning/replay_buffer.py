import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


import torch
import numpy as np
import pickle
from collections import deque


class ReplayBuffer(object):
    def __init__(self, args, save_dir="q_learning_buffer_data"):
        self.batch_size = args.batch_size
        self.buffer_capacity = args.buffer_capacity
        
        # self.current_size = args.buffer_capacity
        # self.count = args.buffer_capacity-1

        self.current_size = 0
        self.count = 0
        
        self.save_dir = save_dir
        self.device = args.device


        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # self.buffer = {'state': np.zeros((self.buffer_capacity, 6, 1000, 1000)),
        #                'action': np.zeros((self.buffer_capacity, 2)), # [row_idx, col_idx]
        #                'reward': np.zeros(self.buffer_capacity),
        #                'next_state': np.zeros((self.buffer_capacity, 6, 1000, 1000)),
        #                'terminal': np.zeros(self.buffer_capacity),
        #                }

        

    # def store_transition(self, state, action, reward, next_state, terminal, done):
    #     self.buffer['state'][self.count] = state
    #     self.buffer['action'][self.count] = action
    #     self.buffer['reward'][self.count] = reward
    #     self.buffer['next_state'][self.count] = next_state
    #     self.buffer['terminal'][self.count] = terminal
    #     self.count = (self.count + 1) % self.buffer_capacity  # When the 'count' reaches buffer_capacity, it will be reset to 0.
    #     self.current_size = min(self.current_size + 1, self.buffer_capacity)


    def store_transition(self, state, action, reward, next_state, terminal, done):
        temp_buffer = {}

        temp_buffer['state'] = state
        temp_buffer['action'] = action
        temp_buffer['reward'] = reward
        temp_buffer['next_state'] = next_state
        temp_buffer['terminal'] = terminal
        self.count = (self.count + 1) % self.buffer_capacity  # When the 'count' reaches buffer_capacity, it will be reset to 0.
        self.current_size = min(self.current_size + 1, self.buffer_capacity)

        if(self.count==0):
            np.save('{}/{}.npy'.format(self.save_dir, self.buffer_capacity), temp_buffer)
        else:
            np.save('{}/{}.npy'.format(self.save_dir, self.count), temp_buffer)

    def sample(self, total_steps):
        index = np.random.randint(1, self.current_size+1, size=self.batch_size)
        batch = {'state': [], 'action': [], 'reward': [], 'next_state': [], 'terminal': []}

        for temp_index in index:
            temp_data = np.load('{}/{}.npy'.format(self.save_dir, temp_index), allow_pickle=True).item()

            batch['state'].append(temp_data['state'])
            batch['action'].append(temp_data['action'])
            batch['reward'].append(temp_data['reward'])
            batch['next_state'].append(temp_data['next_state'])
            batch['terminal'].append(temp_data['terminal'])
        
        for key in batch:  # numpy->tensor
            if key == 'action':
                batch[key] = torch.LongTensor(batch[key]).to(self.device)
            else:
                batch[key] = torch.FloatTensor(batch[key]).to(self.device)

        return batch, None, None

