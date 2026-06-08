from vlfm.arguments import args as main_args

import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(main_args.card_select)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = str(main_args.card_select)

import json
import time
from vlfm.sac_intra.SACD import SACD_agent
from vlfm.sac_intra.arguments import args as q_args
import numpy as np
import torch

import datetime

from torch.utils.tensorboard import SummaryWriter

import copy

device = "cuda"

import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from itertools import islice

# def save_total_steps(steps, filename='main_process_info/total_steps.json'):
#     """保存总步数到JSON文件"""
#     data = {'total_steps': steps}
#     with open(filename, 'w') as f:
#         json.dump(data, f)

# 新增
# 模拟数据生成器 - 实际应用中替换为真实数据加载器
class ObjectGoalDataset(Dataset):
    def __init__(self, all_file_ls):
        self.samples = all_file_ls

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # 加载单个样本
        temp_file = self.samples[idx]

        try:
            data = np.load(temp_file, allow_pickle=True).item()
        except:
            while True:
                try:
                    data = np.load(temp_file, allow_pickle=True).item()
                    break
                except:
                    time.sleep(0.1)
                    continue

        return {
            'state': torch.FloatTensor(data['state']),
            'action': torch.LongTensor(data['action']),
            'reward': torch.tensor(data['reward'], dtype=torch.float32),
            'next_state': torch.FloatTensor(data['next_state']),
            'terminal': torch.tensor(data['terminal'], dtype=torch.float32),
        }

# def find_max_number_in_filenames(folder_path):
#     """
#     查找文件夹中.npy文件名中的最大数字
    
#     参数:
#         folder_path (str): 包含.npy文件的文件夹路径
    
#     返回:
#         int: 文件名中的最大数字
#     """

#     if(len(os.listdir(folder_path))==0):
#         return 0


#     max_number = max([int(filename.split("_")[0]) for filename in os.listdir(folder_path) if (len(filename.split("_"))==2)])    
#     return max_number


# def load_total_steps(now_dir='main_process_info/step_rewards/'):
#     max_id = find_max_number_in_filenames(folder_path=f'{q_args.root}/{q_args.model_file_name}/policy/{q_args.experiment_details}/')
#     file_name_ls = os.listdir(now_dir)

#     if(max_id==0):
#         return len(os.listdir(now_dir)), file_name_ls
#     else:
#         return max_id*q_args.delta_steps+len(file_name_ls)+q_args.random_steps, file_name_ls

def load_total_steps(now_dir='main_process_info/step_rewards/'):
    return len(os.listdir(now_dir))

def get_delta_step_reward(last_reward_steps, now_reward_steps, now_dir='main_process_info/step_rewards/'):
    # last_reward_steps开区间，now_reward_steps闭区间
    file_name_ls = os.listdir(now_dir)
    delta_step_reward_ls = []
    for temp_file in file_name_ls:
        temp_file_step = eval(temp_file.strip(".txt").split("_")[1])
        if(temp_file_step>last_reward_steps) and (temp_file_step<=now_reward_steps):
            delta_step_reward_ls.append(eval(temp_file.strip(".txt").split("_")[2]))
    return delta_step_reward_ls


def find_max_number_in_npy_filenames(folder_path=q_args.buffer_data_folder):
    """
    查找文件夹中.npy文件名中的最大数字
    
    参数:
        folder_path (str): 包含.npy文件的文件夹路径
    
    返回:
        int: 文件名中的最大数字
    """

    if(len(os.listdir(folder_path))==0):
        return 0

    max_number = max([int(filename.strip(".npy")) for filename in os.listdir(folder_path)])    
    return max_number


# def load_step_rewards(filename: str = 'main_process_info/step_rewards.json'):
#     """
#     从JSON文件读取所有step reward
#     参数:
#         filename: 读取文件名
#     返回:
#         包含所有step reward的列表
#     """
#     if not os.path.exists(filename):
#         return []
    
#     with open(filename, 'r') as f:
#         return json.load(f)


# def clear_step_rewards(filename: str = 'main_process_info/step_rewards.json'):
#     """清空reward记录文件"""
#     with open(filename, 'w') as f:
#         json.dump([], f)

# def clear_directory(directory_path):
#     """
#     清空指定目录下的所有文件（不删除子目录）
    
#     :param directory_path: 要清空的目录路径
#     """
#     # 遍历目录下的所有文件和子目录
#     file_name_ls = copy.deepcopy(os.listdir(directory_path))
#     for filename in file_name_ls:
        
#         temp_id = eval(filename.strip(".txt").split("_")[1])//q_args.delta_steps
#         max_id = find_max_number_in_filenames(folder_path=f'{q_args.root}/{q_args.model_file_name}/policy/{q_args.experiment_details}/')
#         if(temp_id<max_id) or (eval(filename.strip(".txt").split("_")[1])==max_id*q_args.delta_steps):
#             file_path = os.path.join(directory_path, filename)
#             # 如果是文件，则删除
#             os.remove(file_path)
#             print(f"已删除文件: {file_path}")

'''
def clear_directory(directory_path):
    """
    清空指定目录下的所有文件（不删除子目录）
    
    :param directory_path: 要清空的目录路径
    """
    # 遍历目录下的所有文件和子目录
    file_name_ls = copy.deepcopy(os.listdir(directory_path))
    for filename in file_name_ls:
        file_path = os.path.join(directory_path, filename)
        
        # 如果是文件，则删除
        os.remove(file_path)
        print(f"已删除文件: {file_path}")
'''

# 更改
# def sample(total_steps):
#     save_dir = q_args.buffer_data_folder

#     # all_file_ls = os.listdir(save_dir)
#     # index = np.random.randint(0, len(all_file_ls), size=q_args.batch_size)


#     if (len(os.listdir(save_dir))==q_args.buffer_capacity):
#         current_size = q_args.buffer_capacity
#     else:
#         current_size = find_max_number_in_npy_filenames()

#     index = np.random.randint(1, current_size+1, size=q_args.batch_size)
#     batch = {'state': [], 'action': [], 'reward': [], 'next_state': [], 'terminal': []}

#     for temp_index in index:

#         try:
#             temp_data = np.load('{}/{}.npy'.format(save_dir, temp_index), allow_pickle=True).item()
#         except:
#             for temp_i in range(1, 10000):
#                 try:
#                     temp_data = np.load('{}/{}.npy'.format(save_dir, temp_i), allow_pickle=True).item()
#                     break
#                 except:
#                     continue

#         # temp_data = np.load('{}/{}'.format(save_dir, all_file_ls[temp_index]), allow_pickle=True).item()

#         batch['state'].append(temp_data['state'])
#         batch['action'].append(temp_data['action'])
#         batch['reward'].append(temp_data['reward'])
#         batch['next_state'].append(temp_data['next_state'])
#         batch['terminal'].append(temp_data['terminal'])
    
#     for key in batch:  # numpy->tensor
#         if key == 'action':
#             batch[key] = torch.LongTensor(batch[key]).to(device)
#         else:
#             batch[key] = torch.FloatTensor(batch[key]).to(device)

#     return batch


def sample():
    save_dir = q_args.buffer_data_folder
    os_file_ls = copy.deepcopy(os.listdir(save_dir))
    all_file_ls = [save_dir+"/"+temp_file for temp_file in os_file_ls]

    train_dataset = ObjectGoalDataset(all_file_ls=all_file_ls)
    # train_loader = DataLoader(train_dataset, batch_size=4, 
    #                          shuffle=True, num_workers=0, pin_memory=False)
    # train_loader = DataLoader(train_dataset, batch_size=16, 
    #                          shuffle=True, num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_dataset, batch_size=q_args.real_batch_size, 
                             shuffle=True, num_workers=0, pin_memory=False)
    
    return train_loader

import time

if __name__ == "__main__":    
    # while True:
    #     temp_id = find_max_number_in_npy_filenames()
    #     if (temp_id<2600):
    #         print("======> temp_id <======", temp_id)
    #         time.sleep(0.1)
    #     else:
    #         break


    date_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # logger_file_name = "./log_files/part2_log_for_main_process_gt_perception_multi_stairs_RL_train_"+date_time # 300
    # logger_file_name = "./log_files_cross/part2_log_for_main_process_based_on_actor_989_"+date_time # 300
    logger_file_name = "./log_files_cross/part2_log_for_main_process_based_on_actor_huawei_408_"+date_time # 300
    writer = SummaryWriter(logger_file_name)

    rl_policy = SACD_agent(q_args)
    rl_policy.actor.train()
    rl_policy.q_critic.train()


    # update_step = 25
    update_step = 408
    # update_step = 989

    # num_batch = int(q_args.batch_size/4)
    # num_batch = int(q_args.batch_size/16)
    num_batch = int(q_args.batch_size/q_args.real_batch_size)
    last_rl_steps = load_total_steps()
    last_reward_steps = last_rl_steps


    while True:
        now_total_steps = load_total_steps() # 现在已经存储的rl_step总量
        # ==========> 真正的主线程代码开始 <==========
        # if ((now_total_steps//q_args.delta_steps)>update_step):

        if ((now_total_steps-last_rl_steps)>=q_args.delta_steps): # 两次进训练的数据间隔大于等于delta_steps
            print(f"training for {now_total_steps}, last step {last_rl_steps}")
            last_rl_steps = now_total_steps
            # update_step = (now_total_steps//q_args.delta_steps)
            if (update_step==408):
                start_time = time.time()
            
            
            
            update_step = update_step+1
            train_loader = sample()
            progress_bar = tqdm(islice(train_loader, num_batch), desc=f'Epoch {update_step}', total=num_batch)

            all_total_actor_loss = 0
            all_total_kl_loss = 0
            all_total_sum_loss = 0
            all_total_critic_loss = 0
            all_total_critic_q = 0
            # all_total_alpha_loss = 0


            for batch in progress_bar:
                total_actor_loss, total_kl_loss, total_sum_loss, total_critic_loss, curr_alpha, total_critic_q = rl_policy.learn(batch)
                
                del batch
                torch.cuda.empty_cache()

                all_total_actor_loss += total_actor_loss
                all_total_kl_loss += total_kl_loss
                all_total_sum_loss += total_sum_loss
                all_total_critic_loss += total_critic_loss
                all_total_critic_q += total_critic_q
                # all_total_alpha_loss += total_alpha_loss

            del train_loader
            del progress_bar
            torch.cuda.empty_cache()

            now_time = time.time()


            print("delta_time:", now_time-start_time)
            # episode_rl_step_buffer = [eval(temp_file.strip(".txt").split("_")[2]) for temp_file in file_name_ls]
            
            # 两次模型保存之间的reward，reward的大小仍然对应上一个模型
            now_reward_steps = load_total_steps()
            episode_rl_step_buffer = get_delta_step_reward(last_reward_steps, now_reward_steps)
            last_reward_steps = now_reward_steps

            mean_episode_rl_step_buffer = np.mean(episode_rl_step_buffer)
            writer.add_scalar("Result_RL/mean_episode_rl_step_buffer", mean_episode_rl_step_buffer, update_step)
            writer.add_scalar("Result_RL/all_total_actor_loss", all_total_actor_loss/num_batch, update_step)
            writer.add_scalar("Result_RL/all_total_kl_loss", all_total_kl_loss/num_batch, update_step)
            writer.add_scalar("Result_RL/all_total_sum_loss", all_total_sum_loss/num_batch, update_step)

            writer.add_scalar("Result_RL/all_total_critic_loss", all_total_critic_loss/num_batch, update_step)

            writer.add_scalar("Result_RL/all_total_critic_q", all_total_critic_q/num_batch, update_step)
            # writer.add_scalar("Result_RL/all_total_alpha_loss", all_total_alpha_loss/num_batch, update_step)
            writer.add_scalar("Result_RL/curr_alpha", curr_alpha, update_step)

            # =====> 创建新目录 <=====
            model_pre_dir = '{0}/{1}/policy/{2}'.format(q_args.root, q_args.model_file_name, q_args.experiment_details)
            if not os.path.exists(model_pre_dir):
                os.makedirs(model_pre_dir)
                print(f"The new path:'{model_pre_dir}' has beed craeted!")
            # =====> 创建新目录 <=====

            rl_policy.save('{0}/{1}/policy/{2}/{3}'.format(q_args.root, q_args.model_file_name, q_args.experiment_details, update_step))
            # clear_directory(directory_path="main_process_info/step_rewards/")

        else:
            print(f"waiting for {now_total_steps}, last step {last_rl_steps}")
        
        # ==========> 真正的主线程代码结束 <==========
        time.sleep(0.1)
