from vlfm.arguments import args as main_args

import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(main_args.card_select)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = str(main_args.card_select)


import torch
import numpy as np
import copy

from vlfm.policy.unet_model import HistoryAwareUNet, Double_Q_Net
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from vlfm.sac_intra.arguments import args as q_args

seed = 233

np.random.seed(seed)
torch.manual_seed(seed)

import random
random.seed(seed)


import torch
import torch.nn as nn

def weights_init_normal(m, mean=0.0, std=0.02):
    """
    将权重初始化为均值为 0, 标准差为 std 的正态分布。
    将偏置初始化为 0。
    """
    classname = m.__class__.__name__
    
    # 我们只对 线性层 和 卷积层 进行初始化
    if classname.find('Linear') != -1 or classname.find('Conv') != -1:
        
        # --- 1. 初始化权重 (Weight) ---
        
        # 选项A: 正态分布 (推荐)
        # m.weight.data 是权重的张量
        torch.nn.init.normal_(m.weight.data, mean=mean, std=std)
        
        # 选项B: 均匀分布 (也可以)
        # init_val = 0.01 
        # torch.nn.init.uniform_(m.weight.data, -init_val, init_val)

        # --- 2. 初始化偏置 (Bias) ---
        if m.bias is not None:
            # 将偏置初始化为 0
            torch.nn.init.constant_(m.bias.data, 0.0)


class SACD_agent(object):
    def __init__(self, args):
        self.H_mean = 0
        self.batch_size = args.batch_size  # batch size
        self.lr = args.lr  # learning rate
        self.actor_lr = args.actor_lr
        self.gamma = args.gamma  # discount factor
        self.tau = args.tau  # Soft update
        self.device =  args.device
        self.adaptive_alpha = args.adaptive_alpha
        self.alpha = args.alpha
        self.random_steps = args.random_steps

        self.actor = HistoryAwareUNet(in_channels=6).to(args.device)
        self.actor_il = HistoryAwareUNet(in_channels=6).to(args.device)
        
        # # load训练好的模型(il)
        # checkpoint = torch.load(args.pre_policy, map_location=args.device)['model_state']
        # self.actor.load_state_dict(checkpoint)
        # self.actor_il.load_state_dict(checkpoint)

        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # ===================
        # load训练好的模型(rl)(在某个RL基础上接着训练)
        checkpoint = torch.load("Models_train_PPO_intra/policy/multi_process_sac/408_actor")
        self.actor.load_state_dict(checkpoint)

        checkpoint = torch.load(args.pre_policy, map_location=args.device)['model_state']
        self.actor_il.load_state_dict(checkpoint)

        # checkpoint = torch.load("Models_train_PPO_intra_copy/policy/multi_process_sac/408_actor")
        # self.actor_il.load_state_dict(checkpoint)


        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        checkpoint = torch.load("Models_train_PPO_intra/policy/multi_process_sac/408_actor_optimizer")
        self.actor_optimizer.load_state_dict(checkpoint)
        # ===================

        self.init_T = 10
        self.q_critic = Double_Q_Net(in_channels=6, device=args.device).to(args.device)

        # # ===================
        # self.q_critic.apply(weights_init_normal)
        # self.q_critic_target = copy.deepcopy(self.q_critic)
        # self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.lr)
        #  # ===================

        # critic_sample暂时不要(导入预训练的critic，暂时会用)
        # ========================================
        # load训练好的模型(rl)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.lr)
        checkpoint = torch.load("Models_train_PPO_intra/policy/multi_process_sac/408_critic_optimizer")
        self.q_critic_optimizer.load_state_dict(checkpoint)
        
        checkpoint = torch.load("Models_train_PPO_intra/policy/multi_process_sac/408_critic")
        self.q_critic.load_state_dict(checkpoint)

        checkpoint = torch.load("Models_train_PPO_intra/policy/multi_process_sac/408_critic_target")
        self.q_critic_target = Double_Q_Net(in_channels=6, device=args.device).to(args.device)
        self.q_critic_target.load_state_dict(checkpoint)
        # ========================================
    
    
        # # critic_sample暂时不要(导入预训练的critic，暂时会用, 继续某个训练采用)
        # # ========================================
        # # load训练好的模型(rl)
        # self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.lr)
        # checkpoint = torch.load("Models_train_PPO/policy/multi_process_sac/1261_critic_optimizer")
        # self.q_critic_optimizer.load_state_dict(checkpoint)
        
        # checkpoint = torch.load("Models_train_PPO/policy/multi_process_sac/1261_critic")
        # self.q_critic.load_state_dict(checkpoint)

        # checkpoint = torch.load("Models_train_PPO/policy/multi_process_sac/1261_critic_target")
        # self.q_critic_target = MultiFloor_Double_Q_Net(pretrained_path="Models_train_PPO_one_floor/policy/multi_process_sac/1002_actor", device=args.device).to(args.device)
        # self.q_critic_target.load_state_dict(checkpoint)
        # # ========================================

        for p in self.q_critic_target.parameters(): p.requires_grad = False
        
        
        # if self.adaptive_alpha:
		# 	# We use 0.6 because the recommended 0.98 will cause alpha explosion.
        #     # action_dim = 1000*1000
        #     # action_dim = 128
        #     action_dim = 32
        #     self.target_entropy = 0.6 * (-np.log(1 / action_dim))  # H(discrete)>0
        #     self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=args.device)
        #     self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)
        

    def choose_action(self, state):
        with torch.no_grad():
            semantic_frontier_map = state[3] 
            mask_map = (semantic_frontier_map > 0).astype(np.float32)

            net_state = torch.FloatTensor(state)
            net_state = net_state.unsqueeze(0).to(self.device)
            preds = self.actor(net_state)
        
            pred_map = preds[0, 0].detach().cpu().numpy()
            masked_pred = pred_map.copy()
            valid_mask = (mask_map > 0)
            masked_pred[~valid_mask] = -float('inf')
            masked_pred_flat = masked_pred.flatten()
            softmax_probs = F.softmax(torch.from_numpy(masked_pred_flat), dim=0)
            sampled_index = Categorical(softmax_probs).sample()
            pred_row = sampled_index.item() // 500
            pred_col = sampled_index.item() % 500

            action = np.array([pred_row, pred_col])
            return action, pred_row, pred_col


    def learn(self, batch):
        # mini_batch_size = 4
        # mini_batch_size = 16
        mini_batch_size = q_args.real_batch_size
        num_mini_batches = len(batch['state']) // mini_batch_size  # Should be 8 for batch size 64

        total_critic_loss = 0
        total_actor_loss = 0
        total_kl_loss = 0
        total_sum_loss = 0
        # total_alpha_loss = 0

        total_critic_q = 0
        
        for i in range(num_mini_batches):
            start_idx = i * mini_batch_size
            end_idx = (i + 1) * mini_batch_size

            # Slice the batch for current mini-batch
            mini_batch = {
                'state': batch['state'][start_idx:end_idx],
                'action': batch['action'][start_idx:end_idx],
                'reward': batch['reward'][start_idx:end_idx],
                'next_state': batch['next_state'][start_idx:end_idx],
                'terminal': batch['terminal'][start_idx:end_idx]
            }

            mini_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in mini_batch.items()}

            #------------------------------------------ Train Critic ----------------------------------------#
            '''Compute the target soft Q value'''
            with torch.no_grad():  # q_target has no gradient
                next_logits = self.actor(mini_batch['next_state'])
                next_probs = []
                for i in range(next_logits.size(0)):
                    next_logits_map = next_logits[i]

                    semantic_frontier_map = mini_batch['next_state'][i, 3]
                    mask_map = (semantic_frontier_map > 0).float()



                    masked_pred = next_logits_map.clone()
                    masked_pred = torch.where(mask_map > 0, masked_pred, torch.tensor(-float('inf'), device=self.device))

                    # 展平并计算softmax
                    masked_pred_flat = masked_pred.view(-1)
                    softmax_probs = F.softmax(masked_pred_flat, dim=0)
                    
                    next_probs.append(softmax_probs)
                

                next_probs = torch.stack(next_probs, dim=0)
                next_log_probs = torch.log(next_probs+1e-8) #[b,a_dim]

                next_q1_map, next_q2_map = self.q_critic_target(mini_batch['next_state'])  
                next_q1_all = next_q1_map.view(next_q1_map.size(0), -1)
                next_q2_all = next_q2_map.view(next_q2_map.size(0), -1)
                min_next_q_all = torch.min(next_q1_all, next_q2_all) # [b,a_dim]
                v_next = torch.sum(next_probs * (min_next_q_all - self.alpha * next_log_probs), dim=1, keepdim=True) # [b,1]
                target_Q = mini_batch['reward'].unsqueeze(1) +  self.gamma * (1 - mini_batch['terminal']).unsqueeze(1) * v_next # [b, 1]

            '''Update soft Q net''' 
            q1_map, q2_map = self.q_critic(mini_batch['state']) 
            row_idx = mini_batch['action'][:, 0]  # 维度: (B,)
            col_idx = mini_batch['action'][:, 1]  # 维度: (B,) 

            # 提取对应 [row, col] 的 Q 值
            batch_idx = torch.arange(mini_batch['state'].size(0), device=self.device)  # 维度: (B,)
            q1 = q1_map[batch_idx, 0, row_idx, col_idx].unsqueeze(1)  # 维度: (B, 1)     
            q2 = q2_map[batch_idx, 0, row_idx, col_idx].unsqueeze(1)  # 维度: (B, 1) 
            q_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)


            total_critic_loss += q_loss.mean().item()

            self.q_critic_optimizer.zero_grad()
            q_loss.backward()
            self.q_critic_optimizer.step()

            #------------------------------------------ Train Actor ----------------------------------------#
            logits = self.actor(mini_batch['state']) #[b,a_dim]
            probs = []
            for i in range(logits.size(0)):
                logits_map = logits[i, 0]  # 不再使用detach()
                curr_semantic_frontier_map = mini_batch['state'][i, 3]
                curr_mask_map = (curr_semantic_frontier_map > 0).float()

                curr_masked_pred = logits_map.clone()
                curr_masked_pred = torch.where(curr_mask_map > 0, curr_masked_pred, torch.tensor(-float('inf'), device=self.device))

                
                # 展平并计算softmax
                curr_masked_pred_flat = curr_masked_pred.view(-1)
                curr_softmax_probs = F.softmax(curr_masked_pred_flat, dim=0)
                probs.append(curr_softmax_probs)
            
            probs = torch.stack(probs, dim=0)
            log_probs = torch.log(probs+1e-8) #[b,a_dim]
            with torch.no_grad():
                q1_map, q2_map = self.q_critic(mini_batch['state'])  
                q1_all = q1_map.view(q1_map.size(0), -1) #[b,a_dim]
                q2_all = q2_map.view(q2_map.size(0), -1) #[b,a_dim]
            min_q_all = torch.min(q1_all, q2_all)

            temp_total_critic_q = torch.sum(probs * min_q_all, dim=1, keepdim=False) #[b,]
            total_critic_q += temp_total_critic_q.mean().item()

            # ==========> actor_il <==========
            logits_il = self.actor_il(mini_batch['state'])
            probs_il = []
            for i in range(logits_il.size(0)):
                logits_map_il = logits_il[i, 0]  # 不再使用detach()
                curr_semantic_frontier_map = mini_batch['state'][i, 3]
                curr_mask_map = (curr_semantic_frontier_map > 0).float()

                curr_masked_pred_il = logits_map_il.clone()
                curr_masked_pred_il = torch.where(curr_mask_map > 0, curr_masked_pred_il, torch.tensor(-float('inf'), device=self.device))
                
                # 展平并计算softmax
                curr_masked_pred_flat_il = curr_masked_pred_il.view(-1)
                curr_softmax_probs_il = F.softmax(curr_masked_pred_flat_il, dim=0)
                probs_il.append(curr_softmax_probs_il)


            probs_il = torch.stack(probs_il, dim=0)
            log_probs_il = torch.log(probs_il+1e-8) #[b,a_dim]
            # ==========> actor_il <==========

            a_loss = torch.sum(probs * (self.alpha*log_probs - min_q_all), dim=1, keepdim=False) #[b,]

            kl_div = F.kl_div(
                input=log_probs, # input
                target=probs_il,  # target
                reduction='batchmean',
                log_target=False
            )

            kl_weight = 0.5

            total_actor_loss += a_loss.mean().item()
            total_kl_loss += kl_weight*kl_div.item()
            total_sum_loss = total_actor_loss+total_kl_loss
            # total_sum_loss = total_actor_loss
            self.actor_optimizer.zero_grad()
            # a_loss.mean().backward()
            (a_loss.mean()+kl_weight*kl_div).backward()
            self.actor_optimizer.step()

            
            #------------------------------------------ Train Alpha ----------------------------------------#
            
            # if self.adaptive_alpha:
            #     with torch.no_grad():
            #         self.H_mean = -torch.sum(probs * log_probs, dim=1).mean()
            #     alpha_loss = self.log_alpha * (self.H_mean - self.target_entropy)

            #     total_alpha_loss += alpha_loss.mean().item()

            #     self.alpha_optim.zero_grad()
            #     alpha_loss.backward()
            #     self.alpha_optim.step()

            #     self.alpha = self.log_alpha.exp().item()
            

            #------------------------------------------ Update Target Net ----------------------------------#
            for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # Clean up memory
            # del mini_batch, next_logits, next_probs, next_log_probs, next_q1_map, next_q2_map, next_q1_all, next_q2_all, min_next_q_all, v_next, target_Q, q1_map, q2_map, q1, q2, q_loss, logits, probs, log_probs, q1_all, q2_all, min_q_all, a_loss, alpha_loss
            del mini_batch, next_logits, next_probs, next_log_probs, next_q1_map, next_q2_map, next_q1_all, next_q2_all, min_next_q_all, v_next, target_Q, q1_map, q2_map, q1, q2, q_loss, logits, probs, log_probs, q1_all, q2_all, min_q_all, a_loss
            torch.cuda.empty_cache()
        
        # return total_actor_loss / num_mini_batches, total_critic_loss / num_mini_batches, total_alpha_loss / num_mini_batches, self.alpha
        # return total_actor_loss / num_mini_batches, total_critic_loss / num_mini_batches, self.alpha
        return total_actor_loss, total_kl_loss, total_sum_loss, total_critic_loss, self.alpha, total_critic_q


        
    
    # def learn_critic(self, batch):
    #     mini_batch_size = q_args.real_batch_size
    #     num_mini_batches = len(batch['state']) // mini_batch_size  # Should be 8 for batch size 64

    #     total_critic_loss = 0
        
    #     for i in range(num_mini_batches):
    #         start_idx = i * mini_batch_size
    #         end_idx = (i + 1) * mini_batch_size

    #         # Slice the batch for current mini-batch
    #         mini_batch = {
    #             'state': batch['state'][start_idx:end_idx],
    #             'action': batch['action'][start_idx:end_idx],
    #             'reward': batch['reward'][start_idx:end_idx],
    #             'next_state': batch['next_state'][start_idx:end_idx],
    #             'terminal': batch['terminal'][start_idx:end_idx]
    #         }

    #         mini_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
    #                   for k, v in mini_batch.items()}

    #         #------------------------------------------ Train Critic ----------------------------------------#
    #         '''Compute the target soft Q value'''
    #         with torch.no_grad():  # q_target has no gradient
    #             next_logits = self.actor(mini_batch['next_state'])
    #             next_probs = []
    #             for i in range(next_logits.size(0)):
    #                 next_logits_map = next_logits[i]

    #                 # floor_mask数据
    #                 final_floor_mask = [1, 1, 1]
    #                 if (np.sum(mini_batch['next_state'][i, :6, :, :].cpu().numpy()[5])==0) or (np.sum(mini_batch['next_state'][i, :6, :, :].cpu().numpy()[5])!=0 and np.sum(mini_batch['next_state'][i, :6, :, :].cpu().numpy()[4])!=0 and np.sum(mini_batch['next_state'][i, :6, :, :].cpu().numpy()[3])==0):
    #                     final_floor_mask[0] = 0
    #                 if (np.sum(mini_batch['next_state'][i, 6:12, :, :].cpu().numpy()[4])!=0 and np.sum(mini_batch['next_state'][i, 6:12, :, :].cpu().numpy()[3])==0):
    #                     final_floor_mask[1] = 0
    #                 if (np.sum(mini_batch['next_state'][i, 12:18, :, :].cpu().numpy()[5])==0) or (np.sum(mini_batch['next_state'][i, 12:18, :, :].cpu().numpy()[5])!=0 and np.sum(mini_batch['next_state'][i, 12:18, :, :].cpu().numpy()[4])!=0 and np.sum(mini_batch['next_state'][i, 12:18, :, :].cpu().numpy()[3])==0):
    #                     final_floor_mask[2] = 0

    #                 mask_map = np.array(final_floor_mask)
    #                 mask_map = torch.tensor(mask_map, device=self.device)
    #                 masked_pred = next_logits_map.clone()
    #                 masked_pred = torch.where(mask_map > 0, masked_pred, torch.tensor(-float('inf'), device=self.device))
                    
    #                 # 展平并计算softmax
    #                 masked_pred_flat = masked_pred.view(-1)
    #                 # softmax_probs = F.softmax(masked_pred_flat/self.init_T, dim=0)
    #                 softmax_probs = F.softmax(masked_pred_flat, dim=0)
                    
    #                 next_probs.append(softmax_probs)
                
    #             # next_probs = np.array(next_probs)
    #             # next_probs = torch.FloatTensor(next_probs).to(self.device) # shape：(batch_size, action_dim)
    #             next_probs = torch.stack(next_probs, dim=0)
    #             next_log_probs = torch.log(next_probs+1e-8) #[b,a_dim]

    #             next_q1_map, next_q2_map = self.q_critic_target(mini_batch['next_state'])  
    #             next_q1_all = next_q1_map.view(next_q1_map.size(0), -1)
    #             next_q2_all = next_q2_map.view(next_q2_map.size(0), -1)
    #             min_next_q_all = torch.min(next_q1_all, next_q2_all) # [b,a_dim]
    #             v_next = torch.sum(next_probs * (min_next_q_all - self.alpha * next_log_probs), dim=1, keepdim=True) # [b,1]
    #             target_Q = mini_batch['reward'].unsqueeze(1) +  self.gamma * (1 - mini_batch['terminal']).unsqueeze(1) * v_next # [b, 1]

    #             # print("next_probs:", next_probs)
    #             # print("min_next_q_all:", min_next_q_all)
    #             # print("next_log_probs:", next_log_probs)

    #             print("next_q1_map:", next_q1_map)
    #             print("next_q2_map:", next_q2_map)


    #             print(mini_batch['reward'].unsqueeze(1))
    #             print(self.gamma * (1 - mini_batch['terminal']).unsqueeze(1) * v_next)
    #             print("v_next:", v_next)


    #             # print("target_Q:", target_Q)
    #             # print(mini_batch['reward'].unsqueeze(1))
    #             # print("v_next:", v_next)
    #             # print(self.gamma * (1 - mini_batch['terminal']).unsqueeze(1))

                


    #         '''Update soft Q net''' 
    #         q1_map, q2_map = self.q_critic(mini_batch['state']) 
    #         floor_idx = mini_batch['action'][:, 0]  # 维度: (B,)

    #         # 提取对应 [row, col] 的 Q 值
    #         batch_idx = torch.arange(mini_batch['state'].size(0), device=self.device)  # 维度: (B,)
    #         q1 = q1_map[batch_idx, floor_idx].unsqueeze(1)  # 维度: (B, 1)     
    #         q2 = q2_map[batch_idx, floor_idx].unsqueeze(1)  # 维度: (B, 1) 
    #         print("q1:", q1)
    #         print("q2:", q2)
            
    #         print("target_Q:", target_Q)

    #         print("F.mse_loss(q1, target_Q):", F.mse_loss(q1, target_Q))
    #         print("F.mse_loss(q2, target_Q):", F.mse_loss(q2, target_Q))

    #         q_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)

    #         print("q_loss:", q_loss)

    #         total_critic_loss += q_loss.mean().item()

    #         print("total_critic_loss:", total_critic_loss)

    #         print("======================================")

    #         self.q_critic_optimizer.zero_grad()
    #         q_loss.backward()
    #         self.q_critic_optimizer.step()

    #         #------------------------------------------ Update Target Net ----------------------------------#
    #         for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
    #             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    #         # Clean up memory
    #         del mini_batch, next_logits, next_probs, next_log_probs, next_q1_map, next_q2_map, next_q1_all, next_q2_all, min_next_q_all, v_next, target_Q, q1_map, q2_map, q1, q2, q_loss
    #         torch.cuda.empty_cache()
        
    #     return total_critic_loss / num_mini_batches


    def save(self, dir_path):
        torch.save(self.actor.state_dict(),  dir_path + "_actor")
        torch.save(self.actor_optimizer.state_dict(),  dir_path + "_actor_optimizer")

        torch.save(self.q_critic.state_dict(),  dir_path + "_critic")
        torch.save(self.q_critic_optimizer.state_dict(), dir_path + "_critic_optimizer")

        torch.save(self.q_critic_target.state_dict(),  dir_path + "_critic_target")

        '''
        alpha_dict = {"alpha": self.alpha, "log_alpha": self.log_alpha}   
        torch.save(alpha_dict, dir_path + "_alpha")     
        torch.save(self.alpha_optim.state_dict(), dir_path + "_alpha_optimizer")
        '''

    def save_critic(self, dir_path):
        torch.save(self.q_critic.state_dict(),  dir_path + "_critic")
        torch.save(self.q_critic_optimizer.state_dict(), dir_path + "_critic_optimizer")
        torch.save(self.q_critic_target.state_dict(),  dir_path + "_critic_target")
        
        '''
        alpha_dict = {"alpha": self.alpha, "log_alpha": self.log_alpha}   
        torch.save(alpha_dict, dir_path + "_alpha")     
        torch.save(self.alpha_optim.state_dict(), dir_path + "_alpha_optimizer")
        '''