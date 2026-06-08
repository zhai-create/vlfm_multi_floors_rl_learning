from vlfm.arguments import args as main_args

import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(main_args.card_select)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = str(main_args.card_select)


import torch
import numpy as np
import copy
# from network import Dueling_Net, Net
from vlfm.policy.unet_model import HistoryAwareUNet

def save_models(net, target_net, optimizer, name, dir_path):
    torch.save(net.state_dict(), dir_path + "_" + name)
    torch.save(target_net.state_dict(), dir_path + "_" + name + "_target")
    torch.save(optimizer.state_dict(), dir_path + "_" + name + "_optimizer")

class DQN(object):
    def __init__(self, args):
        self.batch_size = args.batch_size  # batch size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr  # learning rate
        self.gamma = args.gamma  # discount factor
        self.tau = args.tau  # Soft update
        self.use_soft_update = args.use_soft_update
        self.target_update_freq = args.target_update_freq  # hard update
        self.update_count = 0

        self.grad_clip = args.grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_double = args.use_double
        self.use_dueling = args.use_dueling
        self.use_per = args.use_per
        self.use_n_steps = args.use_n_steps
        self.device =  args.device
        if self.use_n_steps:
            self.gamma = self.gamma ** args.n_steps

        # if self.use_dueling:  # Whether to use the 'dueling network'
        #     self.net = Dueling_Net(args)
        # else:
        #     self.net = Net(args)


        self.net = HistoryAwareUNet(in_channels=6).to(args.device)
        # load训练好的模型(il)
        checkpoint = torch.load(args.pre_policy, map_location=args.device)['model_state']
        self.net.load_state_dict(checkpoint)

        # # load训练好的模型(rl)
        # checkpoint = torch.load("Models_train_PPO/policy/init_q_learning/54_actor")
        # self.net.load_state_dict(checkpoint)

        self.target_net = copy.deepcopy(self.net)  # Copy the online_net to the target_net
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)

        # # load训练好的optimizer(rl)
        # self.optimizer.load_state_dict(torch.load("Models_train_PPO/policy/init_q_learning/54_actor_optimizer"))

    def choose_action(self, state, epsilon):
        with torch.no_grad():
            semantic_frontier_map = state[3] 
            mask_map = (semantic_frontier_map > 0).astype(np.float32)

            if np.random.uniform() > epsilon:
                net_state = torch.FloatTensor(state)
                net_state = net_state.unsqueeze(0).to(self.device)
                preds = self.net(net_state)
                preds = torch.sigmoid(preds)
                pred_map = preds[0, 0].detach().cpu().numpy()
                # mask_map = curr_mask[0].cpu().numpy()
                masked_pred = pred_map.copy()
                masked_pred[mask_map < 0.5] = 0

                pred_row, pred_col = np.unravel_index(masked_pred.argmax(), masked_pred.shape)
            else:
                # mask_map = curr_mask[0].cpu().numpy()
                ones_positions = np.argwhere(mask_map == 1)
                random_index = np.random.choice(len(ones_positions))
                selected_position = ones_positions[random_index]
                pred_row, pred_col = selected_position[0], selected_position[1]

            action = np.array([pred_row, pred_col])
            # action = torch.LongTensor(action)
            # action = action.unsqueeze(0).to(self.device)
            return action, pred_row, pred_col


    # def learn(self, replay_buffer, total_steps):
    #     batch, batch_index, IS_weight = replay_buffer.sample(total_steps)

    #     with torch.no_grad():  # q_target has no gradient
    #         # Use online_net to select the action
    #         a_argmax = self.net(batch['next_state']).argmax(dim=-1, keepdim=True)  # shape：(batch_size,1)
    #         # Use target_net to estimate the q_target
    #         q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * self.target_net(batch['next_state']).gather(-1, a_argmax).squeeze(-1)  # shape：(batch_size,)


    #     q_current = self.net(batch['state']).gather(-1, batch['action']).squeeze(-1)  # shape：(batch_size,)
    #     td_errors = q_current - q_target  # shape：(batch_size,)
    #     loss = (td_errors ** 2).mean()

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     if self.grad_clip: # 暂时用
    #         torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
    #     self.optimizer.step()

    #     if self.use_soft_update:  # soft update # 暂时用这个
    #         for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
    #             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    #     else:  # hard update
    #         self.update_count += 1
    #         if self.update_count % self.target_update_freq == 0:
    #             self.target_net.load_state_dict(self.net.state_dict())

    #     if self.use_lr_decay:  # learning rate Decay # 暂时用
    #         self.lr_decay(total_steps)

    def learn(self, replay_buffer, total_steps):
        batch, batch_index, IS_weight = replay_buffer.sample(total_steps)
        mini_batch_size = 8
        num_mini_batches = len(batch['state']) // mini_batch_size  # Should be 8 for batch size 64

        total_loss = 0.0
        self.optimizer.zero_grad()

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

            with torch.no_grad():  # q_target has no gradient
                preds = self.net(mini_batch['next_state'])
                preds = torch.sigmoid(preds)

                a_argmax = []
                for i in range(preds.size(0)):
                    pred_map = preds[i, 0].detach().cpu().numpy()
                    semantic_frontier_map = mini_batch['next_state'][i, 3].cpu().numpy() 
                    mask_map = (semantic_frontier_map > 0).astype(np.float32)
                    
                    masked_pred = pred_map.copy()
                    masked_pred[mask_map < 0.5] = 0

                    pred_row, pred_col = np.unravel_index(masked_pred.argmax(), masked_pred.shape)

                    a_argmax.append([pred_row, pred_col])
                
                a_argmax = np.array(a_argmax)
                a_argmax = torch.LongTensor(a_argmax).to(self.device) # shape：(batch_size,2)
                
                # # Use online_net to select the action
                # a_argmax = self.net(mini_batch['next_state']).argmax(dim=-1, keepdim=True)  # shape：(batch_size,2)
                
                # =====> <=====
                next_preds = self.target_net(mini_batch['next_state'])
                next_preds = torch.sigmoid(next_preds)

                # 将 a.argmax 分解为 row 和 col 索引
                row_idx = a_argmax[:, 0]  # 维度: (B,)
                col_idx = a_argmax[:, 1]  # 维度: (B,)
                
                # 提取对应 [row, col] 的 Q 值
                batch_idx = torch.arange(next_preds.size(0), device=self.device)  # 维度: (B,)
                q_future = next_preds[batch_idx, 0, row_idx, col_idx]  # 维度: (B,)
                # =====> <=====

                q_target = mini_batch['reward'] + self.gamma * (1 - mini_batch['terminal']) * q_future  # shape：(batch_size,)
                


                # # Use target_net to estimate the q_target
                # q_target = mini_batch['reward'] + self.gamma * (1 - mini_batch['terminal']) * self.target_net(mini_batch['next_state']).gather(-1, a_argmax).squeeze(-1)  # shape：(batch_size,)


            # q_current = self.net(mini_batch['state']).gather(-1, mini_batch['action']).squeeze(-1)  # shape：(batch_size,)
            
            curr_preds = self.net(mini_batch['state'])
            curr_preds = torch.sigmoid(curr_preds)      

            # 将 a.argmax 分解为 row 和 col 索引
            row_idx = mini_batch['action'][:, 0]  # 维度: (B,)
            col_idx = mini_batch['action'][:, 1]  # 维度: (B,)   

            # 提取对应 [row, col] 的 Q 值
            batch_idx = torch.arange(curr_preds.size(0), device=self.device)  # 维度: (B,)
            q_current = curr_preds[batch_idx, 0, row_idx, col_idx]  # 维度: (B,)   
            
            td_errors = q_current - q_target  # shape：(batch_size,)
            loss = (td_errors ** 2).mean()
            total_loss += loss.item()

            # for param in self.net.parameters():
            #     print(param.requires_grad)

            # for param in self.target_net.parameters():
            #     print(param.requires_grad)

            loss.backward()  # Accumulate gradients

            # Clean up memory
            del mini_batch, preds, next_preds, a_argmax, batch_idx, curr_preds, q_future, q_current, q_target, td_errors, loss, row_idx, col_idx
            torch.cuda.empty_cache()

        if self.grad_clip: # 暂时用
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.use_soft_update:  # soft update # 暂时用这个
            for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:  # hard update
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.net.state_dict())

        if self.use_lr_decay:  # learning rate Decay # 暂时用
            self.lr_decay(total_steps)
        
        return total_loss / num_mini_batches

    def lr_decay(self, total_steps):
        lr_now = 0.9 * self.lr * (1 - total_steps / self.max_train_steps) + 0.1 * self.lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

    def save(self, dir_path):
        save_models(self.net, self.target_net, self.optimizer, "actor", dir_path)
