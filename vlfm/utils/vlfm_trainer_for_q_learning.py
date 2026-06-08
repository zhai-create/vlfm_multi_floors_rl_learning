# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os

import math

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

from collections import defaultdict
from typing import Any, Dict, List

seed = 233

import numpy as np
import torch

np.random.seed(seed)
torch.manual_seed(seed)

import tqdm
import habitat
from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat_baselines import PPOTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
)
from habitat_baselines.rl.ddppo.algo import DDPPO  # noqa: F401.
from habitat_baselines.rl.ppo.single_agent_access_mgr import (  # noqa: F401.
    SingleAgentAccessMgr,
)
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import (
    extract_scalars_from_info as extract_scalars_from_info_habitat,
)
from omegaconf import OmegaConf

from torch.utils.tensorboard import SummaryWriter
import datetime

# from vlfm.policy.unet_model import HistoryAwareUNet

import random
random.seed(seed)

import copy

from vlfm.q_learning.arguments import args as q_args
from vlfm.q_learning.replay_buffer import ReplayBuffer
from vlfm.q_learning.rainbow_dqn import DQN

from tensorboard.backend.event_processing import event_accumulator

def extract_scalars_from_info(info: Dict[str, Any]) -> Dict[str, float]:
    info_filtered = {k: v for k, v in info.items() if not isinstance(v, list)}
    return extract_scalars_from_info_habitat(info_filtered)

def get_new_obs(obs):
    # 'base_explorer': array([1]) --> tensor([[3]], device='cuda:0')
    # 'compass': array([-1.8776e-09], dtype=float32) --> tensor([[-4.03073e-09]], device='cuda:0')
    # 'depth': (480, 640, 1) --> torch.Size([1, 480, 640, 1])
    # 'frontier_sensor': (1, 2) --> torch.Size([1, 1, 2])
    #  'gps': (2, ) --> torch.Size([1, 2])
    # 'heading': (1, ) --> torch.Size([1, 1])
    # 'rgb': (480, 640, 3) --> torch.Size([1, 480, 640, 3])
    # 'objectgoal': array([1]) --> tensor([[1]], device='cuda:0')

    obs['base_explorer'] = torch.tensor(obs['base_explorer'], device='cuda')  # unsqueeze(0) 添加一个批
    obs['compass'] = torch.tensor(obs['compass'], device='cuda')  # unsqueeze(0) 添加一个批
    obs['depth'] = torch.tensor(obs['depth'], device='cuda')  # unsqueeze(0) 添加一个批
    obs['frontier_sensor'] = torch.tensor(obs['frontier_sensor'], device='cuda')  # unsqueeze(0) 添加一个批
    obs['gps'] = torch.tensor(obs['gps'], device='cuda')  # unsqueeze(0) 添加一个批
    obs['heading'] = torch.tensor(obs['heading'], device='cuda')  # unsqueeze(0) 添加一个批
    obs['rgb'] = torch.tensor(obs['rgb'], device='cuda')  # unsqueeze(0) 添加一个批
    obs['objectgoal'] = torch.tensor(obs['objectgoal'], device='cuda')  # unsqueeze(0) 添加一个批
    return [obs]




@baseline_registry.register_trainer(name="vlfm")
class VLFMTrainer(PPOTrainer):

    # # load训练好的模型
    # policy_model = HistoryAwareUNet(in_channels=6).to("cuda")
    # pre_policy = 'checkpoints_single_v3_easy_data/epoch_5.pt'
    # checkpoint = torch.load(pre_policy, map_location="cuda")['model_state']
    # policy_model.load_state_dict(checkpoint)

    replay_buffer = ReplayBuffer(q_args)
    rl_policy = DQN(q_args)
    rl_policy.net.train()

    total_steps = 0  # Record the total steps during the training
    # total_steps = 54999  # Record the total steps during the training
    episode_rl_step_buffer = []

    '''
    log_root_hm3d = "log_files/log_for_dubug_final_q_learning_2025_08_07_05_44_56/"
    ea=event_accumulator.EventAccumulator(log_root_hm3d+os.listdir(log_root_hm3d)[0], size_guidance={'scalars': 100000}) 
    ea.Reload()
    rl_step_ls_1 = ea.scalars.Items('Result_RL/rl_step_reward')
    all_rl_step_ls_1 = [temp_num.value for temp_num in rl_step_ls_1 if(temp_num.step>=54001 and temp_num.step<=54999)]
    print(len(all_rl_step_ls_1))
    assert len(all_rl_step_ls_1)==q_args.delta_steps-1
    episode_rl_step_buffer = all_rl_step_ls_1
    '''
    
    epsilon_min = q_args.epsilon_min
    epsilon_decay = (q_args.epsilon_init - q_args.epsilon_min) / q_args.epsilon_decay_steps

    epsilon = q_args.epsilon_init

    # epsilon = q_args.epsilon_init-total_steps*epsilon_decay

    # 每个episode都要重新初始化
    current_state = None
    current_action = None
    next_state = None
    next_mask = None
    episode_reward = 0

    

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """

        date_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        # logger_file_name = "./log_files/log_0.2m_semantic_map_debug_train_"+date_time
        # logger_file_name = "./log_files/log_0.2m_il_train_data_"+date_time
        # logger_file_name = "./log_files/log_0.2m_il_train_data_"+date_time+"_init_905"
        # logger_file_name = "./log_files/log_0.2m_il_train_complete_data_new_weighting_data_train_dataset_for_eval_"+date_time+"_init_900"
        logger_file_name = "./log_files/log_for_dubug_final_q_learning_load_pretrain_"+date_time # 300
        writer = SummaryWriter(logger_file_name)


        # 如何循环同一个episode中的不同step：只有当一个episode结束的时候，stats_episodes才会更新，episode计数变量才会+1
        # 检查点加载：待检查
        # Some configurations require not to load the checkpoint, like when using
        # a hierarchial policy
        if self.config.habitat_baselines.eval.should_load_ckpt: # 默认为True
            # map_location="cpu" is almost always better than mapping to a CUDA device.
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
            step_id = ckpt_dict["extra_state"]["step"]
            print(step_id)
        else:
            ckpt_dict = {"config": None}

        # 加载config
        config = self._get_resume_state_config_or_new_config(ckpt_dict["config"])
        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split
            print("config.habitat_baselines.eval.split:",config.habitat_baselines.eval.split)
            print("\n\n\n\n\n\n")

        print("habitat.environment.max_episode_steps:", config.habitat.environment.max_episode_steps)


        # 视频选项（默认为空，不进入）
        if len(self.config.habitat_baselines.eval.video_option) > 0: 
            agent_config = get_agent_config(config.habitat.simulator)
            agent_sensors = agent_config.sim_sensors
            extra_sensors = config.habitat_baselines.eval.extra_sim_sensors
            with read_write(agent_sensors):
                agent_sensors.update(extra_sensors)
            with read_write(config):
                if config.habitat.gym.obs_keys is not None:
                    for render_view in extra_sensors.values():
                        if render_view.uuid not in config.habitat.gym.obs_keys:
                            config.habitat.gym.obs_keys.append(render_view.uuid)
                config.habitat.simulator.debug_render = True

        # 详细的日志记录
        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")
        
        habitat_env = habitat.Env(config=config)

        # =====> select episodes <=====
        selected_episodes = copy.deepcopy(habitat_env.episodes)
        random.shuffle(selected_episodes)
        # selected_episodes = selected_episodes[900:]
        # selected_episodes = selected_episodes[300:]
        selected_episodes = selected_episodes[:]
        # =====> select episodes <=====


        # # =====> select episodes <=====
        # id_dict = {
        #     "data/scene_datasets/hm3d_v0.2/train/00669-DNWbUAJYsPy/DNWbUAJYsPy.basis.glb":["tv_monitor", "bed", "sofa", "chair", "toilet"],
        #     "data/scene_datasets/hm3d_v0.2/train/00166-RaYrxWt5pR1/RaYrxWt5pR1.basis.glb":["tv_monitor", "toilet", "chair", "plant", "sofa"], 
        #     "data/scene_datasets/hm3d_v0.2/train/00404-QN2dRqwd84J/QN2dRqwd84J.basis.glb":["sofa", "bed", "plant", "tv_monitor", "toilet"], 
        #     "data/scene_datasets/hm3d_v0.2/train/00706-YHmAkqgwe2p/YHmAkqgwe2p.basis.glb":["bed", "toilet", "chair", "sofa"],
        #     "data/scene_datasets/hm3d_v0.2/train/00324-DoSbsoo4EAg/DoSbsoo4EAg.basis.glb":["bed", "tv_monitor"],

        #     "data/scene_datasets/hm3d_v0.2/train/00017-oEPjPNSPmzL/oEPjPNSPmzL.basis.glb":["bed", "tv_monitor", "toilet", "sofa", "plant"],
        #     "data/scene_datasets/hm3d_v0.2/train/00031-Wo6kuutE9i7/Wo6kuutE9i7.basis.glb":["bed", "tv_monitor", "toilet"],
        #     "data/scene_datasets/hm3d_v0.2/train/00099-226REUyJh2K/226REUyJh2K.basis.glb":["bed", "tv_monitor"],
        #     "data/scene_datasets/hm3d_v0.2/train/00105-xWvSkKiWQpC/xWvSkKiWQpC.basis.glb":["tv_monitor", "toilet", "sofa"],
        #     "data/scene_datasets/hm3d_v0.2/train/00250-U3oQjwTuMX8/U3oQjwTuMX8.basis.glb":["bed", "toilet", "sofa", "plant"],
        #     "data/scene_datasets/hm3d_v0.2/train/00251-wsAYBFtQaL7/wsAYBFtQaL7.basis.glb":["bed", "toilet", "sofa"],
        #     "data/scene_datasets/hm3d_v0.2/train/00254-YMNvYDhK8mB/YMNvYDhK8mB.basis.glb":["chair", "plant"],
        #     "data/scene_datasets/hm3d_v0.2/train/00255-NGyoyh91xXJ/NGyoyh91xXJ.basis.glb":["bed", "tv_monitor", "toilet"],
        #     "data/scene_datasets/hm3d_v0.2/train/00323-yHLr6bvWsVm/yHLr6bvWsVm.basis.glb":["bed", "tv_monitor", "toilet"],
        #     "data/scene_datasets/hm3d_v0.2/train/00327-xgLmjqzoAzF/xgLmjqzoAzF.basis.glb":["bed", "toilet", "chair"],
        # }

        # selected_episodes = []
        # for index, temp_episode in enumerate(habitat_env.episodes):
        #     if(temp_episode.scene_id in id_dict.keys()):
        #         if(temp_episode.object_category in id_dict[temp_episode.scene_id]):
        #             selected_episodes.append(temp_episode)

        # random.shuffle(selected_episodes)
        # # =====> select episodes <=====
        
        self._init_envs(config, is_eval=True) # 初始化env(暂时不管)

        self._agent = self._create_agent(None) # 初始化agent(暂时不管)
        action_shape, discrete_actions = get_action_space_info(self._agent.policy_action_space)  # 获取action_space形状+action_space是否离散

        # 加载智能体状态：待检查
        if self._agent.actor_critic.should_load_agent_state:
            self._agent.load_state_dict(ckpt_dict)

        index_in_episodes = 0
        habitat_env.episodes = [selected_episodes[index_in_episodes]]
        observations = habitat_env.reset()
        observations = get_new_obs(observations)

        # 待检查：多个act的执行顺序
        batch = batch_obs(observations, device=self.device) # 将来自多个环境的观察批量处理成单个张量，并将其移动到指定的 self.device
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore # 对批量观察应用任何观察转换（例如，归一化、调整大小）

        # # 临时代码
        # while True:
        #     index_in_episodes += 1
        #     habitat_env.episodes = [selected_episodes[index_in_episodes]]
        #     try:
        #         observations = habitat_env.reset()
        #     except:
        #         continue
        #     # observations = get_new_obs(observations)

        #     distance_to_goal = habitat_env.get_metrics()["distance_to_goal"]
        #     if(math.isinf(distance_to_goal)):
        #         continue
        #     else:
        #         writer.add_scalar("Result/distance_to_goal", distance_to_goal, index_in_episodes)
        # # 临时代码


        # # 临时代码
        # while True:
        #     index_in_episodes += 1
        #     habitat_env.episodes = [selected_episodes[index_in_episodes]]
        #     observations = habitat_env.reset()
        #     # observations = get_new_obs(observations)

        #     distance_to_goal = habitat_env.get_metrics()["distance_to_goal"]

        #     print("\n\n\n\n\n")
        #     print("~~~~~~~~~~~~~~~~~~~~~~~")
        #     print("index_in_episodes:", index_in_episodes)
        #     print(f"distance_to_goal: {distance_to_goal}, {math.isinf(distance_to_goal)}")
        #     print("==========> <==========")
        #     writer.add_scalar("Result/distance_to_goal", distance_to_goal, index_in_episodes)
        #     print("\n\n\n\n\n")

        #     if(index_in_episodes==27):
        #         breakpoint()

        #     # if math.isnan(habitat_env.get_metrics()["distance_to_goal"])==False:
        #     #     break
        # # 临时代码


        not_done_masks = torch.zeros(
            self.config.habitat_baselines.num_environments,
            1,
            device=self.device,
            dtype=torch.bool,
        ) # 指示并行环境中的每个回合是否仍在进行 （已完成为False）
        stats_episodes: Dict[Any, Any] = {}  # 存储每个已完成回合的详细统计信息 # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0) # 计算每个唯一的回合（由场景和回合 ID 标识）已被评估的次数

        rgb_frames: List[List[np.ndarray]] = [[] for _ in range(self.config.habitat_baselines.num_environments)] # 启用视频录制时存储每个环境的 RGB 帧
        if len(self.config.habitat_baselines.eval.video_option) > 0: # 暂时不管
            os.makedirs(self.config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = len(selected_episodes)
        evals_per_ep = self.config.habitat_baselines.eval.evals_per_ep # 每个episode的评估次数

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep) # 进度条
        self._agent.eval() # 禁用更新层

        from vlfm.utils.habitat_visualizer import HabitatVis

        num_successes = 0
        num_total = 0
        hab_vis = HabitatVis()
        all_spl_ls = []
        all_ne_ls = []
        _done_initializing = False


        while len(stats_episodes) < (number_of_eval_episodes * evals_per_ep):
            
            # =====> 初始化开始 <=====
            while (_done_initializing==False):
                action_data, _done_initializing = self._agent.actor_critic.act(
                    batch,
                    None,
                    None,
                    not_done_masks,
                    deterministic=False,
                ) # 关键的一步 (actor_critic是HabitatITMPolicyV2类型,实际调用HabitatMixin类中的act)
                step_data = [a.item() for a in action_data.env_actions.cpu()]

                 # 实际执行step_data
                observations = habitat_env.step(step_data[0])
                observations = get_new_obs(observations)
                infos = [habitat_env.get_metrics()]
                policy_infos = self._agent.actor_critic.get_extra(action_data, None, None)
                for i in range(len(policy_infos)):
                    infos[i].update(policy_infos[i]) # 将策略特定的信息合并到环境信息中

                # 新的 observations 被批量处理和转换，类似于初始设置，为下一次智能体的 act 调用准备 batch
                batch = batch_obs(  # type: ignore
                    observations,
                    device=self.device,
                ) 
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

                not_done_masks = torch.tensor(
                    [[not habitat_env.episode_over]],
                    dtype=torch.bool,
                    device="cpu",
                ) # 根据来自环境的 dones 标志进行更新

                n_envs = 1
                for i in range(n_envs):
                    if len(self.config.habitat_baselines.eval.video_option) > 0:
                        hab_vis.collect_data(batch, infos, action_data.policy_info)
                
                not_done_masks = not_done_masks.to(device=self.device) # 将 not_done_masks 张量移动到目标设备 (self.device)，因为它将在下一次迭代中作为智能体的输入 

                if(_done_initializing==True):
                    action_data, _done_initializing = self._agent.actor_critic.act(
                        batch,
                        None,
                        None,
                        not_done_masks,
                        deterministic=False,
                    ) # 得到next_state
                    VLFMTrainer.current_state = copy.deepcopy(VLFMTrainer.next_state)
                    episode_stats = {} # 创建一个字典来存储此回合的最终统计数据
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    last_geodesic_dis = episode_stats["distance_to_goal"]
                    break
            # =====> 初始化结束 <=====

            if(action_data is None): # 表示action_space不为空
                action_data = self._agent.actor_critic.select_action() # 得到action
            else:
                VLFMTrainer.current_action = None

            step_data = [a.item() for a in action_data.env_actions.cpu()]

            # 实际执行step_data
            observations = habitat_env.step(step_data[0])
            observations = get_new_obs(observations)
            infos = [habitat_env.get_metrics()]
            policy_infos = self._agent.actor_critic.get_extra(action_data, None, None)
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i]) # 将策略特定的信息合并到环境信息中

            # 新的 observations 被批量处理和转换，类似于初始设置，为下一次智能体的 act 调用准备 batch
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
            ) 
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not habitat_env.episode_over]],
                dtype=torch.bool,
                device="cpu",
            ) # 根据来自环境的 dones 标志进行更新

            n_envs = 1
            for i in range(n_envs):
                if len(self.config.habitat_baselines.eval.video_option) > 0:
                    hab_vis.collect_data(batch, infos, action_data.policy_info)

            if (VLFMTrainer.current_action is not None):
                temp_not_done_masks = torch.tensor(
                    [[True]],
                    dtype=torch.bool,
                    device="cpu",
                ) # 根据来自环境的 dones 标志进行更新

                action_data, _done_initializing = self._agent.actor_critic.act(
                    batch,
                    None,
                    None,
                    temp_not_done_masks,
                    deterministic=False,
                ) # 得到next_state
            else:
                VLFMTrainer.next_state = None # 得到next_state


            for i in range(n_envs):
                # episode ended
                if not not_done_masks[i].item(): # 如果环境 i 中的一个回合已结束
                    pbar.update() # 更新进度条
                    episode_stats = {} # 创建一个字典来存储此回合的最终统计数据
                    episode_stats.update(extract_scalars_from_info(infos[i])) # 用于从 infos 字典中提取标量指标（例如，success、spl、distance_to_goal）
                    k = (
                        habitat_env.current_episode.scene_id,
                        habitat_env.current_episode.episode_id,
                    )
                    ep_eval_count[k] += 1 # 增加此特定回合实例的评估计数
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if episode_stats["success"] == 1:
                        num_successes += 1
                        rl_step_reward = 2.5
                    else:
                        now_geodesic_dis = episode_stats["distance_to_goal"]
                        rl_step_reward = -(now_geodesic_dis-last_geodesic_dis)-0.01

                    last_geodesic_dis = now_geodesic_dis
                    VLFMTrainer.episode_reward += rl_step_reward

                    num_total += 1
                    print(f"Success rate: {num_successes / num_total * 100:.2f}% ({num_successes} out of {num_total})")

                    all_spl_ls.append(episode_stats["spl"])
                    all_ne_ls.append(episode_stats["distance_to_goal"])

                    # real_index = num_total+900
                    # real_index = num_total+300
                    real_index = num_total
                    # episode_stats: dict_keys(['reward', 'distance_to_goal', 'success', 'spl', 'soft_spl', 'distance_to_goal_reward', 'traveled_stairs', 'yaw', 'target_detected', 'stop_called'])
                    writer.add_scalar("Result/distance_to_goal", episode_stats["distance_to_goal"], real_index)
                    writer.add_scalar("Result/success_num", num_successes, real_index)
                    writer.add_scalar("Result/spl_per_episode", episode_stats["spl"], real_index)
                    writer.add_scalar("Result/stop_called", episode_stats["stop_called"], real_index)
                    writer.add_scalar("Result/spl_mean", np.mean(all_spl_ls), real_index)
                    writer.add_scalar("Result/ne_mean", np.mean(all_ne_ls), real_index)

                    writer.add_scalar("Result_RL/episode_reward", VLFMTrainer.episode_reward, real_index)
                    # writer.add_scalar('Scene/scene_id', list(id_dict.keys()).index(habitat_env.episodes[0].scene_id), index_in_episodes+1)
                    
                    

                    from vlfm.utils.episode_stats_logger import (
                        log_episode_stats,
                    )


                    # 确定failure_cause
                    try:
                        failure_cause = log_episode_stats(
                            habitat_env.current_episode.episode_id,
                            habitat_env.current_episode.scene_id,
                            infos[i],
                        )
                    except Exception:
                        failure_cause = "Unknown"

                    # 视频选项（暂时不管）
                    if len(self.config.habitat_baselines.eval.video_option) > 0:
                        rgb_frames[i] = hab_vis.flush_frames(failure_cause)
                        generate_video(
                            video_option=self.config.habitat_baselines.eval.video_option,
                            video_dir=self.config.habitat_baselines.video_dir,
                            images=rgb_frames[i],
                            # episode_id=current_episodes_info[i].episode_id,
                            episode_id=real_index,
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(infos[i]),
                            fps=self.config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        rgb_frames[i] = []

                    # 如果 infos 中存在 GfxReplayMeasure（Habitat 中用于可视化重播回合的功能）数据，则将其写入文件
                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            self.config.habitat.task,
                            habitat_env.current_episode.episode_id,
                        )    
                else:
                    episode_stats = {} # 创建一个字典来存储此回合的最终统计数据
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    now_geodesic_dis = episode_stats["distance_to_goal"]

                    rl_step_reward = -(now_geodesic_dis-last_geodesic_dis)-0.01
                    last_geodesic_dis = now_geodesic_dis
                    VLFMTrainer.episode_reward += rl_step_reward

                if(VLFMTrainer.current_action is not None): # action_space为空，不加入buffer中，不记录rl_reward，记录episode_reward
                    VLFMTrainer.total_steps += 1
                    VLFMTrainer.epsilon = VLFMTrainer.epsilon - VLFMTrainer.epsilon_decay if VLFMTrainer.epsilon - VLFMTrainer.epsilon_decay > VLFMTrainer.epsilon_min else VLFMTrainer.epsilon_min
                    if (not not_done_masks[i].item()) and (episode_stats["stop_called"]==1):
                        terminal = True
                    else:
                        terminal = False

                    if (not not_done_masks[i].item()):
                        temp_done = True
                    else:
                        temp_done = False

                    VLFMTrainer.replay_buffer.store_transition(VLFMTrainer.current_state, VLFMTrainer.current_action, rl_step_reward, VLFMTrainer.next_state, terminal, temp_done)
                    VLFMTrainer.current_state = copy.deepcopy(VLFMTrainer.next_state)

                    VLFMTrainer.episode_rl_step_buffer.append(rl_step_reward)

                    '''
                    if VLFMTrainer.replay_buffer.current_size >= q_args.batch_size: # 暂时用每个step训练一次
                        VLFMTrainer.rl_policy.learn(VLFMTrainer.replay_buffer, VLFMTrainer.total_steps)
                        # 在此处记录一个delta_rl_step的reward均值，按照delta_rl_step保存权重参数
                    '''

                    if ((VLFMTrainer.total_steps%q_args.delta_steps)==0):
                        update_step = (VLFMTrainer.total_steps//q_args.delta_steps)
                        total_loss_per_update = VLFMTrainer.rl_policy.learn(VLFMTrainer.replay_buffer, VLFMTrainer.total_steps)
                        assert len(VLFMTrainer.episode_rl_step_buffer)==q_args.delta_steps
                        mean_episode_rl_step_buffer = np.mean(VLFMTrainer.episode_rl_step_buffer)
                        writer.add_scalar("Result_RL/mean_episode_rl_step_buffer", mean_episode_rl_step_buffer, update_step)
                        writer.add_scalar("Result_RL/total_loss_per_update", total_loss_per_update, update_step)

                        VLFMTrainer.episode_rl_step_buffer = []  

                        # =====> 创建新目录 <=====
                        model_pre_dir = '{0}/{1}/policy/{2}'.format(q_args.root, q_args.model_file_name, q_args.experiment_details)
                        if not os.path.exists(model_pre_dir):
                            os.makedirs(model_pre_dir)
                            print(f"The new path:'{model_pre_dir}' has beed craeted!")
                        # =====> 创建新目录 <=====

                        VLFMTrainer.rl_policy.save('{0}/{1}/policy/{2}/{3}'.format(q_args.root, q_args.model_file_name, q_args.experiment_details, update_step))


                    writer.add_scalar("Result_RL/rl_step_reward", rl_step_reward, VLFMTrainer.total_steps)

            not_done_masks = not_done_masks.to(device=self.device) # 将 not_done_masks 张量移动到目标设备 (self.device)，因为它将在下一次迭代中作为智能体的输入 
            if not not_done_masks[0].item(): # 如果当前回合已经结束
                while True:
                    index_in_episodes += 1
                    habitat_env.episodes = [selected_episodes[index_in_episodes]]
                    try:
                        observations = habitat_env.reset()
                    except:
                        continue
                    observations = get_new_obs(observations)

                    if math.isinf(habitat_env.get_metrics()["distance_to_goal"])==False:
                        break


                # 待检查：多个act的执行顺序
                batch = batch_obs(observations, device=self.device) # 将来自多个环境的观察批量处理成单个张量，并将其移动到指定的 self.device
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore # 对批量观察应用任何观察转换（例如，归一化、调整大小）

                not_done_masks = torch.zeros(
                    self.config.habitat_baselines.num_environments,
                    1,
                    device=self.device,
                    dtype=torch.bool,
                ) # 指示并行环境中的每个回合是否仍在进行 （已完成为False）

                VLFMTrainer.current_state = None
                VLFMTrainer.current_action = None
                VLFMTrainer.next_state = None
                VLFMTrainer.episode_reward = 0
                _done_initializing = False
                

        # 评估结束
        pbar.close() # 关闭 tqdm 进度条

        # 发出此评估过程已完成的信号
        if "ZSOS_DONE_PATH" in os.environ:
            # Create an empty file at ZSOS_DONE_PATH to signal that the
            # evaluation is done
            done_path = os.environ["ZSOS_DONE_PATH"]
            with open(done_path, "w") as f:
                f.write("")

        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        # 聚合统计数据
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = np.mean([v[stat_key] for v in stats_episodes.values()])

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar("eval_reward/average_reward", aggregated_stats["reward"], step_id)

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

