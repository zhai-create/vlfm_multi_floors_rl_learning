# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
from vlfm.arguments import args as main_args

import random
# random.seed(233)
# random.seed(42)
random.seed(84)
# random.seed(42)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(main_args.card_select)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = str(main_args.card_select)

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
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
import time

from vlfm.policy.unet_model import HistoryAwareUNet
from vlfm.policy.unet_model_multi_floors import MultiFloorNavigator

import gzip
import json
import math

import copy

import math

import quaternion

from vlfm.sac_intra.arguments import args as q_args
from vlfm.sac_intra.SACD import SACD_agent

# from vlfm.mapping.object_point_cloud_map import HabitatAction


# def transform_point(point_old):
#     """
#     将点从旧坐标系转换到新坐标系
    
#     参数:
#     point_old: 旧坐标系中的点 (x, y)
    
#     返回:
#     新坐标系中的点 (x, y)
#     """
#     old_origin_in_new = (3.40437, 9.08529)
#     new_x_rotation = 30
#     rotation_angle = math.radians(new_x_rotation)
#     rotation_matrix = np.array([
#             [math.cos(-rotation_angle), -math.sin(-rotation_angle)],
#             [math.sin(-rotation_angle), math.cos(-rotation_angle)]
#         ])
#     point_old = np.array(point_old)
    
    
#     # 坐标转换步骤：
#     # 1. 将点平移到以旧系原点为参考
#     # 2. 应用旋转矩阵
#     # 3. 平移到新系坐标系
#     point_new = rotation_matrix @ point_old + old_origin_in_new
    
#     return tuple(point_new)

def transform_point(robot_start_pos, robot_start_rot, point_old):
    """
    将点从旧坐标系转换到新坐标系
    
    参数:
    point_old: 旧坐标系中的点 (x, y)
    
    返回:
    新坐标系中的点 (x, y)
    """
    
    rotation_angle = robot_start_rot # 直接给弧度制的角度
    rotation_matrix = np.array([
            [math.cos(rotation_angle), -math.sin(rotation_angle)],
            [math.sin(rotation_angle), math.cos(rotation_angle)]
        ])


    point_old = np.array(point_old)
    robot_start_pos = np.array(robot_start_pos)
    
    # 坐标转换步骤：
    # 1. 将点相对于新系原点（在旧系中的表示）
    relative_vector = point_old - robot_start_pos
    relative_vector[0] *= -1
    
    # 2. 应用旋转矩阵（从旧系旋转到新系）
    point_new = rotation_matrix @ relative_vector

    point_new[1] *= -1
    
    return point_new



def extract_scalars_from_info(info: Dict[str, Any]) -> Dict[str, float]:
    info_filtered = {k: v for k, v in info.items() if not isinstance(v, list)}
    return extract_scalars_from_info_habitat(info_filtered)

def get_new_obs(obs):

    obs['base_explorer'] = torch.tensor(obs['base_explorer'], device='cuda')  # unsqueeze(0) 添加一个批
    obs['compass'] = torch.tensor(obs['compass'], device='cuda')  # unsqueeze(0) 添加一个批
    obs['depth'] = torch.tensor(obs['depth'], device='cuda')  # unsqueeze(0) 添加一个批
    obs['frontier_sensor'] = torch.tensor(obs['frontier_sensor'], device='cuda')  # unsqueeze(0) 添加一个批
    obs['gps'] = torch.tensor(obs['gps'], device='cuda')  # unsqueeze(0) 添加一个批
    obs['heading'] = torch.tensor(obs['heading'], device='cuda')  # unsqueeze(0) 添加一个批
    obs['rgb'] = torch.tensor(obs['rgb'], device='cuda')  # unsqueeze(0) 添加一个批
    obs['objectgoal'] = torch.tensor(obs['objectgoal'], device='cuda')  # unsqueeze(0) 添加一个批
    
    obs['semantic'] = torch.tensor(obs['semantic'], device='cuda')  # unsqueeze(0) 添加一个批

    return [obs]

def find_max_number_in_filenames(folder_path):
    """
    查找模型权重文件中的最大数字
    
    参数:
        folder_path (str): 包含.npy文件的文件夹路径
    
    返回:
        int: 文件名中的最大数字
    """

    if(len(os.listdir(folder_path))==0):
        return 0


    max_number = max([int(filename.split("_")[0]) for filename in os.listdir(folder_path) if (len(filename.split("_"))==2)])    
    return max_number

def load_total_steps(now_dir='main_process_info/step_rewards/'):
    return len(os.listdir(now_dir))

import time
all_time_ls = []

def store_transition(state, action, reward, next_state, terminal, done, now_total_step):
    temp_buffer = {}
    save_dir = q_args.buffer_data_folder

    temp_buffer['state'] = state
    temp_buffer['action'] = action
    temp_buffer['reward'] = reward
    temp_buffer['next_state'] = next_state
    temp_buffer['terminal'] = terminal
    
    len_dir = len(os.listdir(save_dir))
    if len_dir==q_args.buffer_capacity:
        # now_count = (now_total_step-5000) % q_args.buffer_capacity
        now_count = now_total_step % q_args.buffer_capacity
    else:
        now_count = len_dir+1

    if(now_count==0):
        np.save('{}/{}.npy'.format(save_dir, q_args.buffer_capacity), temp_buffer)
    else:
        np.save('{}/{}.npy'.format(save_dir, now_count), temp_buffer)

    all_time_ls.append(time.time())

    print("all_time_ls_length:", len(all_time_ls))
    print("all_delta_time:", all_time_ls[-1]-all_time_ls[0])

    if(now_count>=101):
        breakpoint()


    # if(now_count>5002):
    #     breakpoint()

def is_on_same_floor(height, ref_floor_height=None, ceiling_height=2.0, episode=None):
    """
    Check if a position is on the same floor as a reference height

    Args:
        height (float): Height to check
        ref_floor_height (float, optional): Reference floor height. Uses episode start if None
        ceiling_height (float): Height of ceiling (default: 2.0m)
        episode: Episode object containing start position

    Returns:
        int: 1 if on same floor, 0 otherwise
    """
    if ref_floor_height is None:
        ref_floor_height = episode.start_position[1]
    if ref_floor_height <= height < ref_floor_height + ceiling_height:
        return 1
    else:
        return 0

def get_absolute_pos(p_loc, r_loc, rr):
    r_matrix = np.array([[np.cos(rr), np.sin(rr)], [-np.sin(rr), np.cos(rr)]])
    return r_loc + np.dot(r_matrix, p_loc)

def get_rgb_image_ls(env):
    """
    Get the image list
    :param env: habitat_env
    :return image_ls: [rgb_1, rgb_2, rgb_3, rgb_4]
    """
    #     1
    # 2        4
    #     3
    rgb_ls = []
    depth_ls = []
    semantic_ls = []
    turn_id_ls = range(2, 5)


    for turn_id in turn_id_ls:
        if(turn_id==2):
            dis, angle = 0, 0.5*np.pi
            p_ref_loc = np.array([dis*np.sin(angle), dis*np.cos(angle)])
            state = env._sim.get_agent_state(0)
            translation = state.position # 1: right; 2: up; 3: back
            rotation = state.rotation # anti-clockwise / up-righthand principle
            euler = quaternion.as_euler_angles(rotation)
            if euler[0]!=0:
                euler[1] = 2*euler[0]-euler[1]
                euler[0] = 0
                euler[2] = 0
            euler[1]+=0.5*np.pi
            ref_true_loc = np.array([-translation[2], translation[0]])
            ref_true_dir = euler[1]
            rotation = quaternion.from_euler_angles(euler)
            p_true_loc = get_absolute_pos(p_ref_loc, ref_true_loc, ref_true_dir)
            goal_position = np.array([p_true_loc[1], translation[1], -p_true_loc[0]])
            obs = env._sim.get_observations_at(position=goal_position, rotation=rotation, keep_agent_at_new_pose=False)
            rgb = torch.Tensor(obs["rgb"])
            depth = torch.Tensor(obs["depth"])
            semantic = torch.Tensor(obs["semantic"])
        
        elif(turn_id==3):
            dis, angle = 0, np.pi
            p_ref_loc = np.array([dis*np.sin(angle), dis*np.cos(angle)])
            state = env._sim.get_agent_state(0)
            translation = state.position # 1: right; 2: up; 3: back
            rotation = state.rotation # anti-clockwise / up-righthand principle
            euler = quaternion.as_euler_angles(rotation)
            if euler[0]!=0:
                euler[1] = 2*euler[0]-euler[1]
                euler[0] = 0
                euler[2] = 0
            euler[1]+=np.pi
            ref_true_loc = np.array([-translation[2], translation[0]])
            ref_true_dir = euler[1]
            rotation = quaternion.from_euler_angles(euler)
            p_true_loc = get_absolute_pos(p_ref_loc, ref_true_loc, ref_true_dir)
            goal_position = np.array([p_true_loc[1], translation[1], -p_true_loc[0]])
            obs = env._sim.get_observations_at(position=goal_position, rotation=rotation, keep_agent_at_new_pose=False)
            rgb = torch.Tensor(obs["rgb"])
            depth = torch.Tensor(obs["depth"])
            semantic = torch.Tensor(obs["semantic"])
        
        elif(turn_id==4):
            dis, angle = 0, -0.5*np.pi
            p_ref_loc = np.array([dis*np.sin(angle), dis*np.cos(angle)])
            state = env._sim.get_agent_state(0)
            translation = state.position # 1: right; 2: up; 3: back
            rotation = state.rotation # anti-clockwise / up-righthand principle
            euler = quaternion.as_euler_angles(rotation)
            if euler[0]!=0:
                euler[1] = 2*euler[0]-euler[1]
                euler[0] = 0
                euler[2] = 0
            euler[1]-=0.5*np.pi
            ref_true_loc = np.array([-translation[2], translation[0]])
            ref_true_dir = euler[1]
            rotation = quaternion.from_euler_angles(euler)
            p_true_loc = get_absolute_pos(p_ref_loc, ref_true_loc, ref_true_dir)
            goal_position = np.array([p_true_loc[1], translation[1], -p_true_loc[0]])
            obs = env._sim.get_observations_at(position=goal_position, rotation=rotation, keep_agent_at_new_pose=False)
            rgb = torch.Tensor(obs["rgb"])
            depth = torch.Tensor(obs["depth"])
            semantic = torch.Tensor(obs["semantic"])

        rgb_ls.append(rgb)
        depth_ls.append(depth)
        semantic_ls.append(semantic)
    return rgb_ls, depth_ls, semantic_ls

def get_stairs_id(habitat_env):
    current_scene_dir = "data/scene_datasets/hm3d/train/"+habitat_env.current_episode.scene_id.split('/')[-2]+"/"
    current_scene_file = current_scene_dir+[temp_file for temp_file in os.listdir(current_scene_dir) if ".semantic.txt" in temp_file][0]
    with open(current_scene_file, 'r', encoding='utf-8') as f:
        all_info_lines = f.readlines()
    res_all_stairs_id = []
    for temp_line in all_info_lines:
        if('HM3D Semantic Annotations' in temp_line):
            continue
        if (temp_line.split(',')[2].strip('"')=="stairs"):
            res_all_stairs_id.append(eval(temp_line.split(',')[0]))
    return res_all_stairs_id


@baseline_registry.register_trainer(name="vlfm")
class VLFMTrainer(PPOTrainer):
    envs: VectorEnv

    habitat_env = None

    # 以下为多个进程共享
    rl_policy = SACD_agent(q_args)
    rl_policy.actor.train()
    rl_policy.q_critic.train()

    # # 以下为多个进程共享
    # floor_policy = SACD_agent(q_args)
    # floor_policy.actor.train()
    # floor_policy.q_critic.train()

    # 每个episode都要重新初始化
    current_state = None
    current_action = None
    next_state = None
    episode_reward = 0
    has_beed_initialize = False
    floor_select_steps = 0
    last_floor_select_steps = 0

    current_floor_steps = 0
    last_current_floor_steps = 0


    stop_call = 0

    # now_model_id = 0 # critic_sample暂时不要
    now_model_id = 408 # critic_sample暂时不要
    # now_model_id = 816 # critic_sample暂时不要
    # now_model_id = 989 # critic_sample暂时不要

    current_episode_object_id = []
    rgb_ls, depth_ls, semantic_ls = [], [], []


    # # 2. load 本层决策模型
    # # load训练好的RL模型
    # policy_model = HistoryAwareUNet(in_channels=6).to("cuda")
    # policy_model.load_state_dict(torch.load("Models_train_PPO_one_floor/policy/multi_process_sac/1002_actor"))
    # policy_model.eval()

    # 2. load 楼层决策模型
    # load训练好的RL模型
    floor_policy = MultiFloorNavigator(pretrained_path='Models_train_PPO_inter/policy/multi_process_sac/1002_actor').to("cuda")
    floor_policy.load_state_dict(torch.load("Models_train_PPO_inter/policy/multi_process_sac/196_actor"))
    floor_policy.eval()   

    res_all_stairs_id = []

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
        # logger_file_name = f"./log_files/log_ascent_actor_gt_perception_multi_stairs_RL_train_sample_data_frozen_model_18_sample_data_no_nan_process_id_{main_args.process_id}_real_train_"+date_time
        logger_file_name = f"./log_files_cross/log_process_id_{main_args.process_id}_huawei_"+date_time
        # logger_file_name = f"./log_files_cross/log_process_id_{main_args.process_id}_real_train_based_816_actor_"+date_time
        writer = SummaryWriter(logger_file_name)

        if self._is_distributed: # 分布式检查（暂时不支持，不用管）
            raise RuntimeError("Evaluation does not support distributed mode")

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
            config.habitat_baselines.eval.split = "train"
            config.habitat.dataset.split = config.habitat_baselines.eval.split
            config.habitat.dataset.data_path = "data/datasets/objectnav/hm3d/v1/train/train.json.gz" # 暂时用V1训练

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

        VLFMTrainer.habitat_env = habitat.Env(config=config)

        # =====> select episodes <=====
        selected_episodes = VLFMTrainer.habitat_env.episodes
        random.shuffle(selected_episodes)
        selected_episodes = selected_episodes[main_args.process_id::10]
        # selected_episodes = selected_episodes[150:] # 0-->5 # 中断修改 # 1
        # selected_episodes = selected_episodes[80:] # 0-->5 # 4

        # selected_episodes = selected_episodes[400:] # 0-->5 # 1
        # selected_episodes = selected_episodes[2000:] # 0-->5 # 1
        # selected_episodes = selected_episodes[4000:] # 0-->5 # 1


        # selected_episodes = selected_episodes[20:] # 0-->5  # 4
        # selected_episodes = selected_episodes[25:] # 0-->5  # 7， 8， 10
        # selected_episodes = selected_episodes[100:] # 0-->5  # 3
        # selected_episodes = selected_episodes[70:] # 0-->5  # 9
        # selected_episodes = selected_episodes[110:] # 0-->5  # 13
        # selected_episodes = selected_episodes[150:] # 0-->5  # 0, 1, 10
        # selected_episodes = selected_episodes[200:] # 0-->5  # 0，3，15
        # selected_episodes = selected_episodes[300:] # 0-->5  # 0，1, 3, 5, 6, 7, 8, 9, 15
        # selected_episodes = selected_episodes[350:] # 0-->5  # 2
        # selected_episodes = selected_episodes[350:] # 0-->5  # 0
        # selected_episodes = selected_episodes[:]
        # =====> select episodes <=====

        self._init_envs(config, is_eval=True) # 初始化env(暂时不管)
        self._agent = self._create_agent(None) # 初始化agent(暂时不管)
        action_shape, discrete_actions = get_action_space_info(self._agent.policy_action_space)  # 获取action_space形状+action_space是否离散

        # 加载智能体状态：待检查
        if self._agent.actor_critic.should_load_agent_state:
            self._agent.load_state_dict(ckpt_dict)

        index_in_episodes = -1

        while True:
            index_in_episodes += 1
            VLFMTrainer.habitat_env.episodes = [selected_episodes[index_in_episodes]]
            # 1. 判断当前episode是否可以reset
            try:
                observations = VLFMTrainer.habitat_env.reset()
            except:
                continue

            # =====> 新增object_id <=====
            try:
                classes_ls = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]
                scene_name = VLFMTrainer.habitat_env.current_episode.scene_id.split('/')[-2].split('-')[1]
                file_path = f'data/datasets/objectnav/hm3d/v1/train/content/{scene_name}.json.gz' 
                temp_object_goal = classes_ls[observations['objectgoal'][0]]

                # 使用 'rt' 模式（读取文本）打开 gzip 文件
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    # json.load() 直接从文件对象中读取并解析 JSON 数据
                    json_data = json.load(f)
                VLFMTrainer.current_episode_object_id = [temp_data["object_id"] for temp_data in json_data['goals_by_category'][scene_name+f".basis.glb_{temp_object_goal}"]]
            
            except:
                print("doing continueeeeeeeee")
                continue
            # =====> 新增object_id <=====
        
            observations = get_new_obs(observations)
            # if (math.isinf(VLFMTrainer.habitat_env.get_metrics()["distance_to_goal"])==False):
            if math.isfinite(VLFMTrainer.habitat_env.get_metrics()["distance_to_goal"]):
                break

        VLFMTrainer.res_all_stairs_id = get_stairs_id(VLFMTrainer.habitat_env)
        # 待检查：多个act的执行顺序
        # 待检查，batch和observations
        batch = batch_obs(observations, device=self.device) # 将来自多个环境的观察批量处理成单个张量，并将其移动到指定的 self.device
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore # 对批量观察应用任何观察转换（例如，归一化、调整大小）

        VLFMTrainer.rgb_ls, VLFMTrainer.depth_ls, VLFMTrainer.semantic_ls = get_rgb_image_ls(VLFMTrainer.habitat_env)


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


        # 开始评估：遍历每一个episode（evals_per_ep默认为1， self.envs.num_envs为1）
        # while len(stats_episodes) < (number_of_eval_episodes * evals_per_ep) and self.envs.num_envs > 0:
        while len(stats_episodes) < (number_of_eval_episodes * evals_per_ep):
            need_continue = False
            # =====> 初始化开始 <=====
            while (VLFMTrainer.has_beed_initialize==False):
                action_data = self._agent.actor_critic.act(
                    batch,
                    None,
                    None,
                    not_done_masks,
                    deterministic=False,
                ) # 关键的一步 (actor_critic是HabitatITMPolicyV2类型,实际调用HabitatMixin类中的act)
                
                if(action_data is not None): # 说明还在一些coner case阶段，初始化将继续
                    step_data = [a.item() for a in action_data.env_actions.cpu()]

                    # 实际执行step_data
                    observations = VLFMTrainer.habitat_env.step(step_data[0])

                    if(step_data[0]==0):
                        VLFMTrainer.stop_call = 1


                    observations = get_new_obs(observations)
                    infos = [VLFMTrainer.habitat_env.get_metrics()]
                    policy_infos = self._agent.actor_critic.get_extra(action_data, None, None)
                    for i in range(len(policy_infos)):
                        infos[i].update(policy_infos[i]) # 将策略特定的信息合并到环境信息中
                    
                    # 新的 observations 被批量处理和转换，类似于初始设置，为下一次智能体的 act 调用准备 batch
                    batch = batch_obs(  # type: ignore
                        observations,
                        device=self.device,
                    ) 
                    batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

                    VLFMTrainer.rgb_ls, VLFMTrainer.depth_ls, VLFMTrainer.semantic_ls = get_rgb_image_ls(VLFMTrainer.habitat_env)

                    not_done_masks = torch.tensor(
                        [[not VLFMTrainer.habitat_env.episode_over]],
                        dtype=torch.bool,
                        device="cpu",
                    ) # 根据来自环境的 dones 标志进行更新

                    n_envs = 1
                    for i in range(n_envs):
                        if len(self.config.habitat_baselines.eval.video_option) > 0:
                            hab_vis.collect_data(batch, infos, action_data.policy_info)

                    not_done_masks = not_done_masks.to(device=self.device) # 将 not_done_masks 张量移动到目标设备 (self.device)，因为它将在下一次迭代中作为智能体的输入 


                    if not not_done_masks[0].item(): # 如果当前回合已经结束
                        need_continue = True
                        pbar.update() # 更新进度条
                        episode_stats = {} # 创建一个字典来存储此回合的最终统计数据
                        episode_stats.update(extract_scalars_from_info(infos[i])) # 用于从 infos 字典中提取标量指标（例如，success、spl、distance_to_goal）
                        k = (
                            VLFMTrainer.habitat_env.current_episode.scene_id,
                            VLFMTrainer.habitat_env.current_episode.episode_id,
                        )
                        ep_eval_count[k] += 1 # 增加此特定回合实例的评估计数
                        # use scene_id + episode_id as unique id for storing stats
                        stats_episodes[(k, ep_eval_count[k])] = episode_stats


                        num_total += 1
                        print(f"Success rate: {num_successes / num_total * 100:.2f}% ({num_successes} out of {num_total})")
                        

                        if math.isfinite(episode_stats["distance_to_goal"]) and math.isfinite(VLFMTrainer.episode_reward):
                            all_spl_ls.append(episode_stats["spl"])
                            all_ne_ls.append(episode_stats["distance_to_goal"])

                            real_index = num_total
                            # real_index = len(all_spl_ls)+1125 # 中断修改
                            # episode_stats: dict_keys(['reward', 'distance_to_goal', 'success', 'spl', 'soft_spl', 'distance_to_goal_reward', 'traveled_stairs', 'yaw', 'target_detected', 'stop_called'])
                            writer.add_scalar("Result/distance_to_goal", episode_stats["distance_to_goal"], real_index)
                            # writer.add_scalar("Result/success_num", num_successes+1010, real_index) # 中断修改
                            writer.add_scalar("Result/success_num", num_successes, real_index) 
                            writer.add_scalar("Result/spl_per_episode", episode_stats["spl"], real_index)
                            writer.add_scalar("Result/stop_called", VLFMTrainer.stop_call, real_index)
                            # writer.add_scalar("Result/spl_mean", (np.sum(all_spl_ls)+0.5518*1125)/real_index, real_index) # 中断修改
                            writer.add_scalar("Result/spl_mean", np.mean(all_spl_ls), real_index)
                            # writer.add_scalar("Result/ne_mean", (np.sum(all_ne_ls)+1.666*1125)/real_index, real_index) # 中断修改
                            writer.add_scalar("Result/ne_mean", np.mean(all_ne_ls), real_index)
                            writer.add_scalar("Result_RL/episode_reward", VLFMTrainer.episode_reward, real_index)


                            # # real_index = num_total
                            # real_index = len(all_spl_ls)
                            # # episode_stats: dict_keys(['reward', 'distance_to_goal', 'success', 'spl', 'soft_spl', 'distance_to_goal_reward', 'traveled_stairs', 'yaw', 'target_detected', 'stop_called'])
                            # writer.add_scalar("Result/distance_to_goal", episode_stats["distance_to_goal"], real_index)
                            # writer.add_scalar("Result/success_num", num_successes, real_index)
                            # writer.add_scalar("Result/spl_per_episode", episode_stats["spl"], real_index)
                            # writer.add_scalar("Result/stop_called", VLFMTrainer.stop_call, real_index)
                            # writer.add_scalar("Result/spl_mean", np.mean(all_spl_ls), real_index)
                            # writer.add_scalar("Result/ne_mean", np.mean(all_ne_ls), real_index)
                            # writer.add_scalar("Result_RL/episode_reward", VLFMTrainer.episode_reward, real_index)


                        
                        from vlfm.utils.episode_stats_logger import (
                            log_episode_stats,
                        )

                        # 确定failure_cause
                        try:
                            failure_cause = log_episode_stats(
                                VLFMTrainer.habitat_env.current_episode.episode_id,
                                VLFMTrainer.habitat_env.current_episode.scene_id,
                                infos[i],
                            )
                        except Exception:
                            failure_cause = "Unknown"

                        # 视频选项（暂时不管）
                        if len(self.config.habitat_baselines.eval.video_option) > 0:
                            rgb_frames[i] = hab_vis.flush_frames(failure_cause)
                            try:
                                generate_video(
                                    video_option=self.config.habitat_baselines.eval.video_option,
                                    video_dir=self.config.habitat_baselines.video_dir,
                                    images=rgb_frames[i],
                                    # episode_id=current_episodes_info[i].episode_id,
                                    episode_id=real_index,
                                    scene_id=VLFMTrainer.habitat_env.current_episode.scene_id.split('/')[-2].split('-')[0],
                                    checkpoint_idx=checkpoint_index,
                                    metrics=extract_scalars_from_info(infos[i]),
                                    fps=self.config.habitat_baselines.video_fps,
                                    tb_writer=writer,
                                    keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name,
                                )
                            except:
                                generate_video(
                                    video_option=self.config.habitat_baselines.eval.video_option,
                                    video_dir=self.config.habitat_baselines.video_dir,
                                    images=rgb_frames[i],
                                    # episode_id=current_episodes_info[i].episode_id,
                                    episode_id=real_index,
                                    scene_id="00000",
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
                                VLFMTrainer.habitat_env.current_episode.episode_id,
                            )    
                        
                        while True:
                            index_in_episodes += 1
                            VLFMTrainer.habitat_env.episodes = [selected_episodes[index_in_episodes]]
                            # 1. 判断当前episode是否可以reset
                            try:
                                observations = VLFMTrainer.habitat_env.reset()
                            except:
                                continue

                            # =====> 新增object_id <=====
                            try:
                                classes_ls = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]
                                scene_name = VLFMTrainer.habitat_env.current_episode.scene_id.split('/')[-2].split('-')[1]
                                file_path = f'data/datasets/objectnav/hm3d/v1/train/content/{scene_name}.json.gz' 
                                temp_object_goal = classes_ls[observations['objectgoal'][0]]

                                # 使用 'rt' 模式（读取文本）打开 gzip 文件
                                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                                    # json.load() 直接从文件对象中读取并解析 JSON 数据
                                    json_data = json.load(f)
                                VLFMTrainer.current_episode_object_id = [temp_data["object_id"] for temp_data in json_data['goals_by_category'][scene_name+f".basis.glb_{temp_object_goal}"]]
                            
                            except:
                                print("doing continueeeeeeeee")
                                continue
                            # =====> 新增object_id <=====
                        
                            observations = get_new_obs(observations)
                            # if math.isinf(VLFMTrainer.habitat_env.get_metrics()["distance_to_goal"])==False:
                            if math.isfinite(VLFMTrainer.habitat_env.get_metrics()["distance_to_goal"]):
                                break

                        VLFMTrainer.res_all_stairs_id = get_stairs_id(VLFMTrainer.habitat_env)
                        # 待检查：多个act的执行顺序
                        # 待检查，batch和observations
                        batch = batch_obs(observations, device=self.device) # 将来自多个环境的观察批量处理成单个张量，并将其移动到指定的 self.device
                        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore # 对批量观察应用任何观察转换（例如，归一化、调整大小）

                        VLFMTrainer.rgb_ls, VLFMTrainer.depth_ls, VLFMTrainer.semantic_ls = get_rgb_image_ls(VLFMTrainer.habitat_env)

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
                        VLFMTrainer.has_beed_initialize = False
                        VLFMTrainer.floor_select_steps = 0
                        VLFMTrainer.last_floor_select_steps = 0

                        VLFMTrainer.current_floor_steps = 0
                        VLFMTrainer.last_current_floor_steps = 0

                        VLFMTrainer.stop_call = 0
                        break
                else: # 说明已经结束了coner case，开始真正得到了一个next_state
                    VLFMTrainer.has_beed_initialize = True
                    VLFMTrainer.current_state = copy.deepcopy(VLFMTrainer.next_state)
                    last_geodesic_dis = VLFMTrainer.habitat_env.get_metrics()['distance_to_goal']
                    break
            # =====> 初始化结束 <=====
            if(need_continue == True):
                continue

            # ===============> 以下开始正式进入训练 <===============

            # ===============> 实时load最新model <===============
            max_id = find_max_number_in_filenames(folder_path=f'{q_args.root}/{q_args.model_file_name}/policy/{q_args.experiment_details}/')
            if (max_id>VLFMTrainer.now_model_id):
                # load新model
                # # load训练好的模型(rl)
                time.sleep(0.1)
                checkpoint = torch.load(f'{q_args.root}/{q_args.model_file_name}/policy/{q_args.experiment_details}/{max_id}_actor')
                while True:
                    try:
                        VLFMTrainer.rl_policy.actor.load_state_dict(checkpoint)
                        break
                    except:
                        time.sleep(0.1)
                        continue
                VLFMTrainer.now_model_id = max_id
            
            if(action_data is None): # 1. 表示获得正常的next_state
                action_data = self._agent.actor_critic.select_action() # 得到VLFMTrainer.current_action
            else: # 反常
                VLFMTrainer.current_action = None # 在下一次获得state之前，都用current_action判断

            if(os.environ["DEBUG_INFO"] == "navigate") and (VLFMTrainer.habitat_env.get_metrics()['distance_to_goal']<1.0) and (not VLFMTrainer.habitat_env.episode_over):
                step_data = [0]
            else:    
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            # 实际执行step_data
            observations = VLFMTrainer.habitat_env.step(step_data[0])
            if(step_data[0]==0):
                VLFMTrainer.stop_call = 1

            observations = get_new_obs(observations)
            infos = [VLFMTrainer.habitat_env.get_metrics()]
            policy_infos = self._agent.actor_critic.get_extra(action_data, None, None)
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i]) # 将策略特定的信息合并到环境信息中
            
            # 新的 observations 被批量处理和转换，类似于初始设置，为下一次智能体的 act 调用准备 batch
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
            ) 
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            VLFMTrainer.rgb_ls, VLFMTrainer.depth_ls, VLFMTrainer.semantic_ls = get_rgb_image_ls(VLFMTrainer.habitat_env)

            not_done_masks = torch.tensor(
                [[not VLFMTrainer.habitat_env.episode_over]],
                dtype=torch.bool,
                device="cpu",
            ) # 根据来自环境的 dones 标志进行更新

            n_envs = 1
            for i in range(n_envs):
                if len(self.config.habitat_baselines.eval.video_option) > 0:
                    hab_vis.collect_data(batch, infos, action_data.policy_info)

            # 每走一步都要获得一次state
            temp_not_done_masks = torch.tensor(
                [[True]],
                dtype=torch.bool,
                device="cpu",
            ) # 根据来自环境的 dones 标志进行更新

            action_data = self._agent.actor_critic.act(
                batch,
                None,
                None,
                temp_not_done_masks, # 此处的temp_not_done_masks只在_pre_step中主要使用
                deterministic=False,
            ) # 关键的一步 (actor_critic是HabitatITMPolicyV2类型,实际调用HabitatMixin类中的act)

            for i in range(n_envs):
                # episode ended
                if not not_done_masks[i].item(): # 如果环境 i 中的一个回合已结束
                    pbar.update() # 更新进度条
                    episode_stats = {} # 创建一个字典来存储此回合的最终统计数据
                    episode_stats.update(extract_scalars_from_info(infos[i])) # 用于从 infos 字典中提取标量指标（例如，success、spl、distance_to_goal）
                    k = (
                        VLFMTrainer.habitat_env.current_episode.scene_id,
                        VLFMTrainer.habitat_env.current_episode.episode_id,
                    )
                    ep_eval_count[k] += 1 # 增加此特定回合实例的评估计数
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if episode_stats["success"] == 1:
                        now_geodesic_dis = episode_stats["distance_to_goal"]
                        num_successes += 1
                        rl_step_reward = 2.5
                    else:
                        now_geodesic_dis = episode_stats["distance_to_goal"]
                        # rl_step_reward = -(now_geodesic_dis-last_geodesic_dis)-0.01*(VLFMTrainer.floor_select_steps-VLFMTrainer.last_floor_select_steps)
                        rl_step_reward = -(now_geodesic_dis-last_geodesic_dis)-0.01*(VLFMTrainer.current_floor_steps-VLFMTrainer.last_current_floor_steps)

                    
                    last_geodesic_dis = now_geodesic_dis
                    VLFMTrainer.episode_reward += rl_step_reward

                    num_total += 1
                    print(f"Success rate: {num_successes / num_total * 100:.2f}% ({num_successes} out of {num_total})")

                    if math.isfinite(episode_stats["distance_to_goal"]) and math.isfinite(VLFMTrainer.episode_reward):
                        all_spl_ls.append(episode_stats["spl"])
                        all_ne_ls.append(episode_stats["distance_to_goal"])


                        real_index = num_total
                        # real_index = len(all_spl_ls)+1125 # 中断修改
                        # episode_stats: dict_keys(['reward', 'distance_to_goal', 'success', 'spl', 'soft_spl', 'distance_to_goal_reward', 'traveled_stairs', 'yaw', 'target_detected', 'stop_called'])
                        writer.add_scalar("Result/distance_to_goal", episode_stats["distance_to_goal"], real_index)
                        # writer.add_scalar("Result/success_num", num_successes+1010, real_index) # 中断修改
                        writer.add_scalar("Result/success_num", num_successes, real_index) 
                        writer.add_scalar("Result/spl_per_episode", episode_stats["spl"], real_index)
                        writer.add_scalar("Result/stop_called", VLFMTrainer.stop_call, real_index)
                        # writer.add_scalar("Result/spl_mean", (np.sum(all_spl_ls)+0.5518*1125)/real_index, real_index) # 中断修改
                        writer.add_scalar("Result/spl_mean", np.mean(all_spl_ls), real_index)
                        # writer.add_scalar("Result/ne_mean", (np.sum(all_ne_ls)+1.666*1125)/real_index, real_index) # 中断修改
                        writer.add_scalar("Result/ne_mean", np.mean(all_ne_ls), real_index)
                        writer.add_scalar("Result_RL/episode_reward", VLFMTrainer.episode_reward, real_index)

                        # # real_index = num_total
                        # real_index = len(all_spl_ls)
                        # # episode_stats: dict_keys(['reward', 'distance_to_goal', 'success', 'spl', 'soft_spl', 'distance_to_goal_reward', 'traveled_stairs', 'yaw', 'target_detected', 'stop_called'])
                        # writer.add_scalar("Result/distance_to_goal", episode_stats["distance_to_goal"], real_index)
                        # writer.add_scalar("Result/success_num", num_successes, real_index)
                        # writer.add_scalar("Result/spl_per_episode", episode_stats["spl"], real_index)
                        # writer.add_scalar("Result/stop_called", VLFMTrainer.stop_call, real_index)
                        # writer.add_scalar("Result/spl_mean", np.mean(all_spl_ls), real_index)
                        # writer.add_scalar("Result/ne_mean", np.mean(all_ne_ls), real_index)
                        # writer.add_scalar("Result_RL/episode_reward", VLFMTrainer.episode_reward, real_index)





                        # writer.add_scalar("Result/has_vis_object_goal", HabitatAction.has_vis_object_goal, real_index)

                    
                    # if episode_stats["success"] == 1:
                    #     writer.add_scalar("Result/result_state", 1, real_index)
                    # elif (os.environ["DEBUG_INFO"] == "initialize"):
                    #     writer.add_scalar("Result/result_state", 0, real_index)
                    # elif (os.environ["DEBUG_INFO"] == "navigate_reliable"):
                    #     writer.add_scalar("Result/result_state", 2, real_index)
                    # elif (os.environ["DEBUG_INFO"] == "navigate_suspect"):
                    #     writer.add_scalar("Result/result_state", 3, real_index)
                    # elif (os.environ["DEBUG_INFO"] == "explore"):
                    #     writer.add_scalar("Result/result_state", 4, real_index)
                    # elif (os.environ["DEBUG_INFO"] == "except_navigate"):
                    #     writer.add_scalar("Result/result_state", 5, real_index)
                    # elif (os.environ["DEBUG_INFO"] == "no_action"):
                    #     writer.add_scalar("Result/result_state", 6, real_index)
                    # else:
                    #     writer.add_scalar("Result/result_state", 7, real_index)


                    # # 新的考虑
                    # if episode_stats["success"] == 1:
                    #     writer.add_scalar("Result/result_state", 1, real_index)
                    # elif (os.environ["DEBUG_INFO"] == "initialize"):
                    #     writer.add_scalar("Result/result_state", 0, real_index)
                    # elif (os.environ["DEBUG_INFO"] == "navigate_reliable"):
                    #     writer.add_scalar("Result/result_state", 2, real_index)
                    # elif (os.environ["DEBUG_INFO"] == "navigate_suspect"):
                    #     writer.add_scalar("Result/result_state", 3, real_index)
                    # elif (os.environ["DEBUG_INFO"] == "explore"):
                    #     writer.add_scalar("Result/result_state", 4, real_index)
                    # elif (os.environ["DEBUG_INFO"] == "except_navigate"):
                    #     writer.add_scalar("Result/result_state", 5, real_index)
                    # elif (os.environ["DEBUG_INFO"] == "no_action"):
                    #     writer.add_scalar("Result/result_state", 6, real_index)
                    # elif (os.environ["DEBUG_INFO"] == "normal_navigate"):
                    #     writer.add_scalar("Result/result_state", 8, real_index)
                    # else:
                    #     writer.add_scalar("Result/result_state", 7, real_index)

                    from vlfm.utils.episode_stats_logger import (
                        log_episode_stats,
                    )


                    # 确定failure_cause
                    try:
                        failure_cause = log_episode_stats(
                            VLFMTrainer.habitat_env.current_episode.episode_id,
                            VLFMTrainer.habitat_env.current_episode.scene_id,
                            infos[i],
                        )
                    except Exception:
                        failure_cause = "Unknown"

                    # 视频选项（暂时不管）
                    if len(self.config.habitat_baselines.eval.video_option) > 0:
                        rgb_frames[i] = hab_vis.flush_frames(failure_cause)
                        try:
                            generate_video(
                                video_option=self.config.habitat_baselines.eval.video_option,
                                video_dir=self.config.habitat_baselines.video_dir,
                                images=rgb_frames[i],
                                # episode_id=current_episodes_info[i].episode_id,
                                episode_id=real_index,
                                scene_id=VLFMTrainer.habitat_env.current_episode.scene_id.split('/')[-2].split('-')[0],
                                checkpoint_idx=checkpoint_index,
                                metrics=extract_scalars_from_info(infos[i]),
                                fps=self.config.habitat_baselines.video_fps,
                                tb_writer=writer,
                                keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name,
                            )
                        except:
                            generate_video(
                                video_option=self.config.habitat_baselines.eval.video_option,
                                video_dir=self.config.habitat_baselines.video_dir,
                                images=rgb_frames[i],
                                # episode_id=current_episodes_info[i].episode_id,
                                episode_id=real_index,
                                scene_id="00000",
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
                            current_episodes_info[i].episode_id,
                        )

                else:
                    episode_stats = {} # 创建一个字典来存储此回合的最终统计数据
                    episode_stats.update(extract_scalars_from_info(infos[i]))

                    if ((VLFMTrainer.current_floor_steps%5)==0) and (action_data is None) and (VLFMTrainer.current_action is not None):
                        now_geodesic_dis = episode_stats["distance_to_goal"]

                        # rl_step_reward = -(now_geodesic_dis-last_geodesic_dis)-0.01
                        # rl_step_reward = -(now_geodesic_dis-last_geodesic_dis)-0.01*(VLFMTrainer.floor_select_steps-VLFMTrainer.last_floor_select_steps)
                        rl_step_reward = -(now_geodesic_dis-last_geodesic_dis)-0.01*(VLFMTrainer.current_floor_steps-VLFMTrainer.last_current_floor_steps)
                        last_geodesic_dis = now_geodesic_dis
                        VLFMTrainer.episode_reward += rl_step_reward
                        VLFMTrainer.last_floor_select_steps = VLFMTrainer.floor_select_steps
                        VLFMTrainer.last_current_floor_steps = VLFMTrainer.current_floor_steps

                if(VLFMTrainer.current_action is not None) and (action_data is None) and ((VLFMTrainer.current_floor_steps%5)==0 or (not not_done_masks[i].item())) and (math.isfinite(rl_step_reward)): # action_space为空，不加入buffer中，不记录rl_reward，记录episode_reward
                    now_total_step = load_total_steps()+1 # 现在该第N个step

                    if (not not_done_masks[i].item()) and (VLFMTrainer.stop_call==1): # 表示在有限步数内，当前episode执行完成(找到正确的intention or 找到错误的intention)
                        terminal = True
                    else: # 超过最大步长在这个里面
                        terminal = False

                    if (not not_done_masks[i].item()):
                        temp_done = True
                    else:
                        temp_done = False
                    
                    store_transition(VLFMTrainer.current_state, VLFMTrainer.current_action, rl_step_reward, VLFMTrainer.next_state, terminal, temp_done, now_total_step)
                    
                    VLFMTrainer.current_state = copy.deepcopy(VLFMTrainer.next_state)
                    
                    # 创建空文件
                    filename = f"{main_args.card_select}_{now_total_step}_{rl_step_reward}_{main_args.process_id}.txt"
                    with open(f"main_process_info/step_rewards/{filename}", 'w', encoding='utf-8') as file:
                        pass

                    # writer.add_scalar("Result_RL/rl_step_reward", rl_step_reward, now_total_step) # 暂时不用

                    # # ==========> 真正的主线程代码开始 <==========
                    # if ((now_total_steps%q_args.delta_steps)==0):
                    #     update_step = (VLFMTrainer.total_steps//q_args.delta_steps)
                    #     total_loss_per_update = VLFMTrainer.rl_policy.learn(VLFMTrainer.replay_buffer, VLFMTrainer.total_steps)
                    #     assert len(VLFMTrainer.episode_rl_step_buffer)==q_args.delta_steps
                    #     mean_episode_rl_step_buffer = np.mean(VLFMTrainer.episode_rl_step_buffer)
                    #     writer.add_scalar("Result_RL/mean_episode_rl_step_buffer", mean_episode_rl_step_buffer, update_step)
                    #     writer.add_scalar("Result_RL/total_loss_per_update", total_loss_per_update, update_step)

                    #     VLFMTrainer.episode_rl_step_buffer = []  

                    #     # =====> 创建新目录 <=====
                    #     model_pre_dir = '{0}/{1}/policy/{2}'.format(q_args.root, q_args.model_file_name, q_args.experiment_details)
                    #     if not os.path.exists(model_pre_dir):
                    #         os.makedirs(model_pre_dir)
                    #         print(f"The new path:'{model_pre_dir}' has beed craeted!")
                    #     # =====> 创建新目录 <=====

                    #     VLFMTrainer.rl_policy.save('{0}/{1}/policy/{2}/{3}'.format(q_args.root, q_args.model_file_name, q_args.experiment_details, update_step))

                    # # ==========> 真正的主线程代码结束 <==========

            not_done_masks = not_done_masks.to(device=self.device) # 将 not_done_masks 张量移动到目标设备 (self.device)，因为它将在下一次迭代中作为智能体的输入
            if not not_done_masks[0].item(): # 如果当前回合已经结束
                while True:
                    index_in_episodes += 1
                    VLFMTrainer.habitat_env.episodes = [selected_episodes[index_in_episodes]]
                    # 1. 判断当前episode是否可以reset
                    try:
                        observations = VLFMTrainer.habitat_env.reset()
                    except:
                        continue

                    # =====> 新增object_id <=====
                    try:
                        classes_ls = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]
                        scene_name = VLFMTrainer.habitat_env.current_episode.scene_id.split('/')[-2].split('-')[1]
                        file_path = f'data/datasets/objectnav/hm3d/v1/train/content/{scene_name}.json.gz' 
                        temp_object_goal = classes_ls[observations['objectgoal'][0]]

                        # 使用 'rt' 模式（读取文本）打开 gzip 文件
                        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                            # json.load() 直接从文件对象中读取并解析 JSON 数据
                            json_data = json.load(f)
                        VLFMTrainer.current_episode_object_id = [temp_data["object_id"] for temp_data in json_data['goals_by_category'][scene_name+f".basis.glb_{temp_object_goal}"]]
                    
                    except:
                        print("doing continueeeeeeeee")
                        continue
                    # =====> 新增object_id <=====
                
                    observations = get_new_obs(observations)
                    # if math.isinf(VLFMTrainer.habitat_env.get_metrics()["distance_to_goal"])==False:
                    if math.isfinite(VLFMTrainer.habitat_env.get_metrics()["distance_to_goal"]):
                        break

                VLFMTrainer.res_all_stairs_id = get_stairs_id(VLFMTrainer.habitat_env)
                # 待检查：多个act的执行顺序
                # 待检查，batch和observations
                batch = batch_obs(observations, device=self.device) # 将来自多个环境的观察批量处理成单个张量，并将其移动到指定的 self.device
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore # 对批量观察应用任何观察转换（例如，归一化、调整大小）

                VLFMTrainer.rgb_ls, VLFMTrainer.depth_ls, VLFMTrainer.semantic_ls = get_rgb_image_ls(VLFMTrainer.habitat_env)

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
                VLFMTrainer.has_beed_initialize = False
                VLFMTrainer.floor_select_steps = 0
                VLFMTrainer.last_floor_select_steps = 0

                VLFMTrainer.current_floor_steps = 0
                VLFMTrainer.last_current_floor_steps = 0

                VLFMTrainer.stop_call = 0

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

        # self.envs.close()
