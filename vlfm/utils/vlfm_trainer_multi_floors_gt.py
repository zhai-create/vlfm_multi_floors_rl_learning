# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
from vlfm.arguments import args as main_args

import random
random.seed(233)

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

from vlfm.policy.unet_model import HistoryAwareUNet, Double_Q_Net

import gzip
import json
import math

import copy

import math

import quaternion

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
    current_scene_dir = "data/scene_datasets/hm3d/val/"+habitat_env.current_episode.scene_id.split('/')[-2]+"/"
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

    current_episode_object_id = []
    rgb_ls, depth_ls, semantic_ls = [], [], []

    # # load训练好的IL模型
    # policy_model = HistoryAwareUNet(in_channels=6).to("cuda")
    # pre_policy = 'checkpoints_all_visions/epoch_9.pt'
    # checkpoint = torch.load(pre_policy, map_location="cuda")['model_state']
    # policy_model.load_state_dict(checkpoint)
    # policy_model.eval()

    # load训练好的RL模型
    # policy_model = Double_Q_Net(in_channels=6, device="cuda").to("cuda")
    policy_model = HistoryAwareUNet(in_channels=6).to("cuda")
    policy_model.load_state_dict(torch.load("Models_train_PPO/policy/multi_process_sac/1002_actor"))
    policy_model.eval()

    current_sim_step = 0 # 5_delta_steps修改
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
        # logger_file_name = "./log_files/log_ascent_get_new_critic_q_value_multi_stairs_RL_model_1002_multi_revise_failure_part_"+date_time+f"_init_{main_args.init_episode}"
        logger_file_name = "./log_files/temp_debug_log_ascent_actor_gt_perception_multi_stairs_RL_model_1002_multi_revise_"+date_time+f"_init_{main_args.init_episode}"
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
            # config.habitat_baselines.eval.split = "train"
            config.habitat_baselines.eval.split = "val"
            config.habitat.dataset.split = config.habitat_baselines.eval.split

            # config.habitat.dataset.data_path = "data/datasets/objectnav/hm3d/v1/train/train.json.gz"
            config.habitat.dataset.data_path = "data/datasets/objectnav/hm3d/v1/val/val.json.gz"


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
        selected_episodes = selected_episodes[main_args.init_episode:]
        # =====> select episodes <=====


        self._init_envs(config, is_eval=True) # 初始化env(暂时不管)

        self._agent = self._create_agent(None) # 初始化agent(暂时不管)
        action_shape, discrete_actions = get_action_space_info(self._agent.policy_action_space)  # 获取action_space形状+action_space是否离散

        # 加载智能体状态：待检查
        if self._agent.actor_critic.should_load_agent_state:
            self._agent.load_state_dict(ckpt_dict)

        index_in_episodes = 0
        habitat_env.episodes = [selected_episodes[index_in_episodes]]
        observations = habitat_env.reset()
        
        # =====> 新增object_id <=====
        classes_ls = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]
        scene_name = habitat_env.current_episode.scene_id.split('/')[-2].split('-')[1]
        file_path = f'data/datasets/objectnav/hm3d/v1/val/content/{scene_name}.json.gz' 
        temp_object_goal = classes_ls[observations['objectgoal'][0]]

        # 使用 'rt' 模式（读取文本）打开 gzip 文件
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            # json.load() 直接从文件对象中读取并解析 JSON 数据
            json_data = json.load(f)
        VLFMTrainer.current_episode_object_id = [temp_data["object_id"] for temp_data in json_data['goals_by_category'][scene_name+f".basis.glb_{temp_object_goal}"]]
        # =====> 新增object_id <=====
        
        observations = get_new_obs(observations)
        VLFMTrainer.res_all_stairs_id = get_stairs_id(habitat_env)

        
        # 待检查：多个act的执行顺序
        # 待检查，batch和observations
        batch = batch_obs(observations, device=self.device) # 将来自多个环境的观察批量处理成单个张量，并将其移动到指定的 self.device
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore # 对批量观察应用任何观察转换（例如，归一化、调整大小）

        VLFMTrainer.rgb_ls, VLFMTrainer.depth_ls, VLFMTrainer.semantic_ls = get_rgb_image_ls(habitat_env)


        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu") # 用于跟踪每个并行环境中当前回合的累积奖励
        test_recurrent_hidden_states = torch.zeros(
            (
                self.config.habitat_baselines.num_environments,
                *self._agent.hidden_state_shape,
            ),
            device=self.device,
        ) # RNN隐含层状态
        prev_actions = torch.zeros(
            self.config.habitat_baselines.num_environments,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        ) # 存储在每个环境中采取的前一个动作 （待检查）
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

        number_of_eval_episodes = self.config.habitat_baselines.test_episode_count # 请求的episode总数
        evals_per_ep = self.config.habitat_baselines.eval.evals_per_ep # 每个episode的评估次数
        if number_of_eval_episodes == -1: # 使用split数据集中所有可用的回合
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes, dataset only has {{total_num_eps}}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert number_of_eval_episodes > 0, "You must specify a number of evaluation episodes with test_episode_count"

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
            with inference_mode(): # 禁用梯度计算
                if ((VLFMTrainer.current_sim_step%5)==0):
                    need_replan = True
                else:
                    need_replan = False
                    # need_replan = True

                action_data, all_save_dict_ls = self._agent.actor_critic.act(
                    need_replan,
                    batch,
                    None,
                    None,
                    not_done_masks,
                    deterministic=False,
                ) # 关键的一步 (actor_critic是HabitatITMPolicyV2类型,实际调用HabitatMixin类中的act)

            # 获得可供env执行的step_data（每个并行环境对应一个动作）
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if is_continuous_action_space(self._env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        self._env_spec.action_space.low,
                        self._env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            # 实际执行step_data
            observations = habitat_env.step(step_data[0])
            observations = get_new_obs(observations)
            infos = [habitat_env.get_metrics()]
            policy_infos = self._agent.actor_critic.get_extra(action_data, None, None)
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i]) # 将策略特定的信息合并到环境信息中


            VLFMTrainer.current_sim_step += 1
            
            # 新的 observations 被批量处理和转换，类似于初始设置，为下一次智能体的 act 调用准备 batch
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
            ) 
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            VLFMTrainer.rgb_ls, VLFMTrainer.depth_ls, VLFMTrainer.semantic_ls = get_rgb_image_ls(habitat_env)

            not_done_masks = torch.tensor(
                [[not habitat_env.episode_over]],
                dtype=torch.bool,
                device="cpu",
            ) # 根据来自环境的 dones 标志进行更新

            n_envs = 1
            for i in range(n_envs):
                if len(self.config.habitat_baselines.eval.video_option) > 0:
                    hab_vis.collect_data(batch, infos, action_data.policy_info)

                # episode ended
                if not not_done_masks[i].item(): # 如果环境 i 中的一个回合已结束
                    VLFMTrainer.current_sim_step = 0
                    pbar.update() # 更新进度条
                    episode_stats = {} # 创建一个字典来存储此回合的最终统计数据
                    episode_stats.update(extract_scalars_from_info(infos[i])) # 用于从 infos 字典中提取标量指标（例如，success、spl、distance_to_goal）
                    current_episode_reward[i] = 0 # 重置此环境的奖励累加器
                    k = (
                        habitat_env.current_episode.scene_id,
                        habitat_env.current_episode.episode_id,
                    )
                    ep_eval_count[k] += 1 # 增加此特定回合实例的评估计数
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if episode_stats["success"] == 1:
                        num_successes += 1
                    num_total += 1
                    print(f"Success rate: {num_successes / num_total * 100:.2f}% ({num_successes} out of {num_total})")
                    
                    all_spl_ls.append(episode_stats["spl"])
                    all_ne_ls.append(episode_stats["distance_to_goal"])

                    real_index = num_total+main_args.init_episode
                    # episode_stats: dict_keys(['reward', 'distance_to_goal', 'success', 'spl', 'soft_spl', 'distance_to_goal_reward', 'traveled_stairs', 'yaw', 'target_detected', 'stop_called'])
                    writer.add_scalar("Result/distance_to_goal", episode_stats["distance_to_goal"], real_index)
                    writer.add_scalar("Result/success_num", num_successes, real_index)
                    writer.add_scalar("Result/spl_per_episode", episode_stats["spl"], real_index)
                    writer.add_scalar("Result/stop_called", episode_stats["stop_called"], real_index)
                    writer.add_scalar("Result/spl_mean", np.mean(all_spl_ls), real_index)
                    writer.add_scalar("Result/ne_mean", np.mean(all_ne_ls), real_index)

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
                            habitat_env.current_episode.episode_id,
                            habitat_env.current_episode.scene_id,
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
                                scene_id=habitat_env.current_episode.scene_id.split('/')[-2].split('-')[0],
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

            not_done_masks = not_done_masks.to(device=self.device) # 将 not_done_masks 张量移动到目标设备 (self.device)，因为它将在下一次迭代中作为智能体的输入
            
            if not not_done_masks[0].item(): # 如果当前回合已经结束
                index_in_episodes += 1
                habitat_env.episodes = [selected_episodes[index_in_episodes]]
                observations = habitat_env.reset()

                # =====> 新增object_id <=====
                classes_ls = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]
                scene_name = habitat_env.current_episode.scene_id.split('/')[-2].split('-')[1]
                file_path = f'data/datasets/objectnav/hm3d/v1/val/content/{scene_name}.json.gz' 
                temp_object_goal = classes_ls[observations['objectgoal'][0]]

                # 使用 'rt' 模式（读取文本）打开 gzip 文件
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    # json.load() 直接从文件对象中读取并解析 JSON 数据
                    json_data = json.load(f)
                VLFMTrainer.current_episode_object_id = [temp_data["object_id"] for temp_data in json_data['goals_by_category'][scene_name+f".basis.glb_{temp_object_goal}"]]
                # =====> 新增object_id <=====

                observations = get_new_obs(observations)
                VLFMTrainer.res_all_stairs_id = get_stairs_id(habitat_env)

                # 待检查：多个act的执行顺序
                batch = batch_obs(observations, device=self.device) # 将来自多个环境的观察批量处理成单个张量，并将其移动到指定的 self.device
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore # 对批量观察应用任何观察转换（例如，归一化、调整大小）

                VLFMTrainer.rgb_ls, VLFMTrainer.depth_ls, VLFMTrainer.semantic_ls = get_rgb_image_ls(habitat_env)

                not_done_masks = torch.zeros(
                    self.config.habitat_baselines.num_environments,
                    1,
                    device=self.device,
                    dtype=torch.bool,
                ) # 指示并行环境中的每个回合是否仍在进行 （已完成为False）

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
