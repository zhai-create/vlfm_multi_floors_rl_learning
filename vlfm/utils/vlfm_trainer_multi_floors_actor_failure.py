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
        logger_file_name = "./log_files/log_ascent_actor_multi_stairs_RL_model_1002_multi_revise_failure_part_"+date_time+f"_init_{main_args.init_episode}"
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
        selected_episodes = []
        all_habitat_episodes = copy.deepcopy(habitat_env.episodes)
        
        '''
        all_false_episode_id = [1, 9, 11, 15, 16, 22, 24, 25, 26, 28, 29, 30, 31, 32, 33, 36, 39, 40, 43, 45, 50, 55, 56, 59, 62, 64, 67, 68, 69, 70, 73, 74, 75, 76, 78, 81, 82, 85, 86, 87, 88, 91, 93, 94, 96, 97, 99, 100, 104, 105, 106, 108, 111, 112, 113, 117, 118, 119, 120, 122, 123, 126, 127, 128, 130, 131, 138, 140, 143, 145, 151, 153, 155, 156, 163, 164, 167, 170, 172, 173, 174, 178, 180, 183, 184, 186, 187, 189, 192, 193, 195, 198, 200, 202, 203, 204, 208, 212, 214, 215, 219, 220, 221, 222, 223, 226, 227, 228, 230, 232, 234, 236, 237, 240, 243, 248, 249, 252, 255, 258, 262, 266, 267, 274, 275, 278, 280, 281, 285, 287, 291, 294, 295, 296, 297, 298, 299, 302, 303, 304, 306, 307, 310, 314, 315, 318, 319, 323, 324, 329, 331, 332, 334, 336, 337, 339, 343, 345, 346, 350, 351, 352, 353, 354, 364, 367, 368, 369, 370, 372, 373, 374, 375, 378, 379, 381, 382, 384, 385, 389, 393, 396, 397, 398, 400, 401, 407, 411, 412, 413, 415, 416, 417, 420, 422, 424, 429, 431, 432, 434, 436, 437, 438, 440, 441, 443, 444, 445, 448, 449, 450, 452, 453, 457, 458, 459, 460, 462, 463, 465, 466, 470, 472, 474, 477, 479, 480, 484, 488, 490, 494, 496, 497, 498, 499, 500, 505, 510, 511, 512, 514, 517, 520, 522, 525, 526, 528, 530, 534, 537, 538, 539, 541, 542, 543, 544, 546, 549, 551, 553, 554, 557, 561, 564, 566, 568, 571, 574, 577, 578, 580, 582, 586, 588, 589, 592, 593, 596, 599, 601, 602, 605, 608, 613, 614, 617, 618, 619, 620, 621, 625, 626, 630, 636, 639, 641, 643, 653, 654, 655, 656, 660, 662, 664, 666, 668, 669, 670, 672, 676, 679, 680, 681, 683, 685, 686, 687, 693, 694, 695, 698, 700, 701, 702, 704, 707, 708, 709, 711, 712, 713, 714, 716, 719, 721, 723, 724, 725, 727, 730, 735, 736, 737, 739, 741, 742, 743, 746, 747, 748, 749, 750, 751, 753, 754, 757, 762, 763, 764, 765, 766, 767, 770, 771, 773, 774, 775, 777, 779, 783, 784, 785, 788, 789, 792, 795, 796, 797, 799, 800, 802, 803, 806, 808, 811, 812, 815, 817, 818, 819, 820, 823, 827, 831, 833, 834, 836, 837, 839, 840, 841, 843, 844, 847, 850, 852, 857, 858, 859, 862, 864, 865, 867, 868, 869, 873, 874, 879, 881, 882, 885, 888, 890, 892, 894, 895, 896, 898, 899, 902, 905, 910, 911, 915, 918, 925, 927, 929, 930, 931, 934, 936, 937, 940, 941, 942, 943, 945, 946, 951, 952, 955, 956, 957, 959, 960, 961, 964, 965, 971, 972, 973, 978, 979, 981, 982, 984, 985, 993, 997, 998, 999, 1001, 1003, 1004, 1010, 1011, 1012, 1015, 1018, 1019, 1020, 1022, 1024, 1025, 1028, 1029, 1032, 1035, 1037, 1038, 1046, 1051, 1052, 1053, 1059, 1062, 1063, 1064, 1067, 1069, 1076, 1080, 1081, 1082, 1084, 1086, 1087, 1088, 1091, 1092, 1093, 1096, 1098, 1099, 1100, 1102, 1105, 1107, 1108, 1109, 1110, 1111, 1115, 1123, 1125, 1128, 1129, 1130, 1132, 1133, 1136, 1139, 1140, 1141, 1142, 1145, 1146, 1148, 1150, 1151, 1152, 1153, 1154, 1159, 1160, 1161, 1162, 1163, 1165, 1166, 1168, 1169, 1170, 1172, 1173, 1174, 1176, 1178, 1182, 1183, 1185, 1187, 1190, 1193, 1194, 1196, 1198, 1200, 1203, 1210, 1211, 1212, 1213, 1215, 1216, 1217, 1218, 1219, 1220, 1222, 1227, 1228, 1231, 1232, 1233, 1234, 1235, 1240, 1241, 1247, 1251, 1252, 1253, 1256, 1257, 1258, 1263, 1264, 1265, 1269, 1270, 1274, 1275, 1278, 1279, 1282, 1284, 1286, 1289, 1291, 1292, 1298, 1299, 1300, 1302, 1303, 1304, 1306, 1307, 1309, 1310, 1311, 1312, 1315, 1316, 1317, 1320, 1321, 1323, 1324, 1329, 1330, 1331, 1333, 1335, 1336, 1339, 1342, 1348, 1349, 1352, 1354, 1355, 1357, 1360, 1361, 1363, 1367, 1370, 1372, 1373, 1374, 1375, 1377, 1378, 1380, 1384, 1385, 1386, 1388, 1389, 1394, 1395, 1398, 1401, 1402, 1404, 1406, 1408, 1409, 1410, 1412, 1413, 1414, 1416, 1419, 1420, 1421, 1425, 1427, 1428, 1432, 1436, 1442, 1443, 1445, 1447, 1448, 1450, 1451, 1452, 1454, 1455, 1458, 1460, 1461, 1463, 1464, 1465, 1467, 1468, 1471, 1473, 1474, 1480, 1481, 1488, 1492, 1495, 1498, 1500, 1501, 1502, 1503, 1504, 1507, 1508, 1510, 1511, 1515, 1517, 1522, 1523, 1527, 1528, 1530, 1532, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1543, 1545, 1547, 1548, 1549, 1550, 1551, 1552, 1559, 1562, 1563, 1564, 1566, 1568, 1569, 1571, 1572, 1576, 1583, 1585, 1587, 1590, 1591, 1596, 1597, 1601, 1603, 1606, 1613, 1614, 1618, 1620, 1621, 1623, 1627, 1628, 1631, 1633, 1635, 1638, 1641, 1642, 1643, 1645, 1649, 1650, 1652, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1664, 1665, 1670, 1671, 1672, 1673, 1674, 1679, 1681, 1682, 1686, 1687, 1688, 1691, 1692, 1693, 1697, 1698, 1699, 1700, 1702, 1703, 1706, 1707, 1708, 1712, 1715, 1717, 1719, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1731, 1740, 1744, 1745, 1746, 1751, 1755, 1760, 1763, 1764, 1765, 1766, 1768, 1771, 1772, 1773, 1774, 1775, 1776, 1780, 1784, 1786, 1787, 1789, 1791, 1793, 1794, 1795, 1796, 1797, 1800, 1801, 1802, 1805, 1808, 1809, 1814, 1816, 1818, 1820, 1821, 1822, 1824, 1826, 1827, 1831, 1832, 1834, 1835, 1837, 1839, 1840, 1844, 1845, 1846, 1850, 1852, 1853, 1858, 1860, 1861, 1862, 1865, 1866, 1868, 1869, 1870, 1872, 1873, 1875, 1881, 1882, 1883, 1886, 1888, 1891, 1893, 1894, 1899, 1903, 1904, 1905, 1906, 1909, 1910, 1916, 1918, 1919, 1920, 1922, 1925, 1926, 1932, 1933, 1934, 1935, 1936, 1938, 1940, 1941, 1942, 1945, 1946, 1957, 1960, 1962, 1963, 1967, 1970, 1971, 1973, 1978, 1984, 1985, 1987, 1989, 1992, 1993, 1996, 1998, 1999, 2000]
        '''
        
        all_false_episode_id = [1, 7, 9, 11, 13, 15, 24, 25, 26, 29, 33, 34, 36, 39, 40, 43, 45, 48, 49, 50, 52, 55, 56, 59, 62, 64, 65, 68, 69, 70, 73, 74, 75, 76, 78, 81, 82, 84, 85, 88, 91, 93, 95, 96, 99, 101, 104, 105, 106, 108, 109, 111, 112, 118, 119, 120, 121, 122, 123, 128, 129, 130, 131, 134, 138, 139, 140, 141, 143, 145, 147, 148, 151, 153, 155, 163, 164, 166, 170, 173, 178, 180, 183, 184, 186, 187, 189, 192, 195, 198, 199, 202, 203, 204, 211, 212, 214, 215, 217, 219, 223, 227, 228, 230, 232, 237, 238, 239, 240, 248, 252, 258, 259, 261, 262, 263, 264, 266, 267, 269, 274, 275, 276, 280, 281, 282, 291, 294, 295, 297, 298, 301, 304, 307, 308, 310, 324, 325, 326, 327, 328, 329, 332, 336, 342, 343, 344, 345, 347, 351, 352, 353, 354, 359, 360, 367, 368, 369, 370, 373, 374, 378, 379, 381, 382, 383, 384, 385, 389, 393, 394, 395, 396, 397, 398, 403, 407, 411, 413, 415, 416, 417, 422, 423, 426, 431, 432, 436, 437, 438, 440, 444, 445, 448, 450, 452, 458, 459, 462, 463, 465, 466, 470, 472, 474, 477, 479, 480, 488, 490, 491, 493, 494, 496, 497, 498, 499, 500, 508, 511, 514, 517, 519, 520, 522, 526, 528, 530, 534, 535, 537, 538, 539, 540, 541, 542, 543, 546, 547, 549, 550, 551, 553, 554, 557, 561, 563, 568, 571, 574, 580, 582, 584, 586, 588, 589, 591, 592, 593, 594, 595, 601, 611, 613, 614, 618, 619, 620, 621, 625, 626, 627, 628, 630, 631, 637, 638, 639, 641, 643, 644, 651, 653, 654, 655, 656, 658, 661, 664, 665, 668, 670, 672, 676, 677, 679, 683, 685, 686, 687, 691, 693, 694, 695, 698, 700, 701, 702, 707, 708, 709, 711, 712, 713, 714, 715, 716, 719, 721, 723, 724, 730, 732, 735, 736, 737, 738, 743, 746, 748, 749, 750, 751, 753, 754, 757, 762, 763, 764, 765, 766, 770, 771, 774, 775, 776, 777, 779, 784, 785, 786, 787, 788, 789, 790, 791, 792, 795, 799, 800, 803, 805, 808, 811, 812, 815, 817, 818, 819, 820, 827, 829, 831, 833, 834, 840, 843, 844, 847, 848, 849, 850, 851, 852, 855, 856, 857, 859, 860, 861, 862, 863, 867, 869, 873, 878, 879, 881, 882, 885, 888, 895, 898, 899, 902, 906, 909, 913, 915, 925, 927, 928, 929, 930, 934, 936, 942, 943, 945, 949, 952, 955, 957, 960, 961, 964, 965, 967, 971, 981, 982, 985, 993, 995, 996, 998, 999, 1001, 1004, 1010, 1011, 1012, 1020, 1021, 1024, 1025, 1027, 1029, 1032, 1037, 1038, 1045, 1046, 1047, 1052, 1053, 1057, 1058, 1059, 1061, 1063, 1064, 1069, 1072, 1076, 1078, 1079, 1080, 1081, 1082, 1085, 1086, 1087, 1089, 1092, 1098, 1100, 1102, 1105, 1107, 1109, 1110, 1115, 1122, 1123, 1125, 1126, 1128, 1129, 1136, 1140, 1142, 1145, 1146, 1148, 1150, 1151, 1152, 1153, 1154, 1159, 1160, 1163, 1164, 1165, 1170, 1174, 1176, 1178, 1180, 1183, 1184, 1185, 1187, 1188, 1189, 1192, 1196, 1198, 1199, 1200, 1208, 1210, 1211, 1212, 1213, 1215, 1216, 1217, 1219, 1228, 1230, 1231, 1232, 1233, 1234, 1241, 1248, 1251, 1252, 1253, 1255, 1256, 1258, 1259, 1261, 1264, 1265, 1269, 1278, 1282, 1284, 1286, 1289, 1291, 1292, 1298, 1299, 1300, 1302, 1303, 1304, 1306, 1307, 1310, 1311, 1312, 1315, 1316, 1320, 1321, 1323, 1324, 1328, 1329, 1330, 1333, 1334, 1335, 1336, 1342, 1353, 1354, 1355, 1357, 1358, 1359, 1361, 1362, 1365, 1367, 1369, 1370, 1372, 1375, 1378, 1380, 1384, 1385, 1388, 1389, 1390, 1394, 1398, 1401, 1402, 1404, 1405, 1408, 1409, 1410, 1413, 1414, 1416, 1419, 1420, 1421, 1425, 1427, 1428, 1436, 1442, 1443, 1445, 1447, 1449, 1450, 1451, 1452, 1454, 1455, 1460, 1462, 1463, 1464, 1465, 1467, 1469, 1471, 1473, 1474, 1480, 1481, 1482, 1488, 1492, 1494, 1495, 1496, 1498, 1500, 1502, 1503, 1504, 1511, 1515, 1516, 1518, 1522, 1523, 1527, 1530, 1535, 1536, 1538, 1539, 1542, 1543, 1545, 1548, 1549, 1551, 1552, 1559, 1562, 1563, 1565, 1567, 1568, 1569, 1572, 1576, 1582, 1583, 1587, 1590, 1591, 1596, 1597, 1601, 1603, 1607, 1612, 1613, 1614, 1616, 1617, 1618, 1620, 1623, 1627, 1628, 1631, 1633, 1636, 1638, 1641, 1642, 1643, 1644, 1645, 1646, 1649, 1650, 1652, 1654, 1655, 1656, 1660, 1661, 1664, 1665, 1670, 1671, 1672, 1673, 1674, 1675, 1677, 1679, 1681, 1682, 1683, 1685, 1686, 1687, 1691, 1692, 1693, 1695, 1698, 1699, 1700, 1704, 1706, 1707, 1708, 1712, 1714, 1715, 1717, 1718, 1719, 1721, 1723, 1724, 1726, 1727, 1728, 1729, 1731, 1736, 1744, 1745, 1746, 1751, 1760, 1764, 1765, 1766, 1768, 1771, 1774, 1775, 1776, 1780, 1781, 1784, 1786, 1789, 1791, 1793, 1794, 1796, 1797, 1798, 1801, 1802, 1805, 1809, 1811, 1812, 1816, 1818, 1819, 1821, 1822, 1824, 1826, 1827, 1831, 1835, 1837, 1839, 1840, 1844, 1845, 1846, 1850, 1852, 1857, 1860, 1861, 1862, 1865, 1866, 1867, 1868, 1869, 1873, 1875, 1881, 1882, 1883, 1886, 1890, 1891, 1893, 1894, 1897, 1899, 1901, 1902, 1904, 1905, 1906, 1908, 1909, 1910, 1918, 1920, 1922, 1924, 1925, 1926, 1929, 1933, 1934, 1935, 1938, 1941, 1942, 1955, 1960, 1962, 1963, 1965, 1970, 1971, 1973, 1978, 1984, 1985, 1986, 1987, 1989, 1993, 1996, 1999]
        
        
        part_false_episode_id = [temp_id for temp_id in all_false_episode_id if (temp_id>main_args.init_episode)]

        for temp_index, temp_e in enumerate(all_habitat_episodes):
            if ((temp_index+1) in part_false_episode_id):
                selected_episodes.append(all_habitat_episodes[temp_index])
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

                    real_index = part_false_episode_id[index_in_episodes]
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
