# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
from typing import Any, Union, Dict, Optional, List

from vlfm.arguments import args as main_args


import os
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from torch import Tensor

from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.obs_transformers.utils import image_resize
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.utils.geometry_utils import get_fov, rho_theta
from vlfm.vlm.blip2 import BLIP2Client
from vlfm.vlm.coco_classes import COCO_CLASSES
from vlfm.vlm.grounding_dino import GroundingDINOClient, ObjectDetections
from vlfm.vlm.sam import MobileSAMClient
from vlfm.vlm.yolov7 import YOLOv7Client


from vlfm.utils.vlfm_trainer import VLFMTrainer

import copy

from vlfm.utils.img_utils import (
    monochannel_to_inferno_rgb,
    pixel_value_within_radius,
    place_img_in_img,
    rotate_image,
)

try:
    from habitat_baselines.common.tensor_dict import TensorDict

    from vlfm.policy.base_policy import BasePolicy
except Exception:

    class BasePolicy:  # type: ignore
        pass

import torch.nn.functional as F

from torch.distributions.categorical import Categorical

from sklearn.cluster import DBSCAN

import time

from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix
from depth_camera_filtering import filter_depth

from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer

from vlfm.mapping.traj_visualizer import TrajectoryVisualizer
from vlfm.utils.geometry_utils import closest_point_within_threshold

class TorchActionIDs:
    STOP = torch.tensor([[0]], dtype=torch.long, device='cuda')
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long, device='cuda')
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long, device='cuda')
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long, device='cuda')
    LOOK_UP = torch.tensor([[4]], dtype=torch.long, device='cuda')
    LOOK_DOWN = torch.tensor([[5]], dtype=torch.long, device='cuda')
    OTHER = torch.tensor([[100]], dtype=torch.long, device='cuda')

def xyz_yaw_pitch_roll_to_tf_matrix(xyz: np.ndarray, yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Converts a given position and yaw, pitch, roll angles to a 4x4 transformation matrix.

    Args:
        xyz (np.ndarray): A 3D vector representing the position.
        yaw (float): The yaw angle in radians (rotation around Z-axis).
        pitch (float): The pitch angle in radians (rotation around Y-axis).
        roll (float): The roll angle in radians (rotation around X-axis).

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    x, y, z = xyz
    
    # Rotation matrices for yaw, pitch, roll
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])
    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)],
    ])
    
    # Combined rotation matrix
    R = R_yaw @ R_pitch @ R_roll
    
    # Construct 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R  # Rotation
    transformation_matrix[:3, 3] = [x, y, z]  # Translation

    return transformation_matrix


def new_normalize_angle(x):
    x = np.mod(x + np.pi, 2 * np.pi) - np.pi  # 约束到 [-π, π]
    return x


# def find_dense_subcluster_dbscan(points, real_min_index, eps=1.0, min_samples=1):
#     """
#     使用DBSCAN找到最密集的子簇
#     :param points: 点云数据
#     :param eps: 邻域半径
#     :param min_samples: 核心点所需的最小邻域点数
#     :return: 最密集子簇的点集和中心
#     """
#     if(len(points)==1):
#         return points[0]

#     time_1 = time.time()
    
#     # 使用DBSCAN聚类
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)

#     time_2 = time.time()

#     all_intention_labels = dbscan.fit_predict(points)

#     time_3 = time.time()

#     real_label = all_intention_labels[real_min_index]

#     dense_mask = (real_label == all_intention_labels)
#     dense_points = points[dense_mask]
    
#     # 计算中心
#     center = np.mean(dense_points, axis=0)

#     time_4 = time.time()

#     print("delta_time1:", time_2-time_1)
#     print("delta_time2:", time_3-time_2)
#     print("delta_time3:", time_4-time_3)

#     return center

def find_dense_subcluster_dbscan(points, real_sub_goal_array, eps=1.0, min_samples=1):
    """
    使用DBSCAN找到最密集的子簇
    :param points: 点云数据
    :param eps: 邻域半径
    :param min_samples: 核心点所需的最小邻域点数
    :return: 最密集子簇的点集和中心
    """
    if(len(points)==1):
        return real_sub_goal_array

    distances = np.linalg.norm(points - real_sub_goal_array, axis=1)

    # 找到距离在max_distance范围内的点的索引
    in_range_indices = np.where(distances <= eps)[0]
    
    # 获取在范围内的点
    points_in_range = points[in_range_indices]

    if(len(points_in_range)==0):
        return real_sub_goal_array
    else:
        center = np.mean(points_in_range, axis=0)
        return center

# resize_gai
def compress_map_ultrafast_not_zero(original_map, new_shape=(500, 500)):
    """
    超快速版本，使用最少的操作
    """
    h_ratio, w_ratio = new_shape[0] / original_map.shape[0], new_shape[1] / original_map.shape[1]
    result = np.zeros(new_shape)
    
    # 一次性处理所有特殊值
    special_mask = (original_map != 0)
    if not np.any(special_mask):
        return result
    
    # 获取所有特殊值的坐标和值
    rows, cols = np.where(special_mask)
    values = original_map[rows, cols]
    
    # 计算新坐标
    new_rows = np.clip((rows * h_ratio).astype(int), 0, new_shape[0] - 1)
    new_cols = np.clip((cols * w_ratio).astype(int), 0, new_shape[1] - 1)
    
    # 一次性赋值
    result[new_rows, new_cols] = values
    
    return result


def find_closest_vector(A, B):
    """
    在矩阵 B 中寻找与向量 A 欧式距离最小的向量
    
    参数:
    A -- 长度为2的np.array向量
    B -- N行2列的np.array矩阵
    
    返回:
    与A距离最小的B中的向量
    """
    # 计算B中每个向量与A的欧式距离
    distances = np.sqrt(np.sum((B - A) ** 2, axis=1))
    
    # 找到最小距离的索引
    min_index = np.argmin(distances)
    
    # 返回距离最小的向量
    return B[min_index], min_index

def normalize_angle(x):
    x = np.mod(x + np.pi, 2 * np.pi) - np.pi  # 约束到 [-π, π]
    return (x + np.pi) / (2 * np.pi)

def sort_by_euclidean_distance(A, B):
    """
    将矩阵B中的每行向量按照与向量A的欧式距离从大到小排序
    
    参数:
    A: 1D array, 参考向量
    B: 2D array, 需要排序的矩阵
    
    返回:
    sorted_B: 排序后的矩阵
    """
    # 计算B中每个向量与A的欧式距离
    # distances = np.linalg.norm(B - A, axis=1)
    distances = np.sqrt(np.sum((B - A) ** 2, axis=1))
    
    # 按照距离从大到小排序，获取排序后的索引
    sorted_indices = np.argsort(distances)[::-1]
    
    # 根据排序后的索引重新排列矩阵B
    sorted_B = B[sorted_indices]
    
    return sorted_B



PROMPT_SEPARATOR = "|"

# # 楼梯和楼层管理相关
# self._map_controller = Map_Controller(text_prompt=text_prompt,
#                                       object_map_erosion_size=object_map_erosion_size,
#                                       min_obstacle_height = min_obstacle_height,
#                                       max_obstacle_height = max_obstacle_height,
#                                       obstacle_map_area_threshold = obstacle_map_area_threshold,
#                                       agent_radius = agent_radius,
#                                       hole_area_thresh = hole_area_thresh,
#                                       use_max_confidence = use_max_confidence,
#                                       coco_threshold = coco_threshold,
#                                       non_coco_threshold = non_coco_threshold)

def angle_difference(angle1, angle2):
    """
    计算两个角度之间的最小角度差
    输入：angle1, angle2 ∈ [-π, π]
    输出：角度差 ∈ [0, π]
    """
    # 计算原始差值
    diff = angle1 - angle2
    
    # 归一化到 [-π, π)
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    
    # 取绝对值得到 [0, π]
    return diff


class BaseObjectNavPolicy(BasePolicy):
    _target_object: str = ""
    _policy_info: Dict[str, Any] = {}
    _object_masks: Union[np.ndarray, Any] = None  # set by ._update_object_map()
    # _stop_action: Union[Tensor, Any] = None  # MUST BE SET BY SUBCLASS
    _observations_cache: Dict[str, Any] = {}
    _non_coco_caption = ""
    _load_yolo: bool = True
    # _load_yolo: bool = False

    def __init__(
        self,
        text_prompt: str,
        use_max_confidence: bool,
        sync_explored_areas: bool,
        pointnav_policy_path: str,
        depth_image_shape: Tuple[int, int],
        pointnav_stop_radius: float,
        object_map_erosion_size: float,
        visualize: bool = True,
        compute_frontiers: bool = True,
        min_obstacle_height: float = 0.15,
        max_obstacle_height: float = 0.88,
        agent_radius: float = 0.18,
        obstacle_map_area_threshold: float = 1.5,
        hole_area_thresh: int = 100000,
        use_vqa: bool = False,
        vqa_prompt: str = "Is this ",
        coco_threshold: float = 0.8,
        non_coco_threshold: float = 0.4,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # self._object_detector = GroundingDINOClient(port=int(os.environ.get("GROUNDING_DINO_PORT", "12081")))
        # self._coco_object_detector = YOLOv7Client(port=int(os.environ.get("YOLOV7_PORT", "12084")))
        # self._mobile_sam = MobileSAMClient(port=int(os.environ.get("SAM_PORT", "12083")))

        self._object_detector = GroundingDINOClient(port=int(main_args.dino_port))
        self._coco_object_detector = YOLOv7Client(port=int(main_args.yolo_port))
        self._mobile_sam = MobileSAMClient(port=int(main_args.sam_port))
        self._itm = BLIP2ITMClient(port=int(main_args.blip_port))
        self._text_prompt = text_prompt

        self._use_vqa = use_vqa
        if use_vqa:
            self._vqa = BLIP2Client(port=int(os.environ.get("BLIP2_PORT", "12185")))
        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._acyclic_enforcer = AcyclicEnforcer()

        self._depth_image_shape = tuple(depth_image_shape)
        self._pointnav_stop_radius = pointnav_stop_radius
        self._visualize = visualize
        self._vqa_prompt = vqa_prompt
        self._coco_threshold = coco_threshold
        self._non_coco_threshold = non_coco_threshold
        self.agent_radius = agent_radius

        self._num_steps = 0
        self._did_reset = False
        self._last_goal = np.zeros(2)
        self._called_stop = False
        self._compute_frontiers = compute_frontiers
        self.current_step_stairs_res = {} # 只记录当前帧检测到的stairs

        # 三个last_map没有楼层的概念，到了新楼层后，第一帧均用启发式构建
        self.last_target_map = None
        self.last_semantic_frontier_map = None
        self.last_value_map = None

        # 初始化所有楼层的地图列表 (统一处理，避免重复逻辑)
        # 楼层越高越往后
        self._object_map_list: List[ObjectPointCloudMap] = []
        self._obstacle_map_list: List[ObstacleMap] = []
        self._value_map_list: List[ValueMap] = []

        self.min_obstacle_height = min_obstacle_height
        self.max_obstacle_height = max_obstacle_height
        self.obstacle_map_area_threshold = obstacle_map_area_threshold
        self.hole_area_thresh = hole_area_thresh
        self.use_max_confidence = use_max_confidence

        self._object_map_list.append(ObjectPointCloudMap(erosion_size=object_map_erosion_size))
        self._obstacle_map_list.append(ObstacleMap(
            min_height=min_obstacle_height,
            max_height=max_obstacle_height,
            area_thresh=obstacle_map_area_threshold,
            agent_radius=agent_radius,
            hole_area_thresh=hole_area_thresh,
        ))
        self._value_map_list.append(ValueMap(
            value_channels=len(text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map if sync_explored_areas else None,
        ))
        
        self.floor_num = len(self._obstacle_map_list)

        # 当前活跃地图的引用
        self._cur_floor_index = 0
        self._object_map: ObjectPointCloudMap = self._object_map_list[self._cur_floor_index]
        self._obstacle_map: ObstacleMap = self._obstacle_map_list[self._cur_floor_index]
        self._value_map: ValueMap = self._value_map_list[self._cur_floor_index]


        # 初始化所有frontier相关的列表 (统一处理，避免重复逻辑)
        # 楼层越高越往后
        self._all_floor_frontier_list = [np.array([])]
        self._all_floor_false_frontier_list = [[]] # frontier_revise
        self._all_floor_false_stair_frontier_list = [[]] # frontier_stair_revise
        self._all_floor_frontier_navigate_cnt_dict_list = [{}] # frontier_multi_navigate_revise

        self.same_frontier_dis_thre = 0.15 # 原本0.5m       

        # 爬楼梯状态
        self._done_initializing = False
        self._initialize_step = 0
        self._reach_stair = False
        self._reach_stair_centroid =False
        self._carrot_goal_xy = []
        self._last_carrot_xy = []
        self._last_carrot_px = []
        self._climb_stair_flag = 0
        self._stair_dilate_flag = False
        self._stair_frontier = None
        self._climb_stair_over = True
        self._temp_stair_map = []
        self._get_close_to_stair_step = 0
        self._frontier_stick_step = 0

        self._passive_up_stair_steps = 0
        self._passive_down_stair_steps = 0
        self.PASSIVE_STAIR_DETECTION_THRESHOLD = 3  # 连续3步在楼梯区域内即触发

        self._pitch_angle_offset = 30

        ## 爬楼梯相关状态重置
        # 防止之前episode爬楼梯异常退出
        self._pitch_angle = 0

        ## 目标检测和探索相关状态重置
        # 防止识别正确之后造成误识别
        self.target_might_detected = False
        self._last_frontier_distance = 0
        self.climb_stair_frontier_ls = []
        self.all_detection_list = None

        ## 辅助缓存和历史记录重置
        self.history_action = []

        ## 导航和探索状态重置
        self._try_to_navigate = False
        self._try_to_navigate_step = 0
        
        self.min_distance_xy = np.inf
        self.cur_frontier = np.array([])

        self._last_stair_frontier = np.zeros(2)
        self.current_floor_steps = 0
        self.current_close_to_stair_steps = 0

        self._last_value: float = float("-inf")
        self._last_frontier: np.ndarray = np.zeros(2)

        # self.PRIOR_VALUE = -14.258771726091627
        self.PRIOR_VALUE = 16.59638837848977

        self.last_mode = ""

        # self.climb_stair_init_angle = None
        # self.climb_stair_pos = None
        # self.climb_stair_heading = None
        # self.climb_stair_action = None
        # self.climb_stair_escape_stuck = 0



    def _reset_map_controller(self) -> None:
        self._cur_floor_index = 0
        # 确保只保留第一层的地图实例并重置它们
        # 如果需要删除多余楼层，则执行以下操作：
        # del self._object_map_list[1:]
        # del self._value_map_list[1:]
        # del self._obstacle_map_list[1:]

        self._object_map_list = [copy.deepcopy(self._object_map_list[0])]
        self._value_map_list = [copy.deepcopy(self._value_map_list[0])]
        self._obstacle_map_list = [copy.deepcopy(self._obstacle_map_list[0])]

        # 重新建立对当前（第一层）地图的引用
        self._object_map = self._object_map_list[0]
        self._value_map = self._value_map_list[0]
        self._obstacle_map = self._obstacle_map_list[0]

        # 重置地图内部状态
        self._object_map.reset()
        self._value_map.reset()
        self._obstacle_map.reset()

        self._object_map._episode_pixel_origin = np.array([1000 // 2, 1000 // 2]) # 先列后行
        self._value_map._episode_pixel_origin = np.array([1000 // 2, 1000 // 2]) # 先列后行
        self._obstacle_map._episode_pixel_origin = np.array([1000 // 2, 1000 // 2]) # 先列后行

        self._object_map._traj_vis = TrajectoryVisualizer(self._object_map._episode_pixel_origin, self._object_map.pixels_per_meter)
        self._value_map._traj_vis = TrajectoryVisualizer(self._value_map._episode_pixel_origin, self._value_map.pixels_per_meter)
        self._obstacle_map._traj_vis = TrajectoryVisualizer(self._obstacle_map._episode_pixel_origin, self._obstacle_map.pixels_per_meter)


        # 初始化所有frontier相关的列表 (统一处理，避免重复逻辑)
        # 楼层越高越往后
        self._all_floor_frontier_list = [np.array([])]
        self._all_floor_false_frontier_list = [[]] # frontier_revise
        self._all_floor_false_stair_frontier_list = [[]] # frontier_stair_revise
        self._all_floor_frontier_navigate_cnt_dict_list = [{}] # frontier_multi_navigate_revise

        self.same_frontier_dis_thre = 0.15 # 原本0.5m       

        ## 楼层管理
        self.floor_num = len(self._obstacle_map_list) # 更新楼层数 (此时应为1)
        assert self.floor_num==1

        ## 爬楼梯状态
        self._initialize_step = 0
        self._done_initializing = False
        self._reach_stair = False
        self._reach_stair_centroid = False
        self._carrot_goal_xy = []
        self._last_carrot_xy = []
        self._last_carrot_px = []
        self._stair_dilate_flag = False
        self._climb_stair_flag = 0 
        self._climb_stair_over = True
        self._temp_stair_map = []
        self._get_close_to_stair_step = 0 # VLM/爬楼梯相关
        self._frontier_stick_step = 0

        # 找目标状态
        self._passive_up_stair_steps = 0
        self._passive_down_stair_steps = 0

        self._last_stair_frontier = np.zeros(2)


    def _reset(self) -> None:
        self._pointnav_policy.reset() # 获得全0的pointnav_test_recurrent_hidden_states和pointnav_prev_actions
        self._last_goal = np.zeros(2)
        self._num_steps = 0
        self._called_stop = False
        self._did_reset = True
        self._acyclic_enforcer = AcyclicEnforcer()

        ## 地图和楼层管理重置
        self._reset_map_controller()

        ## 目标检测
        self._target_object = ""

        ## 导航和探索状态重置
        self._try_to_navigate_step = 0
        self._try_to_navigate = False

        ## 爬楼梯相关状态重置
        # 防止之前episode爬楼梯异常退出
        self._pitch_angle = 0

        ## 目标检测和探索相关状态重置
        # 防止识别正确之后造成误识别
        self.target_might_detected = False
        self._last_frontier_distance = 0
        self.climb_stair_frontier_ls = []
        self.all_detection_list = None

        self.min_distance_xy = np.inf
        self.cur_frontier = np.array([])

        ## 辅助缓存和历史记录重置
        self.history_action.clear()

        self.last_target_map = None
        self.last_semantic_frontier_map = None
        self.last_value_map = None

        self.current_step_stairs_res = {}
        self._start_yaw = None

        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)
        self.current_floor_steps = 0
        self.current_close_to_stair_steps = 0

        self.last_mode = ""

        # self.climb_stair_init_angle = None
        # self.climb_stair_pos = None
        # self.climb_stair_heading = None
        # self.climb_stair_action = None
        # self.climb_stair_escape_stuck = 0



    def _xy_to_px(self, points: np.ndarray) -> np.ndarray: # 只有在重新初始化新楼层的map时才用
        """Converts an array of (x, y) coordinates to pixel coordinates.

        Args:
            points: The array of (x, y) coordinates to convert.

        Returns:
            The array of (x, y) pixel coordinates.
        """
        px = np.rint(points[:, ::-1] * 20) + np.array([500, 500])
        px[:, 0] = 1000 - px[:, 0]
        return px.astype(int)

    def is_robot_in_stair_map_fast(self, robot_px:np.ndarray, stair_map: np.ndarray):
        """
        高效判断以机器人质心为圆心、指定半径（机器人半径）的圆是否覆盖 stair_map 中值为 1 的点。

        Args:
            env: 当前环境标识。
            stair_map (np.ndarray): 地图的 _stair_map。
            robot_xy_2d (np.ndarray): 机器人质心在相机坐标系下的 (x, y) 坐标。
            agent_radius (float): 机器人在相机坐标系中的半径。
            obstacle_map: 包含坐标转换功能和地图信息的对象。

        Returns:
            bool: 如果范围内有值为 1,则返回 True,否则返回 False。
        """
        x, y = robot_px[0, 0], robot_px[0, 1]

        # 转换半径到地图坐标系
        radius_px = self.agent_radius * self._obstacle_map.pixels_per_meter

        # 获取地图边界
        rows, cols = stair_map.shape
        x_min = max(0, int(x - radius_px))
        x_max = min(cols - 1, int(x + radius_px))
        y_min = max(0, int(y - radius_px))
        y_max = min(rows - 1, int(y + radius_px))

        # 提取感兴趣的子矩阵
        sub_matrix = stair_map[y_min:y_max + 1, x_min:x_max + 1]

        # 创建圆形掩码
        y_indices, x_indices = np.ogrid[y_min:y_max + 1, x_min:x_max + 1]
        mask = (y_indices - y) ** 2 + (x_indices - x) ** 2 <= radius_px ** 2

        # 获取sub_matrix中为 True 的坐标
        if np.any(sub_matrix[mask]):  # 在圆形区域内有值为True的元素
            # 找出sub_matrix中值为 True 的位置
            true_coords_in_sub_matrix = np.column_stack(np.where(sub_matrix))  # 获取相对于sub_matrix的坐标

            # 通过mask过滤,只留下圆形区域内为 True 的坐标
            true_coords_filtered = true_coords_in_sub_matrix[mask[true_coords_in_sub_matrix[:, 0], true_coords_in_sub_matrix[:, 1]]]

            # 将相对坐标转换为 stair_map 中的坐标
            true_coords_in_stair_map = true_coords_filtered + [y_min, x_min]
            
            return True, true_coords_in_stair_map
        else:
            return False, None

    def _trigger_stair_climbing(self, climb_direction: int, robot_px: np.ndarray, robot_xy: np.ndarray):
        """
        触发爬楼梯模式（从被动检测）。
        """
        self._climb_stair_over = False
        self._climb_stair_flag = climb_direction
        self._reach_stair = True
        self._reach_stair_centroid = False  # 被动进入时，可能还未到达质心
        self._get_close_to_stair_step = 0
        
        if climb_direction == 1:
            self._obstacle_map._up_stair_start = robot_xy.copy()
            self._stair_frontier = self._obstacle_map._up_stair_frontiers
            print(f"Auto-triggered upstairs climbing (passive detection)")
        else:  # climb_direction == 2
            self._obstacle_map._down_stair_start = robot_xy.copy()
            self._stair_frontier = self._obstacle_map._down_stair_frontiers
            print(f"Auto-triggered downstairs climbing (passive detection)")


    def _detect_passive_stair_entry(self, robot_px: np.ndarray, robot_xy: np.ndarray): 
        """
        检测机器人是否被动进入楼梯区域。
        如果机器人在楼梯区域内停留超过阈值步数，自动触发爬楼梯模式。
        """
        # 检查上楼梯
        # self._obstacle_map[env]._has_up_stair: 初始化和reset为False
        # self._obstacle_map[env]._up_stair_frontiers: 初始化和reset为空array
        if self._obstacle_map._has_up_stair and len(self._obstacle_map._up_stair_frontiers) > 0:
            in_up_stair, _ = self.is_robot_in_stair_map_fast(
                robot_px, self._obstacle_map._up_stair_map
            ) # 高效判断以机器人质心为圆心、指定半径（机器人半径）的圆是否覆盖 stair_map 中值为 1 的点。
            
            if in_up_stair:
                self._passive_up_stair_steps += 1
                self._passive_down_stair_steps = 0  # 重置下楼梯计数
                
                if self._passive_up_stair_steps >= self.PASSIVE_STAIR_DETECTION_THRESHOLD:
                    # 检查是否已经探索过上层
                    next_floor_idx = self._cur_floor_index + 1
                    if next_floor_idx >= len(self._obstacle_map_list) or \
                       not self._obstacle_map_list[next_floor_idx]._this_floor_explored: # 还未来得及建立楼上的map or 楼上还没探索
                        print(f"Passive upstairs detection triggered!")
                        self._trigger_stair_climbing(climb_direction=1, robot_px=robot_px, robot_xy=robot_xy)
                    self._passive_up_stair_steps = 0
            else:
                self._passive_up_stair_steps = 0
        
        # 检查下楼梯
        if self._obstacle_map._has_down_stair and len(self._obstacle_map._down_stair_frontiers) > 0:
            in_down_stair, _ = self.is_robot_in_stair_map_fast(
                robot_px, self._obstacle_map._down_stair_map
            )
            
            if in_down_stair:
                self._passive_down_stair_steps += 1
                self._passive_up_stair_steps = 0  # 重置上楼梯计数
                
                if self._passive_down_stair_steps >= self.PASSIVE_STAIR_DETECTION_THRESHOLD:
                    # 检查是否已经探索过下层
                    prev_floor_idx = self._cur_floor_index - 1
                    if prev_floor_idx < 0 or \
                       not self._obstacle_map_list[prev_floor_idx]._this_floor_explored:
                        print(f"Passive downstairs detection triggered!")
                        self._trigger_stair_climbing(climb_direction=2, robot_px=robot_px, robot_xy=robot_xy)
                    self._passive_down_stair_steps = 0
            else:
                self._passive_down_stair_steps = 0

    def _reset_stair_climb_state(self):
        """重置楼梯攀爬相关的状态变量"""
        self._reach_stair = False
        self._reach_stair_centroid = False
        self._stair_dilate_flag = False
        self._climb_stair_flag = 0
        self._obstacle_map._climb_stair_paused_step = 0
        self._last_carrot_xy = []
        self._last_carrot_px = []

    def _remove_floor_map(self, index: int):
        """移除指定的楼层地图"""
        del self._object_map_list[index]
        del self._value_map_list[index]
        del self._obstacle_map_list[index]

        del self._all_floor_frontier_list[index]
        del self._all_floor_false_frontier_list[index]
        del self._all_floor_false_stair_frontier_list[index]
        del self._all_floor_frontier_navigate_cnt_dict_list[index]

        self.floor_num -= 1

    def _update_linked_stair_map(self, original_stair_map: np.ndarray, stair_frontiers: np.ndarray, 
                                target_map_attr: str, start_attr: str, end_attr: str, frontier_attr: str,
                                prev_start_attr: str, prev_end_attr: str, prev_frontier_attr: str):
        """
        更新新楼层中与已爬楼梯对应的楼梯地图信息。
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            original_stair_map.astype(np.uint8), connectivity=8
        )
        
        closest_label = -1
        min_distance = float('inf')
        
        for i in range(1, num_labels):
            centroid_px = centroids[i]
            centroid = self._obstacle_map._px_to_xy(np.atleast_2d(centroid_px))
            distance = np.abs(stair_frontiers[0][0] - centroid[0][0]) + \
                    np.abs(stair_frontiers[0][1] - centroid[0][1])
            if distance < min_distance:
                min_distance = distance
                closest_label = i
        
        if closest_label != -1:
            filtered_stair_map = original_stair_map.copy()
            filtered_stair_map[labels != closest_label] = 0
            
            # 设置新楼层的楼梯地图
            setattr(self._obstacle_map, target_map_attr, filtered_stair_map)
            
            # ✅ 修复：直接从传入的楼梯信息获取，而不是从其他楼层
            # 因为 _handle_new_floor_initialization 已经在切换楼层后调用此方法
            # 所以这里需要保存上一个楼层（调用前的当前楼层）的起止点
            
            # 获取上一个楼层的索引(与ascent不同，现在已经改正)
            prev_floor_idx = self._cur_floor_index - 1 if "down" in target_map_attr else self._cur_floor_index + 1
            
            if 0 <= prev_floor_idx < len(self._obstacle_map_list):
                prev_obstacle_map = self._obstacle_map_list[prev_floor_idx]
                
                # 起止点互换（上楼时的终点是下楼时的起点）
                setattr(self._obstacle_map, start_attr, getattr(prev_obstacle_map, prev_end_attr).copy())
                setattr(self._obstacle_map, end_attr, getattr(prev_obstacle_map, prev_start_attr).copy())
                setattr(self._obstacle_map, frontier_attr, getattr(prev_obstacle_map, prev_frontier_attr).copy())

    def _handle_new_floor_initialization(self, climb_direction: int):
        """处理新楼层的初始化和地图更新"""
        self._done_initializing = False
        self._initialize_step = 0
        
        if climb_direction == 1: # 上楼
            # 标记当前楼层的上楼梯已探索
            self._obstacle_map._explored_up_stair = True
            # 标记新楼层（上一层）的下楼梯已探索
            self._obstacle_map_list[self._cur_floor_index+1]._explored_down_stair = True
            
            # ✅ 修复：使用当前楼层的索引
            ori_up_stair_map = self._obstacle_map._up_stair_map.copy()
            stair_frontiers = self._obstacle_map._up_stair_frontiers
            
            # 切换到新楼层
            self._cur_floor_index += 1
            self._update_current_maps()
            
            # 将当前楼层的上楼梯信息保存到新楼层的下楼梯属性
            self._update_linked_stair_map(
                ori_up_stair_map, stair_frontiers, 
                target_map_attr="_down_stair_map", 
                start_attr="_down_stair_start", 
                end_attr="_down_stair_end", 
                frontier_attr="_down_stair_frontiers",
                prev_start_attr="_up_stair_start",
                prev_end_attr="_up_stair_end",
                prev_frontier_attr="_up_stair_frontiers"
            )
            
            # 标记新楼层已有下楼梯
            self._obstacle_map._has_down_stair = True

        else: # climb_direction == 2 (下楼)
            # 标记当前楼层的下楼梯已探索
            self._obstacle_map._explored_down_stair = True
            # 标记新楼层（下一层）的上楼梯已探索
            self._obstacle_map_list[self._cur_floor_index-1]._explored_up_stair = True

            # ✅ 修复：使用当前楼层的索引
            ori_down_stair_map = self._obstacle_map._down_stair_map.copy()
            stair_frontiers = self._obstacle_map._down_stair_frontiers

            # 切换到新楼层
            self._cur_floor_index -= 1
            self._update_current_maps()
            
            # 将当前楼层的下楼梯信息保存到新楼层的上楼梯属性
            self._update_linked_stair_map(
                ori_down_stair_map, stair_frontiers, 
                target_map_attr="_up_stair_map", 
                start_attr="_up_stair_start", 
                end_attr="_up_stair_end", 
                frontier_attr="_up_stair_frontiers",
                prev_start_attr="_down_stair_start",
                prev_end_attr="_down_stair_end",
                prev_frontier_attr="_down_stair_frontiers"
            )
            
            # 标记新楼层已有上楼梯
            self._obstacle_map._has_up_stair = True



    def _update_current_maps(self):
        """根据当前楼层索引更新当前环境的地图引用"""
        self._object_map = self._object_map_list[self._cur_floor_index]
        self._obstacle_map = self._obstacle_map_list[self._cur_floor_index]
        self._value_map = self._value_map_list[self._cur_floor_index]




    def _process_stair_climb_state(self, robot_xy: np.ndarray, robot_px: np.ndarray, stair_map: np.ndarray, climb_direction: int):
        """
        处理机器人达到或离开楼梯的状态逻辑。
        climb_direction: 1 表示上楼，2 表示下楼
        """
        already_reach_stair, reach_yx = self.is_robot_in_stair_map_fast(robot_px, stair_map) # 高效判断以机器人质心为圆心、指定半径（机器人半径）的圆是否覆盖 stair_map 中值为 1 的点。

        if not self._reach_stair:
            if already_reach_stair: # 刚刚到达楼梯
                self._reach_stair = True
                self._get_close_to_stair_step = 0
                if climb_direction == 1:
                    self._obstacle_map._up_stair_start = robot_xy.copy()
                else: # climb_direction == 2
                    self._obstacle_map._down_stair_start = robot_xy.copy()
            # else表示确认没有到达楼梯
        elif not self._reach_stair_centroid:
            if self._stair_frontier is not None and \
               np.linalg.norm(self._stair_frontier - np.atleast_2d(robot_xy)) <= 0.3: # 注意这里原来是robot_xy_2d，改为robot_px[0]以匹配px_to_xy的转换
                # 刚刚到达楼梯frontier中心
                self._reach_stair_centroid = True
            # else表示还没到达楼梯frontier中心
        else: # _reach_stair_centroid == True
            if not self.is_robot_in_stair_map_fast(robot_px, stair_map)[0] and \
               self._obstacle_map._climb_stair_paused_step >= 30: # 在某处卡顿时间过程-->不认为是楼梯，结束上下楼过程
                
                self._reset_stair_climb_state()
                self._climb_stair_over = True
                self._obstacle_map._disabled_frontiers.add(tuple(self._stair_frontier[0]))
                print(f"Frontier {self._stair_frontier} is disabled due to no movement.")

                if climb_direction == 1:
                    self._obstacle_map._disabled_stair_map[self._obstacle_map._up_stair_map == 1] = 1
                    self._obstacle_map._up_stair_map.fill(0)
                    self._obstacle_map._has_up_stair = False
                    self._remove_floor_map(self._cur_floor_index + 1)
                else: # climb_direction == 2
                    self._obstacle_map._disabled_stair_map[self._obstacle_map._down_stair_map == 1] = 1
                    self._obstacle_map._down_stair_frontiers.fill(0)
                    self._obstacle_map._has_down_stair = False
                    self._obstacle_map._look_for_downstair_flag = False
                    self._remove_floor_map(self._cur_floor_index - 1)
                    self._cur_floor_index -= 1 # 如果下楼的楼梯是误判，那么当前层需要往下减一层了

            elif not self.is_robot_in_stair_map_fast(robot_px, stair_map)[0]:
                self._reset_stair_climb_state()
                self._climb_stair_over = True
                if climb_direction == 1: # 已经完成“上楼”过程
                    self._obstacle_map._up_stair_end = robot_xy.copy()
                    if not self._obstacle_map_list[self._cur_floor_index+1]._done_initializing:
                        self._handle_new_floor_initialization(climb_direction)
                    else:
                        self._cur_floor_index += 1
                        self._update_current_maps()
                else: # climb_direction == 2 # 已经完成“下楼”过程
                    self._obstacle_map._down_stair_end = robot_xy.copy()
                    if not self._obstacle_map_list[self._cur_floor_index-1]._done_initializing:
                        self._handle_new_floor_initialization(climb_direction)
                    else:
                        self._cur_floor_index -= 1
                        self._update_current_maps()
                print("climb stair success!!!!")

    def _add_floor_map(self, index: int):
        """添加新的楼层地图"""
        new_object_map = ObjectPointCloudMap(erosion_size=5)
        new_obstacle_map = ObstacleMap(
            min_height=self.min_obstacle_height, max_height=self.max_obstacle_height,
            area_thresh=self.obstacle_map_area_threshold, agent_radius=self.agent_radius,
            hole_area_thresh=self.hole_area_thresh
        )
        new_value_map = ValueMap(
            value_channels=len(self._text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence=self.use_max_confidence, obstacle_map=None,
        )

        new_object_map._episode_pixel_origin = self._xy_to_px(np.atleast_2d(self._observations_cache["robot_xy"]))[0]
        new_obstacle_map._episode_pixel_origin = self._xy_to_px(np.atleast_2d(self._observations_cache["robot_xy"]))[0]
        new_value_map._episode_pixel_origin = self._xy_to_px(np.atleast_2d(self._observations_cache["robot_xy"]))[0]


        # new_object_map._traj_vis = TrajectoryVisualizer(new_object_map._episode_pixel_origin[::-1], new_object_map.pixels_per_meter)
        # new_obstacle_map._traj_vis = TrajectoryVisualizer(new_obstacle_map._episode_pixel_origin[::-1], new_obstacle_map.pixels_per_meter)
        # new_value_map._traj_vis = TrajectoryVisualizer(new_value_map._episode_pixel_origin[::-1], new_value_map.pixels_per_meter)

        new_object_map._traj_vis = TrajectoryVisualizer(np.array([1000-new_object_map._episode_pixel_origin[1], 1000-new_object_map._episode_pixel_origin[0]]), new_object_map.pixels_per_meter)
        new_obstacle_map._traj_vis = TrajectoryVisualizer(np.array([1000-new_obstacle_map._episode_pixel_origin[1], 1000-new_obstacle_map._episode_pixel_origin[0]]), new_obstacle_map.pixels_per_meter)
        new_value_map._traj_vis = TrajectoryVisualizer(np.array([1000-new_value_map._episode_pixel_origin[1], 1000-new_value_map._episode_pixel_origin[0]]), new_value_map.pixels_per_meter)
        

        self._object_map_list.insert(index, new_object_map)
        self._obstacle_map_list.insert(index, new_obstacle_map)
        self._value_map_list.insert(index, new_value_map)
        self.floor_num = len(self._obstacle_map_list)

        self._all_floor_frontier_list.insert(index, np.array([]))
        self._all_floor_false_frontier_list.insert(index, [])
        self._all_floor_false_stair_frontier_list.insert(index, [])
        self._all_floor_frontier_navigate_cnt_dict_list.insert(index, {})


    def _update_obstacle_map(self, observations_cache, current_step_stairs_res, pitch_angle) -> None:
        robot_xy = observations_cache["robot_xy"]
        robot_px = self._obstacle_map_list[self._cur_floor_index]._xy_to_px(np.atleast_2d(robot_xy))

        # ✅ 新增：被动检测楼梯（仅在非爬楼梯状态时）
        # self._climb_stair_over[env]: 初始化和reset后均为True
        # self._climb_stair_flag[env]: 初始化和reset后均为0, up是1, down是2
        # if self._climb_stair_over and self._climb_stair_flag == 0 and self.last_mode!="navigate":
        
        # # 修改1
        # if self._climb_stair_over and self._climb_stair_flag == 0:
        #     self._detect_passive_stair_entry(robot_px, robot_xy)
        #     """
        #     检测机器人是否被动进入楼梯区域。
        #     如果机器人在楼梯区域内停留超过阈值步数，自动触发爬楼梯模式。
        #     """
        
        # 原有的爬楼梯状态处理（如果正在爬楼梯，则进入此处）
        if not self._climb_stair_over:
            stair_map_to_use = None
            if self._climb_stair_flag == 1:
                stair_map_to_use = self._obstacle_map._up_stair_map
            elif self._climb_stair_flag == 2:
                stair_map_to_use = self._obstacle_map._down_stair_map

            if stair_map_to_use is not None:
                if not self._stair_dilate_flag:
                    self._temp_stair_map = cv2.dilate(
                        stair_map_to_use.astype(np.uint8),
                        (7, 7),
                        iterations=1,
                    )
                    self._stair_dilate_flag = True
                else:
                    self._temp_stair_map = stair_map_to_use
                
                # 处理楼梯状态
                # 处理机器人达到或离开楼梯的状态逻辑。
                self._process_stair_climb_state(robot_xy, robot_px, self._temp_stair_map, self._climb_stair_flag)

        for temp_index in range(4):
            if (temp_index+1) in self.current_step_stairs_res:
                stair_mask = self.current_step_stairs_res[(temp_index+1)]
            else:
                stair_mask = []

            if(temp_index==0):
                self._obstacle_map.update_map(
                    observations_cache["depth"],
                    observations_cache["tf_camera_to_episodic"],
                    observations_cache["min_depth"],
                    observations_cache["max_depth"],
                    observations_cache["fx"],
                    observations_cache["fy"],
                    observations_cache["camera_fov"],
                    stair_mask,
                    pitch_angle,
                    self._climb_stair_over,
                    self._reach_stair,
                    self._climb_stair_flag,
                )  # 该接口只处理单目
            else:
                if abs(pitch_angle-0)<1e-5:
                    current_camera_yaw = self._observations_cache["robot_heading"]
                    current_camera_position = self._observations_cache["camera_position"]

                    if(temp_index==1): # 2
                        other_angle_camera_yaw = new_normalize_angle(current_camera_yaw+0.5*np.pi)
                    elif(temp_index==2): # 3
                        other_angle_camera_yaw = new_normalize_angle(current_camera_yaw+np.pi)
                    elif(temp_index==3): # 4
                        other_angle_camera_yaw = new_normalize_angle(current_camera_yaw-0.5*np.pi)

                    other_angle_tf_camera_to_episodic = xyz_yaw_to_tf_matrix(current_camera_position, other_angle_camera_yaw)
                    other_angle_depth = VLFMTrainer.depth_ls[temp_index-1].cpu().numpy()
                    other_angle_depth = filter_depth(other_angle_depth.reshape(other_angle_depth.shape[:2]), blur_type=None) # depth过滤

                    self._obstacle_map.update_map(
                        other_angle_depth,
                        other_angle_tf_camera_to_episodic,
                        observations_cache["min_depth"],
                        observations_cache["max_depth"],
                        observations_cache["fx"],
                        observations_cache["fy"],
                        observations_cache["camera_fov"],
                        stair_mask,
                        pitch_angle,
                        self._climb_stair_over,
                        self._reach_stair,
                        self._climb_stair_flag,
                    )  # 该接口只处理单目
        frontiers = self._obstacle_map.frontiers # 楼梯相关的frontier不在这个里面            
        self._obstacle_map.update_agent_traj(observations_cache["robot_xy"], observations_cache["robot_heading"])
        self._all_floor_frontier_list[self._cur_floor_index] = frontiers



        # frontier_revise
        now_right_frontier_ls = []
        for temp_i in range(len(self._all_floor_frontier_list[self._cur_floor_index])):
            should_add = True
            for temp_j in range(len(self._all_floor_false_frontier_list[self._cur_floor_index])):
                temp_dis = ((self._all_floor_frontier_list[self._cur_floor_index][temp_i][0]-self._all_floor_false_frontier_list[self._cur_floor_index][temp_j][0])**2+(self._all_floor_frontier_list[self._cur_floor_index][temp_i][1]-self._all_floor_false_frontier_list[self._cur_floor_index][temp_j][1])**2)**0.5
                if temp_dis<0.5:
                    should_add = False
                    break
            if(should_add==True):
                now_right_frontier_ls.append(self._all_floor_frontier_list[self._cur_floor_index][temp_i])

        
        # frontier_multi_navigate_revise
        final_right_frontier_ls = []
        now_right_frontier_ls = np.array(now_right_frontier_ls)
        for temp_i in range(len(now_right_frontier_ls)):
            should_add = True
            for temp_frontier_key in self._all_floor_frontier_navigate_cnt_dict_list[self._cur_floor_index]:
                if(self._all_floor_frontier_navigate_cnt_dict_list[self._cur_floor_index][temp_frontier_key]>10):
                    temp_dis = ((temp_frontier_key[0]-now_right_frontier_ls[temp_i][0])**2+(temp_frontier_key[1]-now_right_frontier_ls[temp_i][1])**2)**0.5
                    if(temp_dis<self.same_frontier_dis_thre):
                        should_add = False
                        break
            if(should_add==True):
                final_right_frontier_ls.append(now_right_frontier_ls[temp_i])
            
        self._all_floor_frontier_list[self._cur_floor_index] = np.array(final_right_frontier_ls) # 当前帧最终真正需要的frontier


        # 附加：处理新楼层地图的创建
        if self._obstacle_map._has_up_stair and self._cur_floor_index + 1 >= len(self._object_map_list):
            self._add_floor_map(len(self._object_map_list))
        if self._obstacle_map._has_down_stair and self._cur_floor_index == 0: # self._cur_floor_index == 0-->只有之前没有下去过，才会新建下面那层的map
            self._add_floor_map(0)
            self._cur_floor_index += 1 # 当前不是最底层了

        self.floor_num = len(self._obstacle_map_list)

    # def _climb_stair(self, observations: "TensorDict", ori_masks: Tensor) -> Tensor:
    #     """
    #     处理爬楼梯（上楼或下楼）的逻辑，包括视角调整和目标点导航。
    #     """
    #     masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
    #     robot_xy = self._observations_cache["robot_xy"]
    #     heading = self._observations_cache["robot_heading"]

    #     # 根据爬楼梯标志设置目标楼梯前沿
    #     target_stair_frontier = self._obstacle_map._up_stair_frontiers if self._climb_stair_flag == 1 else self._obstacle_map._down_stair_frontiers
        
    #     if target_stair_frontier.size == 0:
    #         print(f"Error: Stair frontier for climb_stair_flag {self._climb_stair_flag} is empty. Returning to explore.")
    #         return TorchActionIDs.OTHER
        
    #     current_distance = np.linalg.norm(target_stair_frontier[0] - robot_xy)
    #     print(f"Climb Stair - Distance Change: {np.abs(self._last_frontier_distance - current_distance):.2f}m, Climb Stair Paused Step: {self._obstacle_map._climb_stair_paused_step}")

    #     # 检测是否卡顿在楼梯上
    #     if np.abs(self._last_frontier_distance - current_distance) > 0.2:
    #         self._obstacle_map._climb_stair_paused_step = 0
    #         self._last_frontier_distance = current_distance

    #     else:
    #         self._obstacle_map._climb_stair_paused_step += 1

    #     if self._obstacle_map._climb_stair_paused_step > 15:
    #         # 如果长时间卡顿，可能楼梯已经走完，或者遇到了障碍
    #         self._obstacle_map._disable_end = True # 标记楼梯终点可能不可达

    #     # 阶段1: 接近楼梯质心 (如果尚未到达)
    #     if not self._reach_stair_centroid:
    #         stair_centroid_point = target_stair_frontier[0] # 楼梯的质心点
    #         rho, theta = rho_theta(robot_xy, heading, stair_centroid_point)
    #         rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
            
    #         obs_pointnav = {
    #             "depth": image_resize(self._observations_cache["nav_depth"], self._depth_image_shape, channels_last=True, interpolation_mode="area"),
    #             "pointgoal_with_gps_compass": rho_theta_tensor,
    #         }
    #         self._policy_info["rho_theta"] = np.array([rho, theta])
    #         action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
            
    #         if action.item() == 0:
    #             self._reach_stair_centroid = True
    #             print("Agent is near stair centroid, switching to move forward.")
    #             action[0] = 1 # 强制向前移动
    #         return action

    #     # 阶段2: 视角调整 (如果需要)
    #     # 仅在下楼梯且俯仰角过高时调整
    #     if self._climb_stair_flag == 2 and self._pitch_angle < -30: 
    #         self._pitch_angle += self._pitch_angle_offset
    #         print("Adjusting pitch angle for downstair (looking up a little).")
    #         return TorchActionIDs.LOOK_UP
        
    #     # 阶段3: 沿楼梯方向前进 (胡萝卜策略)
    #     else:
    #         distance = 0.8 # 目标点距离

    #         depth_map = self._observations_cache["nav_depth"].squeeze(0).cpu().numpy()
    #         if depth_map.size == 0: # 避免空深度图
    #             print("Warning: Depth map is empty. Cannot determine target point.")
    #             return TorchActionIDs.MOVE_FORWARD # 默认向前

    #         max_value = np.max(depth_map)
    #         max_indices = np.argwhere(depth_map == max_value)
            
    #         if max_indices.size == 0: # 避免没有最大值点
    #             print("Warning: No max depth value found. Cannot determine target point.")
    #             return TorchActionIDs.MOVE_FORWARD # 默认向前

    #         center_point = np.mean(max_indices, axis=0).astype(int)
    #         v, u = center_point[0], center_point[1]

    #         normalized_u = np.clip((u - self._cx) / self._cx, -1, 1)
    #         angle_offset = normalized_u * (self._camera_fov / 2)
    #         target_heading = heading - angle_offset # 尝试减去角度偏移
    #         target_heading = target_heading % (2 * np.pi)

    #         x_target = robot_xy[0] + distance * np.cos(target_heading)
    #         y_target = robot_xy[1] + distance * np.sin(target_heading)
    #         current_target_point_xy = np.array([x_target, y_target])
    #         current_target_point_px = self._obstacle_map._xy_to_px(np.atleast_2d(current_target_point_xy))

    #         this_stair_end = self._obstacle_map._up_stair_end if self._climb_stair_flag == 1 else self._obstacle_map._down_stair_end        
            
    #         if(len(this_stair_end)==0):
    #             this_stair_end_px = np.array([])
    #         else:
    #             this_stair_end_px = self._obstacle_map._xy_to_px(np.atleast_2d(this_stair_end))[0]

    #         # 第一次 or 终点不存在 or 接近楼梯终点时 or 终点到不了时：重置胡萝卜目标点
    #         if (len(self._last_carrot_xy) == 0 or this_stair_end_px.size == 0 or 
    #             np.linalg.norm(this_stair_end_px - self._obstacle_map._xy_to_px(np.atleast_2d(robot_xy))[0]) <= 0.5 * self._obstacle_map.pixels_per_meter or 
    #             self._obstacle_map._disable_end):
                
    #             self._carrot_goal_xy = current_target_point_xy
    #             self._obstacle_map._carrot_goal_px = current_target_point_px
    #             self._last_carrot_xy = current_target_point_xy
    #             self._last_carrot_px = current_target_point_px
    #         else:
    #             # 比较当前胡萝卜目标点与上次胡萝卜目标点到楼梯终点的L1距离
    #             l1_distance_current = np.abs(this_stair_end_px[0] - current_target_point_px[0][0]) + np.abs(this_stair_end_px[1] - current_target_point_px[0][1])
    #             l1_distance_last = np.abs(this_stair_end_px[0] - self._last_carrot_px[0][0]) + np.abs(this_stair_end_px[1] - self._last_carrot_px[0][1])
                
    #             if l1_distance_last > l1_distance_current: # 如果新的胡萝卜点离终点更近，则更新
    #                 self._carrot_goal_xy = current_target_point_xy
    #                 self._obstacle_map._carrot_goal_px = current_target_point_px
    #                 self._last_carrot_xy = current_target_point_xy
    #                 self._last_carrot_px = current_target_point_px
    #             # 否则，保持上一个胡萝卜目标点不变，即 self._carrot_goal_xy 和 _carrot_goal_px 已经是 _last_carrot 的值

    #         rho, theta = rho_theta(robot_xy, heading, self._carrot_goal_xy)
    #         rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)

    #         obs_pointnav = {
    #             "depth": image_resize(self._observations_cache["nav_depth"], self._depth_image_shape, channels_last=True, interpolation_mode="area"),
    #             "pointgoal_with_gps_compass": rho_theta_tensor,
    #         }
    #         self._policy_info["rho_theta"] = np.array([rho, theta])
    #         action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
            
    #         if action.item() == 0:
    #             print("Agent might stop, forcing move forward.")
    #             action[0] = 1 # 强制向前移动
    #         return action


    # def _climb_stair(self, observations: "TensorDict", ori_masks: Tensor) -> Tensor:
    #     """
    #     处理爬楼梯（上楼或下楼）的逻辑，包括视角调整和目标点导航。
    #     """
    #     masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
    #     robot_xy = self._observations_cache["robot_xy"]
    #     heading = self._observations_cache["robot_heading"]

    #     # 根据爬楼梯标志设置目标楼梯前沿
    #     target_stair_frontier = self._obstacle_map._up_stair_frontiers if self._climb_stair_flag == 1 else self._obstacle_map._down_stair_frontiers
        
    #     if target_stair_frontier.size == 0:
    #         print(f"Error: Stair frontier for climb_stair_flag {self._climb_stair_flag} is empty. Returning to explore.")
    #         return TorchActionIDs.OTHER
        
    #     current_distance = np.linalg.norm(target_stair_frontier[0] - robot_xy)
        
    #     min_exist_delta_dis = 100000
    #     for temp_frontier in self.climb_stair_frontier_ls:
    #         temp_delta_dis = np.linalg.norm(target_stair_frontier[0] - temp_frontier)
    #         if(temp_delta_dis<min_exist_delta_dis):
    #             min_exist_delta_dis = temp_delta_dis
        
    #     # 检测是否卡顿在楼梯上
    #     if(min_exist_delta_dis>0.5):
    #         self._obstacle_map._climb_stair_paused_step = 0
    #         self.climb_stair_frontier_ls.append(target_stair_frontier[0])
    #     else:
    #         self._obstacle_map._climb_stair_paused_step += 1

    #     print(f"Climb Stair - climb_stair_frontier_ls: {self.climb_stair_frontier_ls}")

    #     if self._obstacle_map._climb_stair_paused_step > 15:
    #         # 如果长时间卡顿，可能楼梯已经走完，或者遇到了障碍
    #         self._obstacle_map._disable_end = True # 标记楼梯终点可能不可达

    #     # 阶段1: 接近楼梯质心 (如果尚未到达)
    #     if not self._reach_stair_centroid:
    #         stair_centroid_point = target_stair_frontier[0] # 楼梯的质心点
    #         rho, theta = rho_theta(robot_xy, heading, stair_centroid_point)
    #         rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
            
    #         obs_pointnav = {
    #             "depth": image_resize(self._observations_cache["nav_depth"], self._depth_image_shape, channels_last=True, interpolation_mode="area"),
    #             "pointgoal_with_gps_compass": rho_theta_tensor,
    #         }
    #         self._policy_info["rho_theta"] = np.array([rho, theta])
    #         action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
            
    #         if action.item() == 0:
    #             self._reach_stair_centroid = True
    #             print("Agent is near stair centroid, switching to move forward.")
    #             action[0] = 1 # 强制向前移动
    #         return action

    #     # 阶段2: 视角调整 (如果需要)
    #     # 仅在下楼梯且俯仰角过高时调整
    #     if self._climb_stair_flag == 2 and self._pitch_angle < -30: 
    #         self._pitch_angle += self._pitch_angle_offset
    #         print("Adjusting pitch angle for downstair (looking up a little).")
    #         return TorchActionIDs.LOOK_UP
        
    #     # 阶段3: 沿楼梯方向前进 (胡萝卜策略)
    #     else:
    #         distance = 0.8 # 目标点距离

    #         depth_map = self._observations_cache["nav_depth"].squeeze(0).cpu().numpy()
    #         if depth_map.size == 0: # 避免空深度图
    #             print("Warning: Depth map is empty. Cannot determine target point.")
    #             return TorchActionIDs.MOVE_FORWARD # 默认向前

    #         max_value = np.max(depth_map)
    #         max_indices = np.argwhere(depth_map == max_value)
            
    #         if max_indices.size == 0: # 避免没有最大值点
    #             print("Warning: No max depth value found. Cannot determine target point.")
    #             return TorchActionIDs.MOVE_FORWARD # 默认向前

    #         center_point = np.mean(max_indices, axis=0).astype(int)
    #         v, u = center_point[0], center_point[1]

    #         normalized_u = np.clip((u - self._cx) / self._cx, -1, 1)
    #         angle_offset = normalized_u * (self._camera_fov / 2)
    #         target_heading = heading - angle_offset # 尝试减去角度偏移
    #         target_heading = target_heading % (2 * np.pi)

    #         x_target = robot_xy[0] + distance * np.cos(target_heading)
    #         y_target = robot_xy[1] + distance * np.sin(target_heading)
    #         current_target_point_xy = np.array([x_target, y_target])
    #         current_target_point_px = self._obstacle_map._xy_to_px(np.atleast_2d(current_target_point_xy))

    #         this_stair_end = self._obstacle_map._up_stair_end if self._climb_stair_flag == 1 else self._obstacle_map._down_stair_end        
            
    #         if(len(this_stair_end)==0):
    #             this_stair_end_px = np.array([])
    #         else:
    #             this_stair_end_px = self._obstacle_map._xy_to_px(np.atleast_2d(this_stair_end))[0]

    #         # 第一次 or 终点不存在 or 接近楼梯终点时 or 终点到不了时：重置胡萝卜目标点
    #         if (len(self._last_carrot_xy) == 0 or this_stair_end_px.size == 0 or 
    #             np.linalg.norm(this_stair_end_px - self._obstacle_map._xy_to_px(np.atleast_2d(robot_xy))[0]) <= 0.5 * self._obstacle_map.pixels_per_meter or 
    #             self._obstacle_map._disable_end):
                
    #             self._carrot_goal_xy = current_target_point_xy
    #             self._obstacle_map._carrot_goal_px = current_target_point_px
    #             self._last_carrot_xy = current_target_point_xy
    #             self._last_carrot_px = current_target_point_px
    #         else:
    #             # 比较当前胡萝卜目标点与上次胡萝卜目标点到楼梯终点的L1距离
    #             l1_distance_current = np.abs(this_stair_end_px[0] - current_target_point_px[0][0]) + np.abs(this_stair_end_px[1] - current_target_point_px[0][1])
    #             l1_distance_last = np.abs(this_stair_end_px[0] - self._last_carrot_px[0][0]) + np.abs(this_stair_end_px[1] - self._last_carrot_px[0][1])
                
    #             if l1_distance_last > l1_distance_current: # 如果新的胡萝卜点离终点更近，则更新
    #                 self._carrot_goal_xy = current_target_point_xy
    #                 self._obstacle_map._carrot_goal_px = current_target_point_px
    #                 self._last_carrot_xy = current_target_point_xy
    #                 self._last_carrot_px = current_target_point_px
    #             # 否则，保持上一个胡萝卜目标点不变，即 self._carrot_goal_xy 和 _carrot_goal_px 已经是 _last_carrot 的值

    #         rho, theta = rho_theta(robot_xy, heading, self._carrot_goal_xy)
    #         rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)

    #         obs_pointnav = {
    #             "depth": image_resize(self._observations_cache["nav_depth"], self._depth_image_shape, channels_last=True, interpolation_mode="area"),
    #             "pointgoal_with_gps_compass": rho_theta_tensor,
    #         }
    #         self._policy_info["rho_theta"] = np.array([rho, theta])
    #         action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
            
    #         if action.item() == 0:
    #             print("Agent might stop, forcing move forward.")
    #             action[0] = 1 # 强制向前移动
    #         return action

    # def _climb_stair(self, observations: "TensorDict", ori_masks: Tensor) -> Tensor:
    #     """
    #     处理爬楼梯（上楼或下楼）的逻辑，包括视角调整和目标点导航。
    #     """
    #     if(self.climb_stair_escape_stuck>0):
    #         heading = self._observations_cache["robot_heading"]
    #         delta_stair_angle = angle_difference(self.climb_stair_init_angle, heading)
    #         print("escape_delta_stair_angle:", delta_stair_angle)
    #         if(self.climb_stair_action[0][0].cpu().numpy()==2):
    #             if(delta_stair_angle>0):
    #                 self.climb_stair_escape_stuck += 1
    #                 return self.climb_stair_action
    #             else:
    #                 self.climb_stair_escape_stuck = 0
    #                 return TorchActionIDs.MOVE_FORWARD
    #         else:
    #             if(delta_stair_angle<0):
    #                 self.climb_stair_escape_stuck += 1
    #                 return self.climb_stair_action
    #             else:
    #                 self.climb_stair_escape_stuck = 0
    #                 return TorchActionIDs.MOVE_FORWARD
        
    #     masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
    #     robot_xy = self._observations_cache["robot_xy"]
    #     heading = self._observations_cache["robot_heading"]

    #     # 根据爬楼梯标志设置目标楼梯前沿
    #     target_stair_frontier = self._obstacle_map._up_stair_frontiers if self._climb_stair_flag == 1 else self._obstacle_map._down_stair_frontiers
        
    #     if target_stair_frontier.size == 0:
    #         print(f"Error: Stair frontier for climb_stair_flag {self._climb_stair_flag} is empty. Returning to explore.")
    #         return TorchActionIDs.OTHER
        
    #     current_distance = np.linalg.norm(target_stair_frontier[0] - robot_xy)
    #     print(f"Climb Stair - Distance Change: {np.abs(self._last_frontier_distance - current_distance):.2f}m, Climb Stair Paused Step: {self._obstacle_map._climb_stair_paused_step}")
        
    #     # 检测是否卡顿在楼梯上
    #     if np.abs(self._last_frontier_distance - current_distance) > 0.2:
    #         self._obstacle_map._climb_stair_paused_step = 0
    #         self._last_frontier_distance = current_distance
    #     else:
    #         self._obstacle_map._climb_stair_paused_step += 1

    #     if self._obstacle_map._climb_stair_paused_step > 15:
    #         # 如果长时间卡顿，可能楼梯已经走完，或者遇到了障碍
    #         self._obstacle_map._disable_end = True # 标记楼梯终点可能不可达

    #     # 阶段1: 接近楼梯质心 (如果尚未到达)
    #     if not self._reach_stair_centroid:
    #         stair_centroid_point = target_stair_frontier[0] # 楼梯的质心点
    #         self.climb_stair_init_angle = heading
    #         rho, theta = rho_theta(robot_xy, heading, stair_centroid_point)
    #         rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
            
    #         obs_pointnav = {
    #             "depth": image_resize(self._observations_cache["nav_depth"], self._depth_image_shape, channels_last=True, interpolation_mode="area"),
    #             "pointgoal_with_gps_compass": rho_theta_tensor,
    #         }
    #         self._policy_info["rho_theta"] = np.array([rho, theta])
    #         action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
            
    #         print("closer_to_stair_centroid")
    #         if action.item() == 0:
    #             self._reach_stair_centroid = True
    #             print("Agent is near stair centroid, switching to move forward.")
    #             action[0] = 1 # 强制向前移动
    #         return action


    #     # 阶段2: 视角调整 (如果需要)
    #     # 仅在下楼梯且俯仰角过高时调整
    #     if self._climb_stair_flag == 2 and self._pitch_angle < -30: 
    #         self._pitch_angle += self._pitch_angle_offset
    #         print("Adjusting pitch angle for downstair (looking up a little).")
    #         return TorchActionIDs.LOOK_UP
        
    #     # 阶段3: 沿楼梯方向前进 (胡萝卜策略)
    #     else:
    #         distance = 0.8 # 目标点距离

    #         depth_map = self._observations_cache["nav_depth"].squeeze(0).cpu().numpy()
    #         if depth_map.size == 0: # 避免空深度图
    #             print("Warning: Depth map is empty. Cannot determine target point.")
    #             return TorchActionIDs.MOVE_FORWARD # 默认向前

    #         max_value = np.max(depth_map)
    #         max_indices = np.argwhere(depth_map == max_value)
            
    #         if max_indices.size == 0: # 避免没有最大值点
    #             print("Warning: No max depth value found. Cannot determine target point.")
    #             return TorchActionIDs.MOVE_FORWARD # 默认向前

    #         center_point = np.mean(max_indices, axis=0).astype(int)
    #         v, u = center_point[0], center_point[1]

    #         normalized_u = np.clip((u - self._cx) / self._cx, -1, 1)
    #         angle_offset = normalized_u * (self._camera_fov / 2)
    #         target_heading = heading - angle_offset # 尝试减去角度偏移
    #         target_heading = target_heading % (2 * np.pi)

    #         print("origin_target_heading:", target_heading)
    #         print("self.climb_stair_init_angle:", self.climb_stair_init_angle)

            
    #         delta_stair_angle = None
    #         if(self.climb_stair_init_angle is not None):
    #             delta_stair_angle = angle_difference(self.climb_stair_init_angle, heading)
    #             print("delta_stair_angle:", delta_stair_angle)
                
    #             # if(delta_stair_angle>np.pi/2):
    #             #     print("LEFTTTTTTTTTTTTT\n\n\n")
    #             #     return TorchActionIDs.TURN_LEFT
    #             # elif(delta_stair_angle<-np.pi/2):
    #             #     print("RIGHTTTTTTTTTTTTT\n\n\n")
    #             #     return TorchActionIDs.TURN_RIGHT

    #             if(delta_stair_angle>np.pi/2): # 左边加
    #                 target_heading = (target_heading + np.pi/2) % (2 * np.pi)
    #             elif(delta_stair_angle<-np.pi/2): # 右边减
    #                 target_heading = (target_heading - np.pi/2) % (2 * np.pi)
    #             print("final_target_heading:", target_heading)            

    #         x_target = robot_xy[0] + distance * np.cos(target_heading)
    #         y_target = robot_xy[1] + distance * np.sin(target_heading)
    #         current_target_point_xy = np.array([x_target, y_target])
    #         current_target_point_px = self._obstacle_map._xy_to_px(np.atleast_2d(current_target_point_xy))

    #         this_stair_end = self._obstacle_map._up_stair_end if self._climb_stair_flag == 1 else self._obstacle_map._down_stair_end        
            
    #         if(len(this_stair_end)==0):
    #             this_stair_end_px = np.array([])
    #         else:
    #             this_stair_end_px = self._obstacle_map._xy_to_px(np.atleast_2d(this_stair_end))[0]

    #         # 第一次 or 终点不存在 or 接近楼梯终点时 or 终点到不了时：重置胡萝卜目标点
    #         if (len(self._last_carrot_xy) == 0 or this_stair_end_px.size == 0 or 
    #             np.linalg.norm(this_stair_end_px - self._obstacle_map._xy_to_px(np.atleast_2d(robot_xy))[0]) <= 0.5 * self._obstacle_map.pixels_per_meter or 
    #             self._obstacle_map._disable_end):
                
    #             self._carrot_goal_xy = current_target_point_xy
    #             self._obstacle_map._carrot_goal_px = current_target_point_px
    #             self._last_carrot_xy = current_target_point_xy
    #             self._last_carrot_px = current_target_point_px
    #             # print("len(self._last_carrot_xy):", len(self._last_carrot_xy))
    #             # print("this_stair_end_px.size:", this_stair_end_px.size)
    #             # print("self._obstacle_map._disable_end:", self._obstacle_map._disable_end)
    #             # print("aaaaaaaaaaaaaa")
    #         else:
    #             # 比较当前胡萝卜目标点与上次胡萝卜目标点到楼梯终点的L1距离
    #             l1_distance_current = np.abs(this_stair_end_px[0] - current_target_point_px[0][0]) + np.abs(this_stair_end_px[1] - current_target_point_px[0][1])
    #             l1_distance_last = np.abs(this_stair_end_px[0] - self._last_carrot_px[0][0]) + np.abs(this_stair_end_px[1] - self._last_carrot_px[0][1])
                
    #             # print("bbbbbbbbbbbbbbbbb")

    #             if l1_distance_last > l1_distance_current: # 如果新的胡萝卜点离终点更近，则更新
    #                 self._carrot_goal_xy = current_target_point_xy
    #                 self._obstacle_map._carrot_goal_px = current_target_point_px
    #                 self._last_carrot_xy = current_target_point_xy
    #                 self._last_carrot_px = current_target_point_px
    #                 # print("ccccccccccccccccccc")
    #             # 否则，保持上一个胡萝卜目标点不变，即 self._carrot_goal_xy 和 _carrot_goal_px 已经是 _last_carrot 的值


    #         print("self._carrot_goal_xy:", self._carrot_goal_xy)
    #         print("robot_xy:", robot_xy)

    #         rho, theta = rho_theta(robot_xy, heading, self._carrot_goal_xy)
    #         rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)

    #         obs_pointnav = {
    #             "depth": image_resize(self._observations_cache["nav_depth"], self._depth_image_shape, channels_last=True, interpolation_mode="area"),
    #             "pointgoal_with_gps_compass": rho_theta_tensor,
    #         }
    #         self._policy_info["rho_theta"] = np.array([rho, theta])
    #         action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
            
    #         if action.item() == 0:
    #             print("Agent might stop, forcing move forward.")
    #             action[0] = 1 # 强制向前移动

    #         if (self.climb_stair_pos is not None) and (self.climb_stair_heading is not None) and (delta_stair_angle is not None) and (self.climb_stair_action[0][0].cpu().numpy()==1) and (self.climb_stair_pos[0]==robot_xy[0]) and (self.climb_stair_pos[1]==robot_xy[1]) and (self.climb_stair_heading==heading):
    #             if(delta_stair_angle>0):
    #                 print("ESCAPE LEFTTTTTTTTTTTTT\n\n\n")
    #                 self.climb_stair_escape_stuck = 1
    #                 return TorchActionIDs.TURN_LEFT
    #             elif(delta_stair_angle<0):
    #                 print("ESCAPE RIGHTTTTTTTTTTTTT\n\n\n")
    #                 self.climb_stair_escape_stuck = 1
    #                 return TorchActionIDs.TURN_RIGHT
    #         return action

    def _climb_stair(self, observations: "TensorDict", ori_masks: Tensor) -> Tensor:
        """
        处理爬楼梯（上楼或下楼）的逻辑，包括视角调整和目标点导航。
        """
        masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]

        # 根据爬楼梯标志设置目标楼梯前沿
        target_stair_frontier = self._obstacle_map._up_stair_frontiers if self._climb_stair_flag == 1 else self._obstacle_map._down_stair_frontiers
        
        if target_stair_frontier.size == 0:
            print(f"Error: Stair frontier for climb_stair_flag {self._climb_stair_flag} is empty. Returning to explore.")
            
            return TorchActionIDs.OTHER
        
        current_distance = np.linalg.norm(target_stair_frontier[0] - robot_xy)
        print(f"Climb Stair - Distance Change: {np.abs(self._last_frontier_distance - current_distance):.2f}m, Climb Stair Paused Step: {self._obstacle_map._climb_stair_paused_step}")
        
        # 检测是否卡顿在楼梯上
        if np.abs(self._last_frontier_distance - current_distance) > 0.2:
            self._obstacle_map._climb_stair_paused_step = 0
            self._last_frontier_distance = current_distance
        else:
            self._obstacle_map._climb_stair_paused_step += 1

        if self._obstacle_map._climb_stair_paused_step > 15:
            # 如果长时间卡顿，可能楼梯已经走完，或者遇到了障碍
            self._obstacle_map._disable_end = True # 标记楼梯终点可能不可达

        # 阶段1: 接近楼梯质心 (如果尚未到达)
        if not self._reach_stair_centroid:
            stair_centroid_point = target_stair_frontier[0] # 楼梯的质心点
            rho, theta = rho_theta(robot_xy, heading, stair_centroid_point)
            rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
            
            obs_pointnav = {
                "depth": image_resize(self._observations_cache["nav_depth"], self._depth_image_shape, channels_last=True, interpolation_mode="area"),
                "pointgoal_with_gps_compass": rho_theta_tensor,
            }
            self._policy_info["rho_theta"] = np.array([rho, theta])
            action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
            
            if action.item() == 0:
                self._reach_stair_centroid = True
                print("Agent is near stair centroid, switching to move forward.")
                action[0] = 1 # 强制向前移动
            return action


        # 阶段2: 视角调整 (如果需要)
        # 仅在下楼梯且俯仰角过高时调整
        if self._climb_stair_flag == 2 and self._pitch_angle < -30: 
            self._pitch_angle += self._pitch_angle_offset
            print("Adjusting pitch angle for downstair (looking up a little).")
            return TorchActionIDs.LOOK_UP
        
        # 阶段3: 沿楼梯方向前进 (胡萝卜策略)
        else:
            distance = 0.8 # 目标点距离

            depth_map = self._observations_cache["nav_depth"].squeeze(0).cpu().numpy()
            if depth_map.size == 0: # 避免空深度图
                print("Warning: Depth map is empty. Cannot determine target point.")
                return TorchActionIDs.MOVE_FORWARD # 默认向前

            max_value = np.max(depth_map)
            max_indices = np.argwhere(depth_map == max_value)
            
            if max_indices.size == 0: # 避免没有最大值点
                print("Warning: No max depth value found. Cannot determine target point.")
                return TorchActionIDs.MOVE_FORWARD # 默认向前

            center_point = np.mean(max_indices, axis=0).astype(int)
            v, u = center_point[0], center_point[1]

            normalized_u = np.clip((u - self._cx) / self._cx, -1, 1)
            angle_offset = normalized_u * (self._camera_fov / 2)
            target_heading = heading - angle_offset # 尝试减去角度偏移
            target_heading = target_heading % (2 * np.pi)           

            x_target = robot_xy[0] + distance * np.cos(target_heading)
            y_target = robot_xy[1] + distance * np.sin(target_heading)
            current_target_point_xy = np.array([x_target, y_target])
            current_target_point_px = self._obstacle_map._xy_to_px(np.atleast_2d(current_target_point_xy))

            this_stair_end = self._obstacle_map._up_stair_end if self._climb_stair_flag == 1 else self._obstacle_map._down_stair_end        
            
            if(len(this_stair_end)==0):
                this_stair_end_px = np.array([])
            else:
                this_stair_end_px = self._obstacle_map._xy_to_px(np.atleast_2d(this_stair_end))[0]

            # 第一次 or 终点不存在 or 接近楼梯终点时 or 终点到不了时：重置胡萝卜目标点
            if (len(self._last_carrot_xy) == 0 or this_stair_end_px.size == 0 or 
                np.linalg.norm(this_stair_end_px - self._obstacle_map._xy_to_px(np.atleast_2d(robot_xy))[0]) <= 0.5 * self._obstacle_map.pixels_per_meter or 
                self._obstacle_map._disable_end):
                
                self._carrot_goal_xy = current_target_point_xy
                self._obstacle_map._carrot_goal_px = current_target_point_px
                self._last_carrot_xy = current_target_point_xy
                self._last_carrot_px = current_target_point_px
            else:
                # 比较当前胡萝卜目标点与上次胡萝卜目标点到楼梯终点的L1距离
                l1_distance_current = np.abs(this_stair_end_px[0] - current_target_point_px[0][0]) + np.abs(this_stair_end_px[1] - current_target_point_px[0][1])
                l1_distance_last = np.abs(this_stair_end_px[0] - self._last_carrot_px[0][0]) + np.abs(this_stair_end_px[1] - self._last_carrot_px[0][1])

                if l1_distance_last > l1_distance_current: # 如果新的胡萝卜点离终点更近，则更新
                    self._carrot_goal_xy = current_target_point_xy
                    self._obstacle_map._carrot_goal_px = current_target_point_px
                    self._last_carrot_xy = current_target_point_xy
                    self._last_carrot_px = current_target_point_px
                # 否则，保持上一个胡萝卜目标点不变，即 self._carrot_goal_xy 和 _carrot_goal_px 已经是 _last_carrot 的值

            rho, theta = rho_theta(robot_xy, heading, self._carrot_goal_xy)
            rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)

            obs_pointnav = {
                "depth": image_resize(self._observations_cache["nav_depth"], self._depth_image_shape, channels_last=True, interpolation_mode="area"),
                "pointgoal_with_gps_compass": rho_theta_tensor,
            }
            self._policy_info["rho_theta"] = np.array([rho, theta])
            action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
            
            if action.item() == 0:
                print("Agent might stop, forcing move forward.")
                action[0] = 1 # 强制向前移动
            return action

    def _update_stair_state(self):
        """更新楼梯相关的内部状态。"""
        self._obstacle_map._climb_stair_paused_step = 0
        self._climb_stair_over = True
        self._climb_stair_flag = 0
        self._reach_stair = False
        self._reach_stair_centroid = False
        self._stair_dilate_flag = False

    def _look_for_downstair(self, observations: Union[Dict[str, Tensor], "TensorDict"], masks: Tensor) -> Tensor: # 针对疑似的“下”楼梯，需要进行的底层action
        # 如果已经有centroid就不用了
        if self._pitch_angle >= 0:
            self._pitch_angle -= self._pitch_angle_offset
            pointnav_action = TorchActionIDs.LOOK_DOWN
        else:
            robot_xy = self._observations_cache["robot_xy"]
            robot_xy_2d = np.atleast_2d(robot_xy) 
            dis_to_potential_stair = np.linalg.norm(self._obstacle_map._potential_stair_centroid - robot_xy_2d)
            if dis_to_potential_stair > 0.2:
                pointnav_action = self._pointnav(self._obstacle_map._potential_stair_centroid[0], stop=False) # 探索的时候可以远一点停？
                
                if pointnav_action.item() == 0:
                    print("Might false recognize down stairs, change to other mode.")
                    self._obstacle_map._disabled_frontiers.add(tuple(self._obstacle_map._potential_stair_centroid[0]))
                    print(f"Frontier {self._obstacle_map._potential_stair_centroid[0]} is disabled due to no movement.")
                    # 需验证，一般来说，如果真有向下的楼梯，并不会执行到这里
                    self._obstacle_map._disabled_stair_map[self._obstacle_map._down_stair_map == 1] = 1
                    self._obstacle_map._down_stair_map.fill(0)
                    self._obstacle_map._has_down_stair = False
                    self._pitch_angle += self._pitch_angle_offset
                    self._obstacle_map._look_for_downstair_flag = False
                    pointnav_action = TorchActionIDs.LOOK_UP

            else:
                print("Might false recognize down stairs, change to other mode.")
                self._obstacle_map._disabled_frontiers.add(tuple(self._obstacle_map._potential_stair_centroid[0]))
                print(f"Frontier {self._obstacle_map._potential_stair_centroid[0]} is disabled due to no movement.")
                # 需验证，一般来说，如果真有向下的楼梯，并不会执行到这里
                self._obstacle_map._disabled_stair_map[self._obstacle_map._down_stair_map == 1] = 1
                self._obstacle_map._down_stair_map.fill(0)
                self._obstacle_map._has_down_stair = False
                self._pitch_angle += self._pitch_angle_offset
                self._obstacle_map._look_for_downstair_flag = False
                pointnav_action = TorchActionIDs.LOOK_UP

        return pointnav_action

    def check_stairs_in_upper_50_percent(self):
        """
        检查在图像的上方30%区域是否有STAIR_CLASS_ID的标记
        参数：
        - mask: 布尔值数组，表示各像素是否属于STARR_CLASS_ID
        
        返回：
        - 如果上方30%区域有True，则返回True，否则返回False
        """
        if(1 not in self.current_step_stairs_res):
            return False

        for temp_stair_mask in self.current_step_stairs_res[1]:
            # 获取图像的高度
            height = temp_stair_mask.shape[0]
            
            # 计算上方50%的区域的高度范围
            upper_50_height = int(height * 0.5)
            
            # 获取上方50%的区域的掩码
            upper_50_mask = temp_stair_mask[:upper_50_height, :]
            
            print(f"Stair upper 50% points: {np.sum(upper_50_mask)}")
            # 检查该区域内是否有True
            if np.sum(upper_50_mask) > 50:  # 如果上方50%区域内有True
                return True
        return False


    def _disable_stair_and_reset_state(self, disabled_frontier: np.ndarray, is_reverse: bool = False):
        """
        辅助函数：禁用楼梯并重置相关状态。
        将重复的楼梯禁用和状态重置逻辑封装起来。
        """
        # 确保禁用前沿是可哈希的
        if disabled_frontier.size > 0:
            self._obstacle_map._disabled_frontiers.add(tuple(disabled_frontier))
            print(f"Frontier {disabled_frontier} is disabled due to no movement or reaching start.")
        
        # 重置卡顿相关计数
        self._get_close_to_stair_step = 0
        self._frontier_stick_step = 0
        self._obstacle_map._climb_stair_paused_step = 0
        self._last_carrot_xy = np.array([]) # 清空胡萝卜点
        self._last_carrot_px = np.array([])
        
        # 重置楼梯状态标志
        self._reach_stair = False
        self._reach_stair_centroid = False
        self._stair_dilate_flag = False
        self._climb_stair_over = True # 标记楼梯操作已结束
        self._climb_stair_flag = 0 # 重置爬楼梯标志
        self._obstacle_map._disable_end = False # 重置楼梯终点禁用标记

        # 根据上楼/下楼情况更新地图和楼层信息
        if self._climb_stair_flag == 1: # 上楼被禁用
            self._obstacle_map._disabled_stair_map[self._obstacle_map._up_stair_map == 1] = 1
            self._obstacle_map._up_stair_map.fill(0)
            self._obstacle_map._up_stair_frontiers = np.array([]) # 清空前沿
            self._obstacle_map._has_up_stair = False
            self._obstacle_map._look_for_downstair_flag = False
            # 误判上楼或无法上楼，删除多余的地图层
            if not is_reverse: # 如果不是反向返回，即是正常上楼失败
                if self._cur_floor_index + 1 < len(self._object_map_list): # 避免索引越界
                    del self._object_map_list[self._cur_floor_index + 1]
                    del self._value_map_list[self._cur_floor_index + 1]
                    del self._obstacle_map_list[self._cur_floor_index + 1]

                    del self._all_floor_frontier_list[self._cur_floor_index + 1]
                    del self._all_floor_false_frontier_list[self._cur_floor_index + 1]
                    del self._all_floor_false_stair_frontier_list[self._cur_floor_index + 1]
                    del self._all_floor_frontier_navigate_cnt_dict_list[self._cur_floor_index + 1]

                    self.floor_num -= 1 # 楼层数减1

        elif self._climb_stair_flag == 2: # 下楼被禁用
            self._obstacle_map._disabled_stair_map[self._obstacle_map._down_stair_map == 1] = 1
            self._obstacle_map._down_stair_map.fill(0)
            self._obstacle_map._down_stair_frontiers = np.array([]) # 清空前沿
            self._obstacle_map._has_down_stair = False
            self._obstacle_map._look_for_downstair_flag = False
            # 误判下楼或无法下楼，删除多余的地图层并调整当前楼层索引
            if not is_reverse: # 如果不是反向返回，即是正常下楼失败
                if self._cur_floor_index - 1 >= 0: # 避免索引越界
                    del self._object_map_list[self._cur_floor_index - 1]
                    del self._value_map_list[self._cur_floor_index - 1]
                    del self._obstacle_map_list[self._cur_floor_index - 1]
                    
                    del self._all_floor_frontier_list[self._cur_floor_index - 1]
                    del self._all_floor_false_frontier_list[self._cur_floor_index - 1]
                    del self._all_floor_false_stair_frontier_list[self._cur_floor_index - 1]
                    del self._all_floor_frontier_navigate_cnt_dict_list[self._cur_floor_index - 1]
                    
                    self.floor_num -= 1 # 楼层数减1
                    self._cur_floor_index -= 1 # 如果下楼是误判，当前层需要往下减一层

    def _get_close_to_stair(self, observations: "TensorDict", ori_masks: Tensor) -> Tensor:
        """
        处理导航到楼梯前沿的逻辑，包括卡顿检测和楼梯禁用。
        """
        # 参数校验和初始化
        if self._climb_stair_flag not in [1, 2]:
            print(f"Warning: _climb_stair_flag is not 1 or 2. Skipping _get_close_to_stair.")
            return TorchActionIDs.OTHER # 兜底，避免非预期状态

        masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
        robot_xy = self._observations_cache["robot_xy"]

        # 根据爬楼梯标志设置目标楼梯前沿
        target_stair_frontier = self._obstacle_map._up_stair_frontiers if self._climb_stair_flag == 1 else self._obstacle_map._down_stair_frontiers
        
        # 确保目标楼梯前沿存在，否则返回探索行为
        if target_stair_frontier.size == 0:
            print(f"Error: Stair frontier for climb_stair_flag {self._climb_stair_flag} is empty. Returning to explore.")
            return TorchActionIDs.OTHER
        
        # 目标楼梯点统一取第一个，如果逻辑允许有多个，需要更复杂的选择策略
        target_stair_point = target_stair_frontier[0]

        # --- 楼梯前沿卡顿检测逻辑重构 ---
        if np.array_equal(self._last_stair_frontier, target_stair_point):
            current_distance = np.linalg.norm(target_stair_point - robot_xy)

            if self._frontier_stick_step == 0: # 按照正在卡计数
                self._last_frontier_distance = current_distance
                self._frontier_stick_step += 1
                self._get_close_to_stair_step += 1
            else:
                # 检查距离变化是否超过阈值（0.3米）
                if np.abs(self._last_frontier_distance - current_distance) > 0.3:
                    self._frontier_stick_step = 0
                    self._last_frontier_distance = current_distance
                else:
                    self._frontier_stick_step += 1
                    self._get_close_to_stair_step += 1 # 记录累计卡住的次数

                    # 达到卡顿阈值，禁用楼梯前沿
                    if self._frontier_stick_step >= 30 or self._get_close_to_stair_step >= 60:
                        self._disable_stair_and_reset_state(target_stair_point) 
                        return TorchActionIDs.OTHER # 禁用后切换到探索 # 暂未看

        else:
            # 如果选中了不同的前沿，重置卡顿计数
            self._frontier_stick_step = 0
            self._last_frontier_distance = 0
            self._get_close_to_stair_step = 0

        # --- 楼梯前沿卡顿检测逻辑重构 ---
        
        # 更新 LLM Planner 的 last_frontier，使其知道当前正在导航到楼梯
        self._last_stair_frontier = target_stair_point

        # --- 使用点导航模型算动作 ---
        heading = self._observations_cache["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, target_stair_point)
        rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)

        obs_pointnav = {
            "depth": image_resize(
                self._observations_cache["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        self._policy_info["rho_theta"] = np.array([rho, theta])
        action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
        # --- 使用点导航模型算动作 ---

        # 如果点导航模型输出停止动作 (0)，则禁用楼梯并切换到探索
        if action.item() == 0:
            print(f"Pointnav policy stopped. Disabling stair frontier {target_stair_point}.")
            self._disable_stair_and_reset_state(target_stair_point)
            return TorchActionIDs.OTHER # 切换到探索

        return action


    def get_model_input_data(self, floor_index, robot_xy, robot_yaw, last_semantic_frontier_map, last_value_map, last_target_map, observations):
        # =====> 准备数据 <=====
        # frontier_map
        if(len(self._all_floor_frontier_list[floor_index])>0):
            now_frontier_map = np.zeros((1000, 1000))
            xy_points = self._all_floor_frontier_list[floor_index][:, :2]
            pixel_points = self._obstacle_map_list[floor_index]._xy_to_px(xy_points) # 点云的在map栅格上的像素坐标
            now_frontier_map[pixel_points[:, 1], pixel_points[:, 0]] = 1 # 计算语义地图（1为相应类别）
        else:
            now_frontier_map = np.zeros((1000, 1000))

        #  semantic_map
        now_semantic_map = self._object_map_list[floor_index].semantic_map

        # semantic_frontier_map
        semantic_frontier_map = np.zeros_like(now_semantic_map, dtype=np.float32)
        semantic_frontier_map[now_frontier_map == 1] = 0.5
        semantic_frontier_map[now_semantic_map == 1] = 1 # 1

        # value_map
        now_value_map = self._value_map_list[floor_index]._value_map_for_vis[:, :, 0]

        # robot_state_map
        robot_px = self._obstacle_map_list[floor_index]._xy_to_px(np.atleast_2d(robot_xy))
        robot_row, robot_col = int(robot_px[0, 1]), int(robot_px[0, 0])

        robot_state_map = np.zeros_like(now_semantic_map, dtype=np.float32)
        epsilon = 1e-5
        now_robot_yaw = normalize_angle(robot_yaw)
        if(now_robot_yaw<0.001):
            now_robot_yaw += epsilon
        robot_state_map[robot_row][robot_col] = now_robot_yaw # 3

        if(last_value_map is None):
            # =====> 第一帧启发式target_map <=====
            target_map = np.zeros((1000, 1000), dtype=np.float32)
            goal = self._get_target_object_location(self._object_map_list[floor_index], robot_xy) # 如果看到了goal，则返回其最佳位置；否则返回None
            
            if(goal is None):
                best_frontier = self._explore(self._all_floor_frontier_list[floor_index], robot_xy, self._value_map_list[floor_index], observations)
                target_sub_goal = np.array([[best_frontier[0], best_frontier[1]]])
            else:
                target_sub_goal = np.array([[goal[0], goal[1]]])
            xy_points = target_sub_goal
            pixel_points = self._obstacle_map_list[floor_index]._xy_to_px(xy_points) # 点云的在map栅格上的像素坐标
            
            if(goal is None):
                if (now_frontier_map[pixel_points[0, 1], pixel_points[0, 0]] == 1):
                    target_map[pixel_points[:, 1], pixel_points[:, 0]] = 1 # 计算语义地图（1为相应类别）
                else:
                    now_frontier_map_one_position = np.argwhere(now_frontier_map == 1)
                    real_row_col_array, real_min_index = find_closest_vector(np.array([pixel_points[0, 1], pixel_points[0, 0]]), now_frontier_map_one_position)
                    
                    pixel_points = np.array([[real_row_col_array[1], real_row_col_array[0]]])
                    target_map[pixel_points[:, 1], pixel_points[:, 0]] = 1 # 计算语义地图（1为相应类别）
            else:
                if (now_semantic_map[pixel_points[0, 1], pixel_points[0, 0]] == 1):
                    target_map[pixel_points[:, 1], pixel_points[:, 0]] = 1 # 计算语义地图（1为相应类别）
                else:
                    now_semantic_map_one_position = np.argwhere(now_semantic_map == 1)
                    real_row_col_array, real_min_index = find_closest_vector(np.array([pixel_points[0, 1], pixel_points[0, 0]]), now_semantic_map_one_position)
                    
                    pixel_points = np.array([[real_row_col_array[1], real_row_col_array[0]]])
                    target_map[pixel_points[:, 1], pixel_points[:, 0]] = 1 # 计算语义地图（1为相应类别）

            # assert (len(pixel_points)==1)
            # =====> 第一帧启发式target_map <=====
            last_semantic_frontier_map = copy.deepcopy(semantic_frontier_map)
            last_value_map = copy.deepcopy(now_value_map)
            last_target_map = copy.deepcopy(target_map)

        # resize_gai
        resize_last_semantic_frontier_map = compress_map_ultrafast_not_zero(last_semantic_frontier_map)
        resize_last_value_map = compress_map_ultrafast_not_zero(last_value_map)
        resize_last_target_map = compress_map_ultrafast_not_zero(last_target_map)
        resize_semantic_frontier_map = compress_map_ultrafast_not_zero(semantic_frontier_map)
        resize_now_value_map = compress_map_ultrafast_not_zero(now_value_map)
        resize_robot_state_map = compress_map_ultrafast_not_zero(robot_state_map)

        # resize_gai
        # 输入地图数据
        input_data = np.stack([
            resize_last_semantic_frontier_map, resize_last_value_map, resize_last_target_map,
            resize_semantic_frontier_map, resize_now_value_map, resize_robot_state_map
        ], axis=0)

        input_data = torch.FloatTensor(input_data)

        # resize_gai
        # 输入mask数据
        superimposed_mask = (resize_semantic_frontier_map > 0).astype(np.float32)
        curr_mask = torch.FloatTensor(superimposed_mask)
        return input_data, curr_mask, semantic_frontier_map, now_value_map

    def get_model_pre_res(self, input_data, curr_mask):
        VLFMTrainer.policy_model.eval()
        with torch.no_grad():
            preds = VLFMTrainer.policy_model(input_data)
            pred_map = preds[0, 0].detach().cpu().numpy()
            mask_map = curr_mask[0].cpu().numpy()

            masked_pred = pred_map.copy()
            valid_mask = (mask_map > 0)
            masked_pred[~valid_mask] = -float('inf')

            # 应用softmax得到概率分布
            masked_pred_flat = masked_pred.flatten()

            softmax_probs = F.softmax(torch.from_numpy(masked_pred_flat), dim=0).numpy()
            softmax_pred = softmax_probs.reshape(masked_pred.shape)
            # 选择最高概率位置
            pred_row, pred_col = np.unravel_index(softmax_pred.argmax(), softmax_pred.shape)
            pred_q_value = pred_map[int(pred_row)][int(pred_col)]
        return pred_row, pred_col, pred_q_value


    # def get_model_pre_res(self, input_data, curr_mask):
    #     VLFMTrainer.policy_model.eval()
    #     with torch.no_grad():
    #         # preds = VLFMTrainer.policy_model(input_data)

    #         q1_map, q2_map = VLFMTrainer.policy_model(input_data)
    #         preds = torch.min(q1_map, q2_map)

    #         pred_map = preds[0, 0].detach().cpu().numpy()
    #         mask_map = curr_mask[0].cpu().numpy()

    #         masked_pred = pred_map.copy()
    #         valid_mask = (mask_map > 0)
    #         masked_pred[~valid_mask] = -float('inf')

    #         # 选择最高概率位置
    #         pred_row, pred_col = np.unravel_index(masked_pred.argmax(), masked_pred.shape)

    #         # # 应用softmax得到概率分布
    #         # masked_pred_flat = masked_pred.flatten()

    #         # softmax_probs = F.softmax(torch.from_numpy(masked_pred_flat), dim=0).numpy()
    #         # softmax_pred = softmax_probs.reshape(masked_pred.shape)
    #         # # 选择最高概率位置
    #         # pred_row, pred_col = np.unravel_index(softmax_pred.argmax(), softmax_pred.shape)
    #         pred_q_value = pred_map[int(pred_row)][int(pred_col)]
    #     return pred_row, pred_col, pred_q_value

    def _navigate_stair_if_unexplored_floor(self, direction, real_stair_frontiers) -> Optional[Tensor]:
        """
        辅助函数：检查是否存在未探索的楼层，并通过楼梯导航。
        Args:
            direction (str): 'up' 或 'down'，表示向上或向下探索楼梯。
        Returns:
            Optional[Tensor]: 如果找到未探索的楼层并成功规划到楼梯，则返回点导航动作；否则返回 None。
        """
        
        # 根据方向选择对应的楼梯属性和爬楼标志值
        has_stair_attr = f"_has_{direction}_stair"
        climb_flag_value = 1 if direction == 'up' else 2

        if getattr(self._obstacle_map, has_stair_attr):
            self._climb_stair_over = False
            self._climb_stair_flag = climb_flag_value
            self._stair_frontier = real_stair_frontiers
            print(f"Navigating {direction}_stairs to unexplored floor.")
            # 假设 _stair_frontier[env] 包含了多个前沿，取第一个作为目标
            pointnav_action = self._pointnav(self._stair_frontier[0], stop=False)
            if(pointnav_action[0][0].cpu().numpy()==0):
                self._disable_stair_and_reset_state(self._stair_frontier[0])
                pointnav_action = TorchActionIDs.TURN_LEFT

                self._climb_stair_over = True
                self._climb_stair_flag = 0
            return pointnav_action
        return None # 未找到符合条件的楼梯或未探索楼层

            # # frontier_stair_revise
            # now_right_stair_frontier_ls = []
            # for temp_i in range(len(self._stair_frontier)):
            #     should_add = True
            #     for temp_j in range(len(self._all_floor_false_stair_frontier_list[self._cur_floor_index])):
            #         temp_dis = ((self._stair_frontier[temp_i][0]-self._all_floor_false_stair_frontier_list[self._cur_floor_index][temp_j][0])**2+(self._stair_frontier[temp_i][1]-self._all_floor_false_stair_frontier_list[self._cur_floor_index][temp_j][1])**2)**0.5
            #         if temp_dis<0.1:
            #             should_add = False
            #             break
            #     if(should_add==True):
            #         now_right_stair_frontier_ls.append(self._stair_frontier[temp_i])
            # self._stair_frontier = now_right_stair_frontier_ls
            # if(len(self._stair_frontier)==0):
            #     return None
            # return self._pointnav(self._stair_frontier[0], stop=False)
        

    def _direct_navigate_stair(self, direction, real_stair_frontiers) -> Optional[Tensor]:
        """
        辅助函数：检查是否存在未探索的楼层，并通过楼梯导航。
        Args:
            direction (str): 'up' 或 'down'，表示向上或向下探索楼梯。
        Returns:
            Optional[Tensor]: 如果找到未探索的楼层并成功规划到楼梯，则返回点导航动作；否则返回 None。
        """
        
        # 根据方向选择对应的楼梯属性和爬楼标志值
        has_stair_attr = f"_has_{direction}_stair"
        climb_flag_value = 1 if direction == 'up' else 2

        self._climb_stair_over = False
        self._climb_stair_flag = climb_flag_value

        if(len(real_stair_frontiers)>0):
            self._stair_frontier = real_stair_frontiers
        else:
            if(direction=="up"):
                self._stair_frontier = np.array([self._obstacle_map._up_stair_start])
            else:
                self._stair_frontier = np.array([self._obstacle_map._down_stair_start])
                    
        print(f"Navigating {direction}_stairs to unexplored floor.")
        # 假设 _stair_frontier[env] 包含了多个前沿，取第一个作为目标
        pointnav_action = self._pointnav(self._stair_frontier[0], stop=False)
        if(pointnav_action[0][0].cpu().numpy()==0):
            self._disable_stair_and_reset_state(self._stair_frontier[0])
            pointnav_action = TorchActionIDs.TURN_LEFT

            self._climb_stair_over = True
            self._climb_stair_flag = 0

        return pointnav_action

        

    def _handle_stairwell_reinitialization(self, masks: Tensor) -> Tensor:
        """
        辅助函数：处理楼梯间场景的地图重置和初始化。
        在没有 Frontier 且处于楼梯间状态时调用。
        """
        # 重置当前环境的对象地图和价值地图
        self._object_map.reset()
        self._value_map.reset()

        # 临时存储现有楼梯信息
        stair_data = {}
        for stair_type in ["up", "down"]:
            has_stair = getattr(self._obstacle_map, f"_has_{stair_type}_stair")
            if has_stair:
                stair_data[stair_type] = {
                    "map": getattr(self._obstacle_map, f"_{stair_type}_stair_map").copy(),
                    "start": getattr(self._obstacle_map, f"_{stair_type}_stair_start").copy(),
                    "end": getattr(self._obstacle_map, f"_{stair_type}_stair_end").copy(),
                    "frontiers": getattr(self._obstacle_map, f"_{stair_type}_stair_frontiers").copy(),
                    "explored": getattr(self._obstacle_map, f"_explored_{stair_type}_stair"),
                }
        
        # 重置障碍物地图
        self._obstacle_map.reset()


        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

        self._obstacle_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

        # 恢复之前存储的楼梯信息
        for stair_type, data in stair_data.items():
            setattr(self._obstacle_map, f"_has_{stair_type}_stair", True)
            setattr(self._obstacle_map, f"_{stair_type}_stair_map", data["map"])
            setattr(self._obstacle_map, f"_{stair_type}_stair_start", data["start"])
            setattr(self._obstacle_map, f"_{stair_type}_stair_end", data["end"])
            setattr(self._obstacle_map, f"_{stair_type}_stair_frontiers", data["frontiers"])
            setattr(self._obstacle_map, f"_explored_{stair_type}_stair", data["explored"])

        self._obstacle_map._reinitialize_flag = True # 标记已重初始化

        # 重置与导航状态相关的其他变量
        self._obstacle_map._tight_search_thresh = True
        self._climb_stair_over = True
        self._reach_stair = False
        self._reach_stair_centroid = False
        self._stair_dilate_flag = False
        self._pitch_angle = 0
        self._done_initializing = False
        self._initialize_step = 0

        self._done_initializing = True
        self._obstacle_map._done_initializing = True

        # 执行初始化动作
        return TorchActionIDs.TURN_LEFT


    def act(
        self,
        need_replan,
        observations: Dict, # 新的episode reset or 执行完上一个step
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        """
        Starts the episode by 'initializing' and allowing robot to get its bearings
        (e.g., spinning in place to get a good view of the scene).
        Then, explores the scene until it finds the target object.
        Once the target object is found, it navigates to the object.
        """
        self._object_masks = np.zeros((480, 640), dtype=np.uint8) # 累计记录当前帧rgb上检测得到的所有mask结果

        # get::当前帧的"frontier_sensor"信息
        self._pre_step(observations, masks) # 根据条件执行reset, 获得当前帧观测信息(存在_observations_cache中)，self._policy_info为空
        
        
        # self._observations_cache = {
        #     "robot_xy": robot_xy,
        #     "robot_heading": camera_yaw,
        #     "tf_camera_to_episodic": tf_camera_to_episodic,
        #     "rgb": rgb,
        #     "depth": depth,
        #     "min_depth": self._min_depth,
        #     "max_depth": self._max_depth,
        #     "fx": self._fx,
        #     "fy": self._fy,
        #     "camera_fov": self._camera_fov,
        #     "habitat_start_yaw": observations["heading"][0].item(),
        #      "camera_position"
            
        # }

        self.current_step_stairs_res = {}
        for temp_index in range(4):
            if(temp_index==0):
                self._update_stair_map(observations['semantic'], self._observations_cache["rgb"], self._observations_cache["depth"], self._observations_cache["tf_camera_to_episodic"], self._observations_cache["min_depth"], self._observations_cache["max_depth"], self._observations_cache["fx"], self._observations_cache["fy"]) # 完成当前rgb检测，累计记录当前帧rgb上检测得到的所有mask结果，将当前rgb观测+当前detect_mask对应的object_name对应的点云累计到self.clouds中+过滤掉clouds中“在当前fov及dis范围下， 不正常（随机数不为1）的点云”

            else:
                if abs(self._pitch_angle-0)<1e-5:
                    current_camera_yaw = self._observations_cache["robot_heading"]
                    current_camera_position = self._observations_cache["camera_position"]

                    if(temp_index==1): # 2
                        other_angle_camera_yaw = new_normalize_angle(current_camera_yaw+0.5*np.pi)
                    elif(temp_index==2): # 3
                        other_angle_camera_yaw = new_normalize_angle(current_camera_yaw+np.pi)
                    elif(temp_index==3): # 4
                        other_angle_camera_yaw = new_normalize_angle(current_camera_yaw-0.5*np.pi)

                    other_angle_tf_camera_to_episodic = xyz_yaw_to_tf_matrix(current_camera_position, other_angle_camera_yaw)
                    other_angle_depth = VLFMTrainer.depth_ls[temp_index-1].cpu().numpy()
                    other_angle_depth = filter_depth(other_angle_depth.reshape(other_angle_depth.shape[:2]), blur_type=None) # depth过滤

                    self._update_other_angle_stair_map(VLFMTrainer.semantic_ls[temp_index-1], temp_index+1, VLFMTrainer.rgb_ls[temp_index-1].cpu().numpy(), other_angle_depth, other_angle_tf_camera_to_episodic, self._observations_cache["min_depth"], self._observations_cache["max_depth"], self._observations_cache["fx"], self._observations_cache["fy"]) # 完成当前rgb检测，累计记录当前帧rgb上检测得到的所有mask结果，将当前rgb观测+当前detect_mask对应的object_name对应的点云累计到self.clouds中+过滤掉clouds中“在当前fov及dis范围下， 不正常（随机数不为1）的点云”

        self._update_obstacle_map(self._observations_cache, self.current_step_stairs_res, self._pitch_angle) # 直接完成所有全景的事情
        
        for temp_index in range(4):
            if(temp_index==0):
                detections = [
                    self._update_object_map(observations['semantic'], self._observations_cache["rgb"], self._observations_cache["depth"], self._observations_cache["tf_camera_to_episodic"], self._observations_cache["min_depth"], self._observations_cache["max_depth"], self._observations_cache["fx"], self._observations_cache["fy"]) # 完成当前rgb检测，累计记录当前帧rgb上检测得到的所有mask结果，将当前rgb观测+当前detect_mask对应的object_name对应的点云累计到self.clouds中+过滤掉clouds中“在当前fov及dis范围下， 不正常（随机数不为1）的点云”
                ] # get::当前帧的semantic_map # get::当前帧的机器人位置
            else:
                if abs(self._pitch_angle-0)<1e-5:
                    current_camera_yaw = self._observations_cache["robot_heading"]
                    current_camera_position = self._observations_cache["camera_position"]

                    if(temp_index==1): # 2
                        other_angle_camera_yaw = new_normalize_angle(current_camera_yaw+0.5*np.pi)
                    elif(temp_index==2): # 3
                        other_angle_camera_yaw = new_normalize_angle(current_camera_yaw+np.pi)
                    elif(temp_index==3): # 4
                        other_angle_camera_yaw = new_normalize_angle(current_camera_yaw-0.5*np.pi)

                    other_angle_tf_camera_to_episodic = xyz_yaw_to_tf_matrix(current_camera_position, other_angle_camera_yaw)
                    other_angle_depth = VLFMTrainer.depth_ls[temp_index-1].cpu().numpy()
                    other_angle_depth = filter_depth(other_angle_depth.reshape(other_angle_depth.shape[:2]), blur_type=None) # depth过滤

                    other_angle_detections = [
                        self._update_other_angle_object_map(VLFMTrainer.semantic_ls[temp_index-1], VLFMTrainer.rgb_ls[temp_index-1].cpu().numpy(), other_angle_depth, other_angle_tf_camera_to_episodic, self._observations_cache["min_depth"], self._observations_cache["max_depth"], self._observations_cache["fx"], self._observations_cache["fy"]) # 完成当前rgb检测，累计记录当前帧rgb上检测得到的所有mask结果，将当前rgb观测+当前detect_mask对应的object_name对应的点云累计到self.clouds中+过滤掉clouds中“在当前fov及dis范围下， 不正常（随机数不为1）的点云”
                    ] # get::当前帧的semantic_map # get::当前帧的机器人位置
        
        # print("self._object_map.clouds:", self._object_map.clouds)
        self._object_map.get_semantic_map(self._target_object) # 更新当前楼层的semantic_map
        self._update_value_map(self._observations_cache) # get::当前帧的value_map  # 直接完成所有全景的事情

        # 与操作
        explored_numeric = self._obstacle_map.explored_area.astype(np.float32)
        now_value_map = (explored_numeric * self._value_map._value_map.squeeze())
        now_value_map[self._obstacle_map._navigable_map==0] = -1
        now_value_map[self._obstacle_map._map==1] = -1
        now_value_map = now_value_map[:, :, np.newaxis]
        self._value_map._value_map_for_vis = now_value_map


        '''
        # 根据value_map的frontier归纳
        all_res_frontier = []
        for temp_frontier in self._all_floor_frontier_list[self._cur_floor_index]:
            x, y = temp_frontier[0], temp_frontier[1]
            px = int(x * self._value_map.pixels_per_meter) + self._value_map._episode_pixel_origin[1] # row
            py = int(-y * self._value_map.pixels_per_meter) + (1000-self._value_map._episode_pixel_origin[0]) # col
            point_px = (px, py)
            radius_px = int(0.5 * self._value_map.pixels_per_meter)
            temp_frontier_value = pixel_value_within_radius(self._value_map._value_map_for_vis[..., 0], point_px, radius_px)
            if(temp_frontier_value<=0):
                continue
            all_res_frontier.append(temp_frontier)
        self._all_floor_frontier_list[self._cur_floor_index] = np.array(all_res_frontier) # 当前帧最终真正需要的frontier
        '''


        # 界限以上是更新所有地图
        # =============================================
        # 界限以下是决策和实际执行底层action
        pointnav_action_env_list = []

        robot_xy = self._observations_cache["robot_xy"]
        robot_yaw = self._observations_cache["robot_heading"]

        print("robot_xy:", robot_xy)
        print("robot_yaw:", robot_yaw)

        # 寻找机器人所在层的goal
        robot_px = self._obstacle_map._xy_to_px(np.atleast_2d(robot_xy))
        x, y = int(robot_px[0, 0]), int(robot_px[0, 1]) # 确保索引是整数
        
        mode = "unknown" # 明确初始化 mode 变量
        is_in_current_floor_flag = False
        pointnav_action = None     

        # # 修改1
        num_frontiers = np.sum([len(temp_floor_frontiers) for temp_floor_frontiers in self._all_floor_frontier_list])
        num_semantic = np.sum([np.sum(temp_semantic_map.semantic_map) for temp_semantic_map in self._object_map_list])

        if (self._climb_stair_over) and (self._climb_stair_flag == 0) and ((num_frontiers+num_semantic)==0):
            self._detect_passive_stair_entry(robot_px, robot_xy) # 只有在所有action_space为空的情况下，才会进入此处

        if not self._climb_stair_over: # 楼梯状态判断与动作逻辑
            if self._reach_stair: # 到达了楼梯
                if self._pitch_angle == 0 and self._climb_stair_flag == 2:
                    self._pitch_angle -= self._pitch_angle_offset
                    mode = "look_down"
                    pointnav_action = TorchActionIDs.LOOK_DOWN
                elif self._climb_stair_flag == 2 and self._pitch_angle >= -30 and not self._reach_stair_centroid:
                    self._pitch_angle -= self._pitch_angle_offset
                    mode = "look_down_twice"
                    pointnav_action = TorchActionIDs.LOOK_DOWN
                else:
                    if self._obstacle_map._climb_stair_paused_step < 30:
                        mode = "climb_stair"
                        pointnav_action = self._climb_stair(observations, masks) 
                    else:
                        current_floor = self._cur_floor_index
                        # 楼层切换逻辑 - 上楼
                        if self._climb_stair_flag == 1: # 类似link的操作
                            next_floor = current_floor + 1
                            if next_floor < len(self._obstacle_map_list) and \
                            not self._obstacle_map_list[next_floor]._done_initializing:
                                
                                # 保存当前楼层的上楼梯信息到新楼层的下楼梯属性
                                self._obstacle_map_list[next_floor]._down_stair_map = \
                                    self._obstacle_map._up_stair_map.copy()
                                self._obstacle_map_list[next_floor]._down_stair_start = \
                                    self._obstacle_map._up_stair_start.copy()
                                self._obstacle_map_list[next_floor]._down_stair_end = \
                                    self._obstacle_map._up_stair_end.copy()
                                self._obstacle_map_list[next_floor]._down_stair_frontiers = \
                                    self._obstacle_map._up_stair_frontiers.copy()
                                self._obstacle_map_list[next_floor]._has_down_stair = True
                                
                                print(f"Saved upstairs info to floor {next_floor} as downstairs")
                        # 楼层切换逻辑 - 下楼
                        elif self._climb_stair_flag == 2: # 类似link的操作
                            prev_floor = current_floor - 1
                            if prev_floor >= 0 and \
                            not self._obstacle_map_list[prev_floor]._done_initializing:
                                
                                # 保存当前楼层的下楼梯信息到新楼层的上楼梯属性
                                self._obstacle_map_list[prev_floor]._up_stair_map = \
                                    self._obstacle_map._down_stair_map.copy()
                                self._obstacle_map_list[prev_floor]._up_stair_start = \
                                    self._obstacle_map._down_stair_start.copy()
                                self._obstacle_map_list[prev_floor]._up_stair_end = \
                                    self._obstacle_map._down_stair_end.copy()
                                self._obstacle_map_list[prev_floor]._up_stair_frontiers = \
                                    self._obstacle_map._down_stair_frontiers.copy()
                                self._obstacle_map_list[prev_floor]._has_up_stair = True
                                
                                print(f"Saved downstairs info to floor {prev_floor} as upstairs")

                        # 继续原有的初始化逻辑（卡住次数超过30， 则认为上下楼梯过程已经结束）
                        mode = "climb_stair_initialize"
                        if self._pitch_angle > 0:
                            self._pitch_angle -= self._pitch_angle_offset
                            pointnav_action = TorchActionIDs.LOOK_DOWN
                        elif self._pitch_angle < 0:
                            self._pitch_angle += self._pitch_angle_offset
                            pointnav_action = TorchActionIDs.LOOK_UP
                        else:
                            self._obstacle_map._done_initializing = False # for initial floor and new floor
                            self._initialize_step = 0
                            self._initialize()
                            pointnav_action = TorchActionIDs.OTHER # 初始化已完成，可以直接开始探索
                        self._update_stair_state() # 重置楼梯相关状态--> 跳出“if not self._climb_stair_over”

            else: # 未达到楼梯，但在楼梯附近
                # 检查是否不知不觉到了楼梯
                # 状态是爬楼梯结束 and 当前位置处在“下”楼梯上 and “下”楼梯的frontier不为空 and 下面一层的上楼过程还没完成（当前还没下去过）
                if self._climb_stair_over and self._obstacle_map._down_stair_map[y,x] == 1 and len(self._obstacle_map._down_stair_frontiers) > 0 and not self._obstacle_map_list[self._cur_floor_index - 1]._explored_up_stair:
                    self._reach_stair = True
                    self._get_close_to_stair_step = 0
                    self._climb_stair_over = False
                    self._climb_stair_flag = 2
                    self._obstacle_map._down_stair_start = robot_xy.copy()
                    mode = "down_stair_detected"
                elif self._climb_stair_over and self._obstacle_map._up_stair_map[y,x] == 1 and len(self._obstacle_map._up_stair_frontiers) > 0 and not self._obstacle_map_list[self._cur_floor_index + 1]._explored_down_stair:
                    self._reach_stair = True
                    self._get_close_to_stair_step = 0
                    self._climb_stair_over = False
                    self._climb_stair_flag = 1
                    self._obstacle_map._up_stair_start = robot_xy.copy()
                    mode = "up_stair_detected"
                
                # 待定的look_for
                if self._obstacle_map._look_for_downstair_flag: # 若有疑似的downstairs，则_look_for_downstair_flag为True；若确实有较大面积的downstairs or 不存在downstairs，则_look_for_downstair_flag为False
                    mode = "look_for_downstair"
                    pointnav_action = self._look_for_downstair(observations, masks) # 针对疑似的“下”楼梯，需要进行的底层action
                elif self._climb_stair_flag == 1 and self._pitch_angle == 0 and np.sum(self._obstacle_map._up_stair_map) > 0:
                    min_dis_to_upstair = np.min(np.abs(np.argwhere(self._obstacle_map._up_stair_map) - robot_px[0]).sum(axis=1))
                    print(f"min_dis_to_upstair: {min_dis_to_upstair}")
                    if min_dis_to_upstair <= 2.0 * self._obstacle_map.pixels_per_meter and self.check_stairs_in_upper_50_percent():
                        # 若机器人距离“上楼梯”的区域很近 and RGB上方50%具有楼梯mask --> 执行抬头操作
                        self._pitch_angle += self._pitch_angle_offset
                        mode = "look_up"
                        pointnav_action = TorchActionIDs.LOOK_UP
                    else:
                        mode = "get_close_to_stair"
                        pointnav_action = self._get_close_to_stair(observations, masks)
                
                elif self._climb_stair_flag == 2 and self._pitch_angle == 0 and np.sum(self._obstacle_map._down_stair_map) > 0:
                    min_dis_to_downstair = np.min(np.abs(np.argwhere(self._obstacle_map._down_stair_map) - robot_px[0]).sum(axis=1))
                    print(f"min_dis_to_downstair: {min_dis_to_downstair}")
                    if min_dis_to_downstair <= 2.0 * self._obstacle_map.pixels_per_meter:
                        # 若机器人距离“上楼梯”的区域很近 --> 执行低头操作
                        self._pitch_angle -= self._pitch_angle_offset
                        mode = "look_down"
                        pointnav_action = TorchActionIDs.LOOK_DOWN
                    else:
                        mode = "get_close_to_stair"
                        pointnav_action = self._get_close_to_stair(observations, masks)
                else:
                    mode = "get_close_to_stair"
                    pointnav_action = self._get_close_to_stair(observations, masks)

        if(mode == "get_close_to_stair"):
            self.current_close_to_stair_steps += 1
            if(self.current_close_to_stair_steps>=5): # 已经执行了5步
                self.current_close_to_stair_steps = 0
                pointnav_action = TorchActionIDs.OTHER
                self._climb_stair_over = True
        else:
            self.current_close_to_stair_steps = 0

        # =======================================
        # # 非楼梯爬行状态下的通用导航和探索逻辑
        if(pointnav_action is None) or (pointnav_action==TorchActionIDs.OTHER):
            if not self._done_initializing:
                self._initialize()
            

             # 不随便低头
            '''
            # 待定的look_for
            elif self._pitch_angle < 0 and not self._obstacle_map._look_for_downstair_flag:
                mode = "look_up_back"
                self._pitch_angle += self._pitch_angle_offset
                pointnav_action = TorchActionIDs.LOOK_UP
            # 待定的look_for
            elif self._obstacle_map._look_for_downstair_flag:
                mode = "look_for_downstair"
                pointnav_action = self._look_for_downstair(observations, masks)
            '''


            if self._pitch_angle > 0:
                mode = "look_down_back"
                self._pitch_angle -= self._pitch_angle_offset
                pointnav_action = TorchActionIDs.LOOK_DOWN
           
            # 不随便低头
            elif self._pitch_angle < 0:
                mode = "look_up_back"
                self._pitch_angle += self._pitch_angle_offset
                pointnav_action = TorchActionIDs.LOOK_UP

            else:
                num_frontiers = np.sum([len(temp_floor_frontiers) for temp_floor_frontiers in self._all_floor_frontier_list])
                num_semantic = np.sum([np.sum(temp_semantic_map.semantic_map) for temp_semantic_map in self._object_map_list])

                if (num_frontiers+num_semantic)>0:
                    # 1. 当前楼层有action_space
                    if (len(self._all_floor_frontier_list[self._cur_floor_index])+np.sum(self._object_map.semantic_map))>0:
                        # 1.1 首先用网络决策当前楼层
                        input_data, curr_mask, curr_floor_semantic_frontier_map, curr_floor_now_value_map = self.get_model_input_data(self._cur_floor_index, self._observations_cache["robot_xy"], self._observations_cache["robot_heading"], self.last_semantic_frontier_map, self.last_value_map, self.last_target_map, observations)
                        input_data = input_data.unsqueeze(0).to("cuda")  # [1, seq_len]
                        curr_mask = curr_mask.unsqueeze(0).to("cuda")  # [1, seq_len]

                        if(self.current_floor_steps%5)==0: # 这里需要协同考虑：楼梯frontiet + 其他楼层frontier
                            pred_row, pred_col, pred_q_value = self.get_model_pre_res(input_data, curr_mask)

                            '''
                            # ==========> 判断其他楼层 <==========
                            import math
                            up_floor_pred_xy, up_floor_pred_q_value = None, -math.inf

                            real_up_stair_frontiers = np.array([f for f in self._obstacle_map._up_stair_frontiers if tuple(f) not in self._obstacle_map._disabled_frontiers])

                            if(self._obstacle_map._explored_up_stair) and (np.sum(self._object_map.semantic_map)==0): # 如果上面一层已经去过
                                up_floor_num_frontiers = len(self._all_floor_frontier_list[self._cur_floor_index+1])
                                up_floor_num_semantic = np.sum(self._object_map_list[self._cur_floor_index+1].semantic_map)
                                if(up_floor_num_frontiers+up_floor_num_semantic)>0: # 上面一层的action_space不为空
                                    # 1.2 用网络决策上面的楼层
                                    up_floor_input_data, up_floor_curr_mask, _, _ = self.get_model_input_data(self._cur_floor_index+1, self._obstacle_map._up_stair_end, np.pi, None, None, None, observations)
                                    up_floor_input_data = up_floor_input_data.unsqueeze(0).to("cuda")  # [1, seq_len]
                                    up_floor_curr_mask = up_floor_curr_mask.unsqueeze(0).to("cuda")  # [1, seq_len]

                                    up_floor_pred_row, up_floor_pred_col, up_floor_pred_q_value = self.get_model_pre_res(up_floor_input_data, up_floor_curr_mask)
                                    # up_floor_pred_q_value -= (np.linalg.norm(self._obstacle_map._up_stair_start - self._observations_cache["robot_xy"])/0.25*0.01+15*0.01)

                                    delta_steps_current_to_stairs = math.ceil(np.linalg.norm(self._obstacle_map._up_stair_start - self._observations_cache["robot_xy"])/0.25)+15
                                    delta_steps_gamma_factor = 1
                                    while True:
                                        if(delta_steps_current_to_stairs>=5):
                                            up_floor_pred_q_value -= delta_steps_gamma_factor*0.01*5
                                            delta_steps_current_to_stairs -= 5
                                            delta_steps_gamma_factor *= 0.99
                                        elif(delta_steps_current_to_stairs<5) and (delta_steps_current_to_stairs>0):
                                            up_floor_pred_q_value -= delta_steps_gamma_factor*0.01*delta_steps_current_to_stairs
                                            delta_steps_current_to_stairs -= 5
                                            delta_steps_gamma_factor *= 0.99 
                                        else:
                                            break
                                    up_floor_pred_xy = self._obstacle_map._up_stair_start

                            elif(len(real_up_stair_frontiers)>0) and (np.sum(self._object_map.semantic_map)==0):# 如果上面一层没有去过，但是有上去的frontier
                                # up_floor_pred_q_value = self.PRIOR_VALUE - (np.linalg.norm(real_up_stair_frontiers[0] - self._observations_cache["robot_xy"])/0.25*0.01+15*0.01)
                                up_floor_pred_q_value = self.PRIOR_VALUE
                                
                                delta_steps_current_to_stairs = math.ceil(np.linalg.norm(real_up_stair_frontiers[0] - self._observations_cache["robot_xy"])/0.25)+15
                                delta_steps_gamma_factor = 1
                                while True:
                                    if(delta_steps_current_to_stairs>=5):
                                        up_floor_pred_q_value -= delta_steps_gamma_factor*0.01*5
                                        delta_steps_current_to_stairs -= 5
                                        delta_steps_gamma_factor *= 0.99
                                    elif(delta_steps_current_to_stairs<5) and (delta_steps_current_to_stairs>0):
                                        up_floor_pred_q_value -= delta_steps_gamma_factor*0.01*delta_steps_current_to_stairs
                                        delta_steps_current_to_stairs -= 5
                                        delta_steps_gamma_factor *= 0.99 
                                    else:
                                        break
                                
                                up_floor_pred_xy = real_up_stair_frontiers[0]


                            down_floor_pred_xy, down_floor_pred_q_value = None, -math.inf

                            real_down_stair_frontiers = np.array([f for f in self._obstacle_map._down_stair_frontiers if tuple(f) not in self._obstacle_map._disabled_frontiers])

                            if(self._obstacle_map._explored_down_stair) and (np.sum(self._object_map.semantic_map)==0): # 如果下面一层已经去过
                                down_floor_num_frontiers = len(self._all_floor_frontier_list[self._cur_floor_index-1])
                                down_floor_num_semantic = np.sum(self._object_map_list[self._cur_floor_index-1].semantic_map)
                                if(down_floor_num_frontiers+down_floor_num_semantic)>0: # 下面一层的action_space不为空
                                    # 1.3 用网络决策下面的楼层
                                    down_floor_input_data, down_floor_curr_mask, _, _ = self.get_model_input_data(self._cur_floor_index-1, self._obstacle_map._down_stair_end, np.pi, None, None, None, observations)
                                    down_floor_input_data = down_floor_input_data.unsqueeze(0).to("cuda")  # [1, seq_len]
                                    down_floor_curr_mask = down_floor_curr_mask.unsqueeze(0).to("cuda")  # [1, seq_len]

                                    down_floor_pred_row, down_floor_pred_col, down_floor_pred_q_value = self.get_model_pre_res(down_floor_input_data, down_floor_curr_mask)
                                    
                                    # down_floor_pred_q_value -= (np.linalg.norm(self._obstacle_map._down_stair_start - self._observations_cache["robot_xy"])/0.25*0.01+15*0.01)
                                    
                                    delta_steps_current_to_stairs = math.ceil(np.linalg.norm(self._obstacle_map._down_stair_start - self._observations_cache["robot_xy"])/0.25)+15
                                    delta_steps_gamma_factor = 1
                                    while True:
                                        if(delta_steps_current_to_stairs>=5):
                                            down_floor_pred_q_value -= delta_steps_gamma_factor*0.01*5
                                            delta_steps_current_to_stairs -= 5
                                            delta_steps_gamma_factor *= 0.99
                                        elif(delta_steps_current_to_stairs<5) and (delta_steps_current_to_stairs>0):
                                            down_floor_pred_q_value -= delta_steps_gamma_factor*0.01*delta_steps_current_to_stairs
                                            delta_steps_current_to_stairs -= 5
                                            delta_steps_gamma_factor *= 0.99 
                                        else:
                                            break
                                    
                                    down_floor_pred_xy = self._obstacle_map._down_stair_start

                            elif(len(real_down_stair_frontiers)>0) and (np.sum(self._object_map.semantic_map)==0):# 如果下面一层没有去过，但是有下去的frontier
                                # down_floor_pred_q_value = self.PRIOR_VALUE - (np.linalg.norm(real_down_stair_frontiers[0] - self._observations_cache["robot_xy"])/0.25*0.01+15*0.01)
                                down_floor_pred_q_value = self.PRIOR_VALUE
                                
                                delta_steps_current_to_stairs = math.ceil(np.linalg.norm(real_down_stair_frontiers[0] - self._observations_cache["robot_xy"])/0.25)+15
                                delta_steps_gamma_factor = 1
                                while True:
                                    if(delta_steps_current_to_stairs>=5):
                                        down_floor_pred_q_value -= delta_steps_gamma_factor*0.01*5
                                        delta_steps_current_to_stairs -= 5
                                        delta_steps_gamma_factor *= 0.99
                                    elif(delta_steps_current_to_stairs<5) and (delta_steps_current_to_stairs>0):
                                        down_floor_pred_q_value -= delta_steps_gamma_factor*0.01*delta_steps_current_to_stairs
                                        delta_steps_current_to_stairs -= 5
                                        delta_steps_gamma_factor *= 0.99 
                                    else:
                                        break
                                
                                down_floor_pred_xy = real_down_stair_frontiers[0]

                            all_q_value_ls = [pred_q_value, up_floor_pred_q_value, down_floor_pred_q_value]
                            

                            # max_q_value_index = all_q_value_ls.index(max(all_q_value_ls)) # 修改2
                            max_q_value_index = 0 # 修改2
                            if(max_q_value_index==1): # 表示往上
                                mode = "GO UP FLOOR"
                                # pointnav_action = self._navigate_stair_if_unexplored_floor(direction="up")
                                pointnav_action = self._direct_navigate_stair(direction="up", real_stair_frontiers=real_up_stair_frontiers)

                                self.current_floor_steps = 0
                                self.last_target_map = None
                                self.last_semantic_frontier_map = None
                                self.last_value_map = None
                                self._last_value = float("-inf")
                                self._last_frontier = np.zeros(2)
                                
                                action_numpy = pointnav_action.detach().cpu().numpy()[0]
                                if len(action_numpy) == 1:
                                    action_numpy = action_numpy[0]
                                print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")
                                self._policy_info.update(self._get_policy_info(detections[0])) # 每act一次，有一个新的_policy_info
                                self._num_steps += 1
                                self._obstacle_map._floor_num_steps += 1

                                self._observations_cache = {} # 每act一次，有一个新的_observations_cache
                                self._did_reset = False # 每act结束一次后，_did_reset都变为False
                                print("==========> <==========")
                                os.environ["DEBUG_INFO"] = mode
                                self.last_mode = mode
                                return pointnav_action, rnn_hidden_states, None

                            elif(max_q_value_index==2): # 表示往下
                                mode = "GO DOWN FLOOR"
                                # pointnav_action = self._navigate_stair_if_unexplored_floor(direction="down")
                                pointnav_action = self._direct_navigate_stair(direction="down", real_stair_frontiers=real_down_stair_frontiers)
                                
                                self.current_floor_steps = 0
                                self.last_target_map = None
                                self.last_semantic_frontier_map = None
                                self.last_value_map = None
                                self._last_value = float("-inf")
                                self._last_frontier = np.zeros(2)
                                
                                action_numpy = pointnav_action.detach().cpu().numpy()[0]
                                if len(action_numpy) == 1:
                                    action_numpy = action_numpy[0]
                                print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")
                                self._policy_info.update(self._get_policy_info(detections[0])) # 每act一次，有一个新的_policy_info
                                self._num_steps += 1
                                self._obstacle_map._floor_num_steps += 1

                                self._observations_cache = {} # 每act一次，有一个新的_observations_cache
                                self._did_reset = False # 每act结束一次后，_did_reset都变为False
                                print("==========> <==========")
                                os.environ["DEBUG_INFO"] = mode
                                self.last_mode = mode
                                return pointnav_action, rnn_hidden_states, None
                            '''
                        else:
                            # resize_gai
                            resize_last_target_map = input_data[0][2].cpu().numpy()
                            rows, cols = np.where(resize_last_target_map == 1)
                            pred_row, pred_col = rows[0], cols[0]

                        is_in_current_floor_flag = True
                        # resize_gai
                        pred_row, pred_col = int(pred_row), int(pred_col)
                        resize_pred_row, resize_pred_col = int(np.clip(pred_row*2, 0, 1000-1)), int(np.clip(pred_col*2, 0, 1000-1))
                        sub_goal_array = self._obstacle_map._px_to_xy(np.atleast_2d(np.array([resize_pred_col, resize_pred_row])))[0]

                        resize_semantic_frontier_map = input_data[0][3].cpu().numpy()
                        # resize_gai
                        if(resize_semantic_frontier_map[pred_row, pred_col]==1):
                            mode = "navigate"
                            navigate_target_cloud = self._object_map.get_target_cloud(self._target_object)
                            if(len(navigate_target_cloud)>0):
                                real_sub_goal_array, real_min_index = find_closest_vector(sub_goal_array, navigate_target_cloud[:, :2])
                                real_mean_center = find_dense_subcluster_dbscan(points=navigate_target_cloud[:, :2], real_sub_goal_array=real_sub_goal_array, eps=1.0, min_samples=1)
                                real_sub_goal_array, real_min_index = find_closest_vector(real_mean_center, navigate_target_cloud[:, :2])
                                
                                pointnav_action = self._pointnav(real_sub_goal_array, stop=True)
                            else:
                                navigate_target_cloud = None # is_need_closer
                                real_sub_goal_array = sub_goal_array
                                pointnav_action = self._pointnav(real_sub_goal_array, stop=True)

                        else:
                            explore_target_cloud = self._all_floor_frontier_list[self._cur_floor_index]
                            if(len(explore_target_cloud)>0):    
                                mode = "explore"
                                real_sub_goal_array, real_min_index = find_closest_vector(sub_goal_array, explore_target_cloud[:, :2])
                                pointnav_action = self._pointnav(real_sub_goal_array, stop=False)
                            
                                # frontier_multi_navigate_revise
                                if (self.current_floor_steps%5)==0:
                                    min_dis = 100000
                                    min_key = None
                                    for temp_key in self._all_floor_frontier_navigate_cnt_dict_list[self._cur_floor_index]:
                                        temp_dis = ((temp_key[0]-real_sub_goal_array[0])**2+(temp_key[1]-real_sub_goal_array[1])**2)**0.5
                                        if(temp_dis<self.same_frontier_dis_thre):
                                            if (temp_dis<min_dis):
                                                min_dis = temp_dis
                                                min_key = temp_key
                                    if(min_key is not None):
                                        self._all_floor_frontier_navigate_cnt_dict_list[self._cur_floor_index][min_key] += 1
                                    else:
                                        self._all_floor_frontier_navigate_cnt_dict_list[self._cur_floor_index][(real_sub_goal_array[0], real_sub_goal_array[1])] = 1
                                
                                # frontier_revise
                                if(pointnav_action[0][0].cpu().numpy()==0):
                                    self._all_floor_false_frontier_list[self._cur_floor_index].append(real_sub_goal_array)
                                    pointnav_action = TorchActionIDs.TURN_LEFT
                                
                            else:
                                mode = "navigate"
                                pointnav_action = self._pointnav(sub_goal_array, stop=True)

                        # resize_gai
                        # =====> latst_target_map <=====
                        last_target_map = np.zeros((1000, 1000), dtype=np.float32)
                        last_target_map[resize_pred_row, resize_pred_col] = 1 # 计算语义地图（1为相应类别）
                        # =====> latst_target_map <=====

                        self.last_semantic_frontier_map = copy.deepcopy(curr_floor_semantic_frontier_map)
                        self.last_value_map = copy.deepcopy(curr_floor_now_value_map)
                        self.last_target_map = copy.deepcopy(last_target_map)
                    else: # 2.1 当前楼层没有action -> 尝试选择其他楼层
                        # ==========> 判断其他楼层 <==========
                        import math
                        up_floor_pred_xy, up_floor_pred_q_value = None, -math.inf

                        real_up_stair_frontiers = np.array([f for f in self._obstacle_map._up_stair_frontiers if tuple(f) not in self._obstacle_map._disabled_frontiers])

                        if(self._obstacle_map._explored_up_stair): # 如果上面一层已经去过
                            up_floor_num_frontiers = len(self._all_floor_frontier_list[self._cur_floor_index+1])
                            up_floor_num_semantic = np.sum(self._object_map_list[self._cur_floor_index+1].semantic_map)
                            if(up_floor_num_frontiers+up_floor_num_semantic)>0: # 上面一层的action_space不为空
                                # 1.2 用网络决策上面的楼层
                                print("1.5--up_floor --------------------")
                                
                                '''
                                up_floor_input_data, up_floor_curr_mask, _, _ = self.get_model_input_data(self._cur_floor_index+1, self._obstacle_map._up_stair_end, np.pi, None, None, None, observations)
                                up_floor_input_data = up_floor_input_data.unsqueeze(0).to("cuda")  # [1, seq_len]
                                up_floor_curr_mask = up_floor_curr_mask.unsqueeze(0).to("cuda")  # [1, seq_len]

                                up_floor_pred_row, up_floor_pred_col, up_floor_pred_q_value = self.get_model_pre_res(up_floor_input_data, up_floor_curr_mask)
                                # up_floor_pred_q_value -= (np.linalg.norm(self._obstacle_map._up_stair_start - self._observations_cache["robot_xy"])/0.25*0.01+15*0.01)
                                
                                delta_steps_current_to_stairs = math.ceil(np.linalg.norm(self._obstacle_map._up_stair_start - self._observations_cache["robot_xy"])/0.25)+15
                                delta_steps_gamma_factor = 1
                                while True:
                                    if(delta_steps_current_to_stairs>=5):
                                        up_floor_pred_q_value -= delta_steps_gamma_factor*0.01*5
                                        delta_steps_current_to_stairs -= 5
                                        delta_steps_gamma_factor *= 0.99
                                    elif(delta_steps_current_to_stairs<5) and (delta_steps_current_to_stairs>0):
                                        up_floor_pred_q_value -= delta_steps_gamma_factor*0.01*delta_steps_current_to_stairs
                                        delta_steps_current_to_stairs -= 5
                                        delta_steps_gamma_factor *= 0.99 
                                    else:
                                        break
                                '''
                                up_floor_pred_q_value = -np.linalg.norm(self._obstacle_map._up_stair_start - self._observations_cache["robot_xy"])
                                up_floor_pred_xy = self._obstacle_map._up_stair_start

                        elif(len(real_up_stair_frontiers)>0):# 如果上面一层没有去过，但是有上去的frontier
                            # up_floor_pred_q_value = self.PRIOR_VALUE - (np.linalg.norm(real_up_stair_frontiers[0] - self._observations_cache["robot_xy"])/0.25*0.01+15*0.01)
                            
                            '''
                            up_floor_pred_q_value = self.PRIOR_VALUE
                            delta_steps_current_to_stairs = math.ceil(np.linalg.norm(real_up_stair_frontiers[0] - self._observations_cache["robot_xy"])/0.25)+15
                            delta_steps_gamma_factor = 1
                            while True:
                                if(delta_steps_current_to_stairs>=5):
                                    up_floor_pred_q_value -= delta_steps_gamma_factor*0.01*5
                                    delta_steps_current_to_stairs -= 5
                                    delta_steps_gamma_factor *= 0.99
                                elif(delta_steps_current_to_stairs<5) and (delta_steps_current_to_stairs>0):
                                    up_floor_pred_q_value -= delta_steps_gamma_factor*0.01*delta_steps_current_to_stairs
                                    delta_steps_current_to_stairs -= 5
                                    delta_steps_gamma_factor *= 0.99 
                                else:
                                    break
                            '''
                            up_floor_pred_q_value = -np.linalg.norm(real_up_stair_frontiers[0] - self._observations_cache["robot_xy"])         
                            up_floor_pred_xy = real_up_stair_frontiers[0]


                        down_floor_pred_xy, down_floor_pred_q_value = None, -math.inf
                        real_down_stair_frontiers = np.array([f for f in self._obstacle_map._down_stair_frontiers if tuple(f) not in self._obstacle_map._disabled_frontiers])

                        if(self._obstacle_map._explored_down_stair): # 如果下面一层已经去过
                            down_floor_num_frontiers = len(self._all_floor_frontier_list[self._cur_floor_index-1])
                            down_floor_num_semantic = np.sum(self._object_map_list[self._cur_floor_index-1].semantic_map)
                            if(down_floor_num_frontiers+down_floor_num_semantic)>0: # 下面一层的action_space不为空
                                # 1.3 用网络决策下面的楼层
                                print("1.5--down_floor --------------------")

                                '''
                                down_floor_input_data, down_floor_curr_mask, _, _ = self.get_model_input_data(self._cur_floor_index-1, self._obstacle_map._down_stair_end, np.pi, None, None, None, observations)
                                down_floor_input_data = down_floor_input_data.unsqueeze(0).to("cuda")  # [1, seq_len]
                                down_floor_curr_mask = down_floor_curr_mask.unsqueeze(0).to("cuda")  # [1, seq_len]

                                down_floor_pred_row, down_floor_pred_col, down_floor_pred_q_value = self.get_model_pre_res(down_floor_input_data, down_floor_curr_mask)
                                # down_floor_pred_q_value -= (np.linalg.norm(self._obstacle_map._down_stair_start - self._observations_cache["robot_xy"])/0.25*0.01+15*0.01)
                                
                                
                                delta_steps_current_to_stairs = math.ceil(np.linalg.norm(self._obstacle_map._down_stair_start - self._observations_cache["robot_xy"])/0.25)+15
                                delta_steps_gamma_factor = 1
                                while True:
                                    if(delta_steps_current_to_stairs>=5):
                                        down_floor_pred_q_value -= delta_steps_gamma_factor*0.01*5
                                        delta_steps_current_to_stairs -= 5
                                        delta_steps_gamma_factor *= 0.99
                                    elif(delta_steps_current_to_stairs<5) and (delta_steps_current_to_stairs>0):
                                        down_floor_pred_q_value -= delta_steps_gamma_factor*0.01*delta_steps_current_to_stairs
                                        delta_steps_current_to_stairs -= 5
                                        delta_steps_gamma_factor *= 0.99 
                                    else:
                                        break
                                '''
                                down_floor_pred_q_value = -np.linalg.norm(self._obstacle_map._down_stair_start - self._observations_cache["robot_xy"])
                                down_floor_pred_xy = self._obstacle_map._down_stair_start

                        elif(len(real_down_stair_frontiers)>0):# 如果下面一层没有去过，但是有下去的frontier
                            # down_floor_pred_q_value = self.PRIOR_VALUE - (np.linalg.norm(real_down_stair_frontiers[0] - self._observations_cache["robot_xy"])/0.25*0.01+15*0.01)
                            
                            '''
                            down_floor_pred_q_value = self.PRIOR_VALUE
                            
                            delta_steps_current_to_stairs = math.ceil(np.linalg.norm(real_down_stair_frontiers[0] - self._observations_cache["robot_xy"])/0.25)+15
                            delta_steps_gamma_factor = 1
                            while True:
                                if(delta_steps_current_to_stairs>=5):
                                    down_floor_pred_q_value -= delta_steps_gamma_factor*0.01*5
                                    delta_steps_current_to_stairs -= 5
                                    delta_steps_gamma_factor *= 0.99
                                elif(delta_steps_current_to_stairs<5) and (delta_steps_current_to_stairs>0):
                                    down_floor_pred_q_value -= delta_steps_gamma_factor*0.01*delta_steps_current_to_stairs
                                    delta_steps_current_to_stairs -= 5
                                    delta_steps_gamma_factor *= 0.99 
                                else:
                                    break
                            '''
                            down_floor_pred_q_value = -np.linalg.norm(real_down_stair_frontiers[0] - self._observations_cache["robot_xy"])
                            down_floor_pred_xy = real_down_stair_frontiers[0]

                        all_q_value_ls = [up_floor_pred_q_value, down_floor_pred_q_value]
                        max_q_value_index = all_q_value_ls.index(max(all_q_value_ls))
                        if(max_q_value_index==0) and (max(all_q_value_ls)!=-math.inf): # 表示往上
                            mode = "NO ACTION, GO UP FLOOR"
                            # pointnav_action = self._navigate_stair_if_unexplored_floor(direction="up")
                            pointnav_action = self._direct_navigate_stair(direction="up", real_stair_frontiers=real_up_stair_frontiers)
                            
                            self.current_floor_steps = 0
                            self.last_target_map = None
                            self.last_semantic_frontier_map = None
                            self.last_value_map = None
                            self._last_value = float("-inf")
                            self._last_frontier = np.zeros(2)
                            
                            action_numpy = pointnav_action.detach().cpu().numpy()[0]
                            if len(action_numpy) == 1:
                                action_numpy = action_numpy[0]
                            print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")
                            self._policy_info.update(self._get_policy_info(detections[0])) # 每act一次，有一个新的_policy_info
                            self._num_steps += 1
                            self._obstacle_map._floor_num_steps += 1

                            self._observations_cache = {} # 每act一次，有一个新的_observations_cache
                            self._did_reset = False # 每act结束一次后，_did_reset都变为False
                            print("==========> <==========")
                            os.environ["DEBUG_INFO"] = mode
                            self.last_mode = mode
                            return pointnav_action, rnn_hidden_states, None

                        elif(max_q_value_index==1) and (max(all_q_value_ls)!=-math.inf): # 表示往下
                            mode = "NO ACTION, GO DOWN FLOOR"
                            # pointnav_action = self._navigate_stair_if_unexplored_floor(direction="down")
                            pointnav_action = self._direct_navigate_stair(direction="down", real_stair_frontiers=real_down_stair_frontiers)
                            
                            self.current_floor_steps = 0
                            self.last_target_map = None
                            self.last_semantic_frontier_map = None
                            self.last_value_map = None
                            self._last_value = float("-inf")
                            self._last_frontier = np.zeros(2)
                            
                            action_numpy = pointnav_action.detach().cpu().numpy()[0]
                            if len(action_numpy) == 1:
                                action_numpy = action_numpy[0]
                            print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")
                            self._policy_info.update(self._get_policy_info(detections[0])) # 每act一次，有一个新的_policy_info
                            self._num_steps += 1
                            self._obstacle_map._floor_num_steps += 1

                            self._observations_cache = {} # 每act一次，有一个新的_observations_cache
                            self._did_reset = False # 每act结束一次后，_did_reset都变为False
                            print("==========> <==========")
                            os.environ["DEBUG_INFO"] = mode
                            self.last_mode = mode
                            return pointnav_action, rnn_hidden_states, None
                        
                        # 剩余的情况在外面直接STOP
                
                else: # 整个环境都没有action
                    # 如果还没有重新初始化过，并且在该楼层步数很短，并且有没探索过的高层或者低层并且没有找到对应的楼梯，如果在楼梯间且未探索完（防止卡在楼梯间），尝试重置并初始化.
                    if not self._obstacle_map._reinitialize_flag and \
                    self._obstacle_map._floor_num_steps < 50 and \
                    ((self._obstacle_map._explored_up_stair == False and self._obstacle_map._up_stair_frontiers.size == 0) or \
                        (self._obstacle_map._explored_down_stair == False and self._obstacle_map._down_stair_frontiers.size == 0)):
                        mode = "STUCK_IN_STAIRS"
                        pointnav_action = self._handle_stairwell_reinitialization(masks)
                        
                        self.current_floor_steps = 0
                        self.last_target_map = None
                        self.last_semantic_frontier_map = None
                        self.last_value_map = None
                        self._last_value = float("-inf")
                        self._last_frontier = np.zeros(2)
                        
                        action_numpy = pointnav_action.detach().cpu().numpy()[0]
                        if len(action_numpy) == 1:
                            action_numpy = action_numpy[0]
                        print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")
                        self._policy_info.update(self._get_policy_info(detections[0])) # 每act一次，有一个新的_policy_info
                        self._num_steps += 1
                        self._obstacle_map._floor_num_steps += 1

                        self._observations_cache = {} # 每act一次，有一个新的_observations_cache
                        self._did_reset = False # 每act结束一次后，_did_reset都变为False
                        print("==========> <==========")
                        os.environ["DEBUG_INFO"] = mode
                        self.last_mode = mode
                        return pointnav_action, rnn_hidden_states, None

                    # 标记当前楼层已探索结束
                    self._obstacle_map._this_floor_explored = True

                    # 尝试导航到未探索的楼层 (优先上楼，其次下楼)
                    pointnav_action = None
                    real_up_stair_frontiers = np.array([f for f in self._obstacle_map._up_stair_frontiers if tuple(f) not in self._obstacle_map._disabled_frontiers])
                    real_down_stair_frontiers = np.array([f for f in self._obstacle_map._down_stair_frontiers if tuple(f) not in self._obstacle_map._disabled_frontiers])
                    
                    # ACTION为空的STAIR_FRONTIER_revise
                    if (len(real_up_stair_frontiers)+len(real_down_stair_frontiers)==0):
                        real_up_stair_frontiers = np.array([np.array([f[0], f[1]+0.05]) for f in self._obstacle_map._up_stair_frontiers])
                        real_down_stair_frontiers = np.array([np.array([f[0], f[1]+0.05]) for f in self._obstacle_map._down_stair_frontiers])

                    if (not self._obstacle_map._explored_up_stair) and (len(real_up_stair_frontiers>0)):
                        pointnav_action = self._navigate_stair_if_unexplored_floor('up', real_up_stair_frontiers)
                    
                    if (pointnav_action is None) and (not self._obstacle_map._explored_down_stair) and (len(real_down_stair_frontiers)>0):
                        pointnav_action = self._navigate_stair_if_unexplored_floor('down', real_down_stair_frontiers)

                    if pointnav_action is not None:
                        # # frontier_stair_revise
                        # if(pointnav_action[0][0].cpu().numpy()==0):
                        #     self._all_floor_false_stair_frontier_list[self._cur_floor_index].append(self._stair_frontier[0])
                        #     pointnav_action = TorchActionIDs.TURN_LEFT

                        #     self._climb_stair_over = True
                        #     self._climb_stair_flag = 0
                        mode = "PRIOR_UP_OR_DOWN"
                        self.current_floor_steps = 0
                        self.last_target_map = None
                        self.last_semantic_frontier_map = None
                        self.last_value_map = None
                        self._last_value = float("-inf")
                        self._last_frontier = np.zeros(2)

                        action_numpy = pointnav_action.detach().cpu().numpy()[0]
                        if len(action_numpy) == 1:
                            action_numpy = action_numpy[0]
                        print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")
                        self._policy_info.update(self._get_policy_info(detections[0])) # 每act一次，有一个新的_policy_info
                        self._num_steps += 1
                        self._obstacle_map._floor_num_steps += 1

                        self._observations_cache = {} # 每act一次，有一个新的_observations_cache
                        self._did_reset = False # 每act结束一次后，_did_reset都变为False
                        print("==========> <==========")
                        os.environ["DEBUG_INFO"] = mode
                        self.last_mode = mode
                        return pointnav_action, rnn_hidden_states, None

                    else:
                        print(f"In all floors, no unexplored stairs or frontiers found, stopping.")
                        mode = "NO ACTION"
                        pointnav_action = TorchActionIDs.STOP

                        self.current_floor_steps = 0
                        self.last_target_map = None
                        self.last_semantic_frontier_map = None
                        self.last_value_map = None
                        self._last_value = float("-inf")
                        self._last_frontier = np.zeros(2)

                        action_numpy = pointnav_action.detach().cpu().numpy()[0]
                        if len(action_numpy) == 1:
                            action_numpy = action_numpy[0]
                        print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")
                        self._policy_info.update(self._get_policy_info(detections[0])) # 每act一次，有一个新的_policy_info
                        self._num_steps += 1
                        self._obstacle_map._floor_num_steps += 1

                        self._observations_cache = {} # 每act一次，有一个新的_observations_cache
                        self._did_reset = False # 每act结束一次后，_did_reset都变为False
                        print("==========> <==========")
                        os.environ["DEBUG_INFO"] = mode
                        self.last_mode = mode
                        return pointnav_action, rnn_hidden_states, None

        if(is_in_current_floor_flag==False):
            self.current_floor_steps = 0
            
            self.last_target_map = None
            self.last_semantic_frontier_map = None
            self.last_value_map = None

            self._last_value = float("-inf")
            self._last_frontier = np.zeros(2)
        else:
            # 最后的最后
            self.current_floor_steps += 1
        
        if(pointnav_action is None):
            mode = "ACCIDENT STOP"
            pointnav_action = TorchActionIDs.STOP

        elif (pointnav_action==TorchActionIDs.OTHER):
            mode = "OTHER STOP"
            pointnav_action = TorchActionIDs.STOP



        '''
        # 尝试：长度为30的action_buffer判断是否卡住
        # Check history and potentially override the action
        if len(self.history_action) >= 30: # Use >= to correctly handle the window size
            # Store the first action before popping to see what it was, if needed for debugging
            # old_action_in_history = self.history_action[env][0] 
            self.history_action.pop(0) # Remove the oldest action to maintain window size

            if all(temp_action in [TorchActionIDs.TURN_LEFT, TorchActionIDs.TURN_RIGHT] for temp_action in self.history_action):
                mode = "REVISE_ACTION"
                pointnav_action = TorchActionIDs.MOVE_FORWARD # Force forward
                print("Continuous turns to force forward.")
            elif all(temp_action in [TorchActionIDs.MOVE_FORWARD] for temp_action in self.history_action):
                mode = "REVISE_ACTION"
                pointnav_action = TorchActionIDs.TURN_LEFT # Force turn right
                print("Continuous forward to force turn right.")
        self.history_action.append(pointnav_action)
        '''

        action_numpy = pointnav_action.detach().cpu().numpy()[0]
        if len(action_numpy) == 1:
            action_numpy = action_numpy[0]
        print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")
        
        
        self._policy_info.update(self._get_policy_info(detections[0])) # 每act一次，有一个新的_policy_info
        self._num_steps += 1
        self._obstacle_map._floor_num_steps += 1

        self._observations_cache = {} # 每act一次，有一个新的_observations_cache
        self._did_reset = False # 每act结束一次后，_did_reset都变为False
        print("==========> <==========")
        os.environ["DEBUG_INFO"] = mode
        self.last_mode = mode
        return pointnav_action, rnn_hidden_states, None

    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None: # 根据条件执行reset, 获得当前帧观测信息(存在_observations_cache中)，self._policy_info为空
        assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        if not self._did_reset and masks[0] == 0: # reset之前，_did_reset为False
            self._reset()
            self._target_object = observations["objectgoal"]
        try:
            self._cache_observations(observations) # 根据当前观察，得到rgb，depth，frontiers，xy，yaw等信息    

        except IndexError as e:
            print(e)
            print("Reached edge of map, stopping.")
            raise StopIteration
        self._policy_info = {}

    '''
    def _initialize(self) -> Tensor:
        raise NotImplementedError
    '''

    def _initialize(self):
        """Turn left 30 degrees 12 times to get a 360 view at the beginning"""
        self._done_initializing = True
        self._obstacle_map._done_initializing = True
        self._obstacle_map._tight_search_thresh = False 


    '''
    def _explore(self, observations: "TensorDict") -> Tensor:
        raise NotImplementedError
    '''

    def _explore(self, floor_frontiers, robot_xy, floor_value_map, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        frontiers = floor_frontiers
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("No frontiers found during exploration, stopping.")
            return self._stop_action, None
        best_frontier, best_value = self._get_best_frontier(observations, robot_xy, floor_value_map, frontiers)
        print(f"Best value: {best_value*100:.2f}%")

        return best_frontier

    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        robot_xy,
        floor_value_map,
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Returns the best frontier and its value based on self._value_map.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from
                the environment.
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """

        # The points and values will be sorted in descending order
        sorted_pts, sorted_values = self._sort_frontiers_by_value(robot_xy, floor_value_map, observations, frontiers)
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])

        # If there is a last point pursued, then we consider sticking to pursuing it
        # if it is still in the list of frontiers and its current value is not much
        # worse than self._last_value.
        
        # 优先选择上一帧的frontier，若上一帧的frontier不在 or 上一帧的frontier周围没有frontier or 找到上一帧的frontier但没有很高于上一帧的分数，则选择分数最高的frontier 
        if not np.array_equal(self._last_frontier, np.zeros(2)): # 若没有上一个frontier，则没有匹配到上一个frontier
            curr_index = None

            # 若上一个frontier还在，则curr_index对应上一个fronier
            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_frontier):
                    # Last point is still in the list of frontiers
                    curr_index = idx
                    break

            # 若上一个frontier不在，则找在一定阈值范围内与上一个frontier最近的frontier
            if curr_index is None: # last_frontier已经被消除
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier, threshold=0.5)

                if closest_index != -1:
                    # There is a point close to the last point pursued
                    curr_index = closest_index

            if curr_index is not None:
                curr_value = sorted_values[curr_index]
                if curr_value + 0.01 > self._last_value:
                    # The last point pursued is still in the list of frontiers and its
                    # value is not much worse than self._last_value
                    print("Sticking to last point.")
                    best_frontier_idx = curr_index

        # If there is no last point pursued, then just take the best point, given that
        # it is not cyclic.
        # 若没有匹配到上一个frontier
        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                if cyclic:
                    print("Suppressed cyclic frontier.")
                    continue
                best_frontier_idx = idx
                break

        if best_frontier_idx is None:
            print("All frontiers are cyclic. Just choosing the closest one.")
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier

        return best_frontier, best_value





    def _update_value_map(self, observations_cache: List[Dict]) -> None:
        for temp_index in range(4):
            if(temp_index==0):
                cosines = [
                    [
                        self._itm.cosine(
                            observations_cache["rgb"],
                            p.replace("target_object", self._target_object.replace("|", "/")),
                        )
                        for p in self._text_prompt.split(PROMPT_SEPARATOR)
                    ]
                ]
                self._value_map.update_map(np.array(cosines[0]), observations_cache["depth"], observations_cache["tf_camera_to_episodic"], observations_cache["min_depth"], observations_cache["max_depth"], observations_cache["camera_fov"]) # 用blip分数更新
            
            else:
                if abs(self._pitch_angle-0)<1e-5:
                    cosines = [
                        [
                            self._itm.cosine(
                                VLFMTrainer.rgb_ls[temp_index-1].cpu().numpy(),
                                p.replace("target_object", self._target_object.replace("|", "/")),
                            )
                            for p in self._text_prompt.split(PROMPT_SEPARATOR)
                        ]
                    ]

                    current_camera_yaw = observations_cache["robot_heading"]
                    current_camera_position = observations_cache["camera_position"]

                    if(temp_index==1): # 2
                        other_angle_camera_yaw = new_normalize_angle(current_camera_yaw+0.5*np.pi)
                    elif(temp_index==2): # 3
                        other_angle_camera_yaw = new_normalize_angle(current_camera_yaw+np.pi)
                    elif(temp_index==3): # 4
                        other_angle_camera_yaw = new_normalize_angle(current_camera_yaw-0.5*np.pi)

                    other_angle_tf_camera_to_episodic = xyz_yaw_to_tf_matrix(current_camera_position, other_angle_camera_yaw)

                    other_angle_depth = VLFMTrainer.depth_ls[temp_index-1].cpu().numpy()
                    other_angle_depth = filter_depth(other_angle_depth.reshape(other_angle_depth.shape[:2]), blur_type=None) # depth过滤

                    self._value_map.update_map(np.array(cosines[0]), other_angle_depth, other_angle_tf_camera_to_episodic, observations_cache["min_depth"], observations_cache["max_depth"], observations_cache["camera_fov"]) # 用blip分数更新

        self._value_map.update_agent_traj(
            observations_cache["robot_xy"],
            observations_cache["robot_heading"],
        )

    def _sort_frontiers_by_value(
        self, robot_xy, floor_value_map, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = floor_value_map.sort_waypoints(robot_xy, frontiers, 0.5)
        return sorted_frontiers, sorted_values


    def _get_target_object_location(self, floor_object_map, position: np.ndarray) -> Union[None, np.ndarray]: # 如果看到了goal，则返回其最佳位置；否则返回None
        if floor_object_map.has_object(self._target_object):
            return floor_object_map.get_best_object(self._target_object, position) # 根据“连续多帧target点云的变化情况+机器人与target点云的距离”-->"最佳target点云"
        else:
            return None

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        # 获取目标点云信息
        if self._object_map.has_object(self._target_object):
            target_point_cloud = self._object_map.get_target_cloud(self._target_object) # 得到target_object对应的点云
        else:
            target_point_cloud = np.array([])
        
        
        policy_info = {
            "target_object": self._target_object.split("|")[0],
            "gps": str(self._observations_cache["robot_xy"] * np.array([1, -1])),
            "yaw": np.rad2deg(self._observations_cache["robot_heading"]),
            "target_detected": self._object_map.has_object(self._target_object),
            "target_point_cloud": target_point_cloud,
            "nav_goal": self._last_goal,
            "stop_called": self._called_stop, # 只有在导航到object_goal时，才可能调用stop==True
            # don't render these on egocentric images when making videos:
            "render_below_images": [
                "target_object",
            ],
            "num_steps": self._num_steps,
        }

        if not self._visualize:
            return policy_info

        # 处理注释深度图和 RGB 图
        annotated_depth = self._observations_cache["depth"] * 255
        annotated_depth = cv2.cvtColor(annotated_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        if self._object_masks.sum() > 0:
            # If self._object_masks isn't all zero, get the object segmentations and
            # draw them on the rgb and depth images
            contours, _ = cv2.findContours(self._object_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # contours, _ = cv2.findContours(self._object_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            annotated_rgb = self._observations_cache["rgb"]
            annotated_rgb = cv2.drawContours(annotated_rgb, contours, -1, (255, 0, 0), 2)
            annotated_depth = cv2.drawContours(annotated_depth, contours, -1, (255, 0, 0), 2)

            # annotated_rgb = cv2.drawContours(detections.annotated_frame, contours, -1, (255, 0, 0), 2)
            # annotated_depth = cv2.drawContours(annotated_depth, contours, -1, (255, 0, 0), 2)
        else:
            annotated_rgb = self._observations_cache["rgb"]
        policy_info["annotated_rgb"] = annotated_rgb
        policy_info["annotated_depth"] = annotated_depth

        if self._compute_frontiers:
            policy_info["obstacle_map"] = cv2.cvtColor(self._obstacle_map.visualize(), cv2.COLOR_BGR2RGB)

        if "DEBUG_INFO" in os.environ:
            policy_info["render_below_images"].append("debug")
            policy_info["debug"] = "debug: " + os.environ["DEBUG_INFO"]

        return policy_info

    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections: # 根据开集和闭集模型同时得到所有检测
        target_classes = self._target_object.split("|")
        has_coco = any(c in COCO_CLASSES for c in target_classes) and self._load_yolo
        has_non_coco = any(c not in COCO_CLASSES for c in target_classes)

        detections = (
            self._coco_object_detector.predict(img)
            if has_coco
            else self._object_detector.predict(img, caption=self._non_coco_caption)
        )
        detections.filter_by_class(target_classes) # 得到当前rgb图片的所有bbox，分数，对应的短语
        det_conf_threshold = self._coco_threshold if has_coco else self._non_coco_threshold
        detections.filter_by_conf(det_conf_threshold) # 根据分数阈值筛选

        if has_coco and has_non_coco and detections.num_detections == 0:
            # Retry with non-coco object detector
            detections = self._object_detector.predict(img, caption=self._non_coco_caption)
            detections.filter_by_class(target_classes)
            detections.filter_by_conf(self._non_coco_threshold)

        return detections

    def _pointnav(self, goal: np.ndarray, stop: bool = False) -> Tensor:
        """
        Calculates rho and theta from the robot's current position to the goal using the
        gps and heading sensors within the observations and the given goal, then uses
        it to determine the next action to take using the pre-trained pointnav policy.

        Args:
            goal (np.ndarray): The goal to navigate to as (x, y), where x and y are in
                meters.
            stop (bool): Whether to stop if we are close enough to the goal.

        """
        masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
        if not np.array_equal(goal, self._last_goal):
            if np.linalg.norm(goal - self._last_goal) > 0.1:
                self._pointnav_policy.reset() # WrappedPointNavResNetPolicy类型
                masks = torch.zeros_like(masks)
            self._last_goal = goal # self._last_goal用于底层导航
        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)
        rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
        obs_pointnav = {
            "depth": image_resize(
                self._observations_cache["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        self._policy_info["rho_theta"] = np.array([rho, theta])
        if rho < self._pointnav_stop_radius and stop: # _pointnav_stop_radius为0.9m
        # if (rho < 1.0) and (stop): # _pointnav_stop_radius为0.9m
            self._called_stop = True
            return self._stop_action
        action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
        return action

    # def _update_object_map(
    #     self,
    #     semantic_mask,
    #     rgb: np.ndarray,
    #     depth: np.ndarray,
    #     tf_camera_to_episodic: np.ndarray,
    #     min_depth: float,
    #     max_depth: float,
    #     fx: float,
    #     fy: float,
    # ) -> ObjectDetections: # 完成当前rgb检测，累计记录当前帧rgb上检测得到的所有mask结果，将当前rgb观测+当前detect_mask对应的object_name对应的点云累计到self.clouds中+过滤掉clouds中“在当前fov及dis范围下， 不正常（随机数不为1）的点云”
    #     """
    #     Updates the object map with the given rgb and depth images, and the given
    #     transformation matrix from the camera to the episodic coordinate frame.

    #     Args:
    #         rgb (np.ndarray): The rgb image to use for updating the object map. Used for
    #             object detection and Mobile SAM segmentation to extract better object
    #             point clouds.
    #         depth (np.ndarray): The depth image to use for updating the object map. It
    #             is normalized to the range [0, 1] and has a shape of (height, width).
    #         tf_camera_to_episodic (np.ndarray): The transformation matrix from the
    #             camera to the episodic coordinate frame.
    #         min_depth (float): The minimum depth value (in meters) of the depth image.
    #         max_depth (float): The maximum depth value (in meters) of the depth image.
    #         fx (float): The focal length of the camera in the x direction.
    #         fy (float): The focal length of the camera in the y direction.

    #     Returns:
    #         ObjectDetections: The object detections from the object detector.
    #     """
    #     detections = self._get_object_detections(rgb) # 根据开集和闭集模型同时得到所有检测
        
    #     height, width = rgb.shape[:2]
    #     # self._object_masks = np.zeros((height, width), dtype=np.uint8) # 累计记录当前帧rgb上检测得到的所有mask结果
        
    #     # if np.array_equal(depth, np.ones_like(depth)) and detections.num_detections > 0: # 暂时不存在该情况
    #     #     depth = self._infer_depth(rgb, min_depth, max_depth)
    #     #     obs = list(self._observations_cache["object_map_rgbd"][0])
    #     #     obs[1] = depth
    #     #     self._observations_cache["object_map_rgbd"][0] = tuple(obs)
        
        
    #     for idx in range(len(detections.logits)):
    #         bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
    #         object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist()) # 当前rgb中的每个检测mask

    #         # If we are using vqa, then use the BLIP2 model to visually confirm whether
    #         # the contours are actually correct.
            
    #         # # 不使用
    #         # if self._use_vqa:
    #         #     contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #         #     annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255, 0, 0), 2)
    #         #     question = f"Question: {self._vqa_prompt}"
    #         #     if not detections.phrases[idx].endswith("ing"):
    #         #         question += "a "
    #         #     question += detections.phrases[idx] + "? Answer:"
    #         #     answer = self._vqa.ask(annotated_rgb, question)
    #         #     if not answer.lower().startswith("yes"):
    #         #         continue

    #         self._object_masks[object_mask > 0] = 1
    #         self._object_map.update_map( # ObjectPointCloudMap类型
    #             self._target_object,
    #             depth,
    #             object_mask,
    #             tf_camera_to_episodic,
    #             min_depth,
    #             max_depth,
    #             fx,
    #             fy,
    #         ) # 将当前rgb观测+当前detect_mask对应的object_name对应的点云累计到self.clouds中

    #     cone_fov = get_fov(fx, depth.shape[1])
    #     self._object_map.update_explored(tf_camera_to_episodic, max_depth, cone_fov) # 过滤掉“在当前fov及dis范围下， 不正常（随机数不为1）的点云”

        
    #     return detections


    def _update_object_map(
        self,
        semantic_mask,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> ObjectDetections: # 完成当前rgb检测，累计记录当前帧rgb上检测得到的所有mask结果，将当前rgb观测+当前detect_mask对应的object_name对应的点云累计到self.clouds中+过滤掉clouds中“在当前fov及dis范围下， 不正常（随机数不为1）的点云”
        """
        Updates the object map with the given rgb and depth images, and the given
        transformation matrix from the camera to the episodic coordinate frame.

        Args:
            rgb (np.ndarray): The rgb image to use for updating the object map. Used for
                object detection and Mobile SAM segmentation to extract better object
                point clouds.
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).
            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.

        Returns:
            ObjectDetections: The object detections from the object detector.
        """
        semantic_mask = semantic_mask[0, :, :, 0]
        mask_res_ls = []
        for temp_id in VLFMTrainer.current_episode_object_id:
            temp_mask = (semantic_mask==temp_id).cpu().numpy().astype(np.uint8)
            if np.sum(temp_mask)>0:
                mask_res_ls.append(temp_mask)  

        height, width = rgb.shape[:2]  

        for idx in range(len(mask_res_ls)):
            object_mask = mask_res_ls[idx]
            self._object_masks[object_mask > 0] = 1
            self._object_map.update_map( # ObjectPointCloudMap类型
                self._target_object,
                depth,
                object_mask,
                tf_camera_to_episodic,
                min_depth,
                max_depth,
                fx,
                fy,
            ) # 将当前rgb观测+当前detect_mask对应的object_name对应的点云累计到self.clouds中

        cone_fov = get_fov(fx, depth.shape[1])
        self._object_map.update_explored(tf_camera_to_episodic, max_depth, cone_fov) # 过滤掉“在当前fov及dis范围下， 不正常（随机数不为1）的点云”
        return mask_res_ls


    def _update_stair_map(
        self,
        semantic_mask,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> ObjectDetections: # 完成当前rgb检测，累计记录当前帧rgb上检测得到的所有mask结果，将当前rgb观测+当前detect_mask对应的object_name对应的点云累计到self.clouds中+过滤掉clouds中“在当前fov及dis范围下， 不正常（随机数不为1）的点云”
        # ==============================================================
        # 检测楼梯
        # height, width = rgb.shape[:2]
        # self._object_masks = np.zeros((height, width), dtype=np.uint8) # 累计记录当前帧rgb上检测得到的所有mask结果

        semantic_mask = semantic_mask[0, :, :, 0]
        mask_res_ls = []
        for temp_id in VLFMTrainer.res_all_stairs_id:
            temp_mask = (semantic_mask==temp_id).cpu().numpy().astype(np.uint8)
            if np.sum(temp_mask)>0:
                mask_res_ls.append(temp_mask)  

        if(len(mask_res_ls)>0):
            self.current_step_stairs_res[1] = mask_res_ls

        for idx in range(len(mask_res_ls)): # 便于可视化stairs
            temp_stair_mask = mask_res_ls[idx]
            self._object_masks[temp_stair_mask > 0] = 1
        


    # def _update_other_angle_object_map(
    #     self,
    #     semantic_mask,
    #     direct_id,
    #     rgb: np.ndarray,
    #     depth: np.ndarray,
    #     tf_camera_to_episodic: np.ndarray,
    #     min_depth: float,
    #     max_depth: float,
    #     fx: float,
    #     fy: float,
    # ) -> ObjectDetections: # 完成当前rgb检测，累计记录当前帧rgb上检测得到的所有mask结果，将当前rgb观测+当前detect_mask对应的object_name对应的点云累计到self.clouds中+过滤掉clouds中“在当前fov及dis范围下， 不正常（随机数不为1）的点云”
    #     """
    #     Updates the object map with the given rgb and depth images, and the given
    #     transformation matrix from the camera to the episodic coordinate frame.

    #     Args:
    #         rgb (np.ndarray): The rgb image to use for updating the object map. Used for
    #             object detection and Mobile SAM segmentation to extract better object
    #             point clouds.
    #         depth (np.ndarray): The depth image to use for updating the object map. It
    #             is normalized to the range [0, 1] and has a shape of (height, width).
    #         tf_camera_to_episodic (np.ndarray): The transformation matrix from the
    #             camera to the episodic coordinate frame.
    #         min_depth (float): The minimum depth value (in meters) of the depth image.
    #         max_depth (float): The maximum depth value (in meters) of the depth image.
    #         fx (float): The focal length of the camera in the x direction.
    #         fy (float): The focal length of the camera in the y direction.

    #     Returns:
    #         ObjectDetections: The object detections from the object detector.
    #     """
    #     detections = self._get_object_detections(rgb) # 根据开集和闭集模型同时得到所有检测
    #     height, width = rgb.shape[:2]
        
    #     # if np.array_equal(depth, np.ones_like(depth)) and detections.num_detections > 0: # 暂时不存在该情况
    #     #     depth = self._infer_depth(rgb, min_depth, max_depth)
    #     #     obs = list(self._observations_cache["object_map_rgbd"][0])
    #     #     obs[1] = depth
    #     #     self._observations_cache["object_map_rgbd"][0] = tuple(obs)
        
        
    #     for idx in range(len(detections.logits)):
    #         bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
    #         object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist()) # 当前rgb中的每个检测mask

    #         # If we are using vqa, then use the BLIP2 model to visually confirm whether
    #         # the contours are actually correct.
            
    #         # # 不使用
    #         # if self._use_vqa:
    #         #     contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #         #     annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255, 0, 0), 2)
    #         #     question = f"Question: {self._vqa_prompt}"
    #         #     if not detections.phrases[idx].endswith("ing"):
    #         #         question += "a "
    #         #     question += detections.phrases[idx] + "? Answer:"
    #         #     answer = self._vqa.ask(annotated_rgb, question)
    #         #     if not answer.lower().startswith("yes"):
    #         #         continue

    #         self._object_map.update_map( # ObjectPointCloudMap类型
    #             self._target_object,
    #             depth,
    #             object_mask,
    #             tf_camera_to_episodic,
    #             min_depth,
    #             max_depth,
    #             fx,
    #             fy,
    #         ) # 将当前rgb观测+当前detect_mask对应的object_name对应的点云累计到self.clouds中

    #     cone_fov = get_fov(fx, depth.shape[1])
    #     self._object_map.update_explored(tf_camera_to_episodic, max_depth, cone_fov) # 过滤掉“在当前fov及dis范围下， 不正常（随机数不为1）的点云”


    #     return detections

    def _update_other_angle_object_map(
        self,
        semantic_mask,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> ObjectDetections: # 完成当前rgb检测，累计记录当前帧rgb上检测得到的所有mask结果，将当前rgb观测+当前detect_mask对应的object_name对应的点云累计到self.clouds中+过滤掉clouds中“在当前fov及dis范围下， 不正常（随机数不为1）的点云”
        """
        Updates the object map with the given rgb and depth images, and the given
        transformation matrix from the camera to the episodic coordinate frame.

        Args:
            rgb (np.ndarray): The rgb image to use for updating the object map. Used for
                object detection and Mobile SAM segmentation to extract better object
                point clouds.
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).
            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.

        Returns:
            ObjectDetections: The object detections from the object detector.
        """
        semantic_mask = semantic_mask[:, :, 0]
        mask_res_ls = []
        for temp_id in VLFMTrainer.current_episode_object_id:
            temp_mask = (semantic_mask==temp_id).cpu().numpy().astype(np.uint8)
            if np.sum(temp_mask)>0:
                mask_res_ls.append(temp_mask)  

        height, width = rgb.shape[:2]

        for idx in range(len(mask_res_ls)):
            # bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
            # object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist()) # 当前rgb中的每个检测mask
            object_mask = mask_res_ls[idx]


            self._object_map.update_map( # ObjectPointCloudMap类型
                self._target_object,
                depth,
                object_mask,
                tf_camera_to_episodic,
                min_depth,
                max_depth,
                fx,
                fy,
            ) # 将当前rgb观测+当前detect_mask对应的object_name对应的点云累计到self.clouds中

        cone_fov = get_fov(fx, depth.shape[1])
        self._object_map.update_explored(tf_camera_to_episodic, max_depth, cone_fov) # 过滤掉“在当前fov及dis范围下， 不正常（随机数不为1）的点云”
        return mask_res_ls
        

    def _update_other_angle_stair_map(
        self,
        semantic_mask,
        direct_id,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> ObjectDetections: # 完成当前rgb检测，累计记录当前帧rgb上检测得到的所有mask结果，将当前rgb观测+当前detect_mask对应的object_name对应的点云累计到self.clouds中+过滤掉clouds中“在当前fov及dis范围下， 不正常（随机数不为1）的点云”

        # ==============================================================
        # 检测楼梯
        semantic_mask = semantic_mask[:, :, 0]
        mask_res_ls = []
        for temp_id in VLFMTrainer.res_all_stairs_id:
            temp_mask = (semantic_mask==temp_id).cpu().numpy().astype(np.uint8)
            if np.sum(temp_mask)>0:
                mask_res_ls.append(temp_mask)  

        if(len(mask_res_ls)>0):
            self.current_step_stairs_res[direct_id] = mask_res_ls


    '''
    def _cache_observations(self, observations: "TensorDict") -> None:
        """Extracts the rgb, depth, and camera transform from the observations.

        Args:
            observations ("TensorDict"): The observations from the current timestep.
        """
        raise NotImplementedError
    '''

    def _cache_observations(self, observations: "TensorDict") -> None:
        if len(self._observations_cache) > 0:
            return

        rgb = observations["rgb"][0].cpu().numpy()
        depth = observations["depth"][0].cpu().numpy()
        x, y = observations["gps"][0].cpu().numpy()
        camera_yaw = observations["compass"][0].cpu().item()

        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None) # depth过滤
        camera_position = np.array([x, -y, self._camera_height])
        robot_xy = camera_position[:2]
        camera_pitch = np.radians(-self._pitch_angle) # 应该是弧度制 -
        camera_roll = 0
        tf_camera_to_episodic = xyz_yaw_pitch_roll_to_tf_matrix(camera_position, camera_yaw, camera_pitch, camera_roll)

        self._observations_cache = {
            "robot_xy": robot_xy,
            "robot_heading": camera_yaw,
            "tf_camera_to_episodic": tf_camera_to_episodic,
            "rgb": rgb,
            "depth": depth,
            "min_depth": self._min_depth,
            "max_depth": self._max_depth,
            "fx": self._fx,
            "fy": self._fy,
            "camera_fov": self._camera_fov,
            "habitat_start_yaw": observations["heading"][0].item(),
            "camera_position": camera_position,
            
        }

        self._observations_cache["nav_rgb"]=torch.unsqueeze(observations["rgb"][0], dim=0)
        self._observations_cache["nav_depth"]=torch.unsqueeze(observations["depth"][0], dim=0)


    def _infer_depth(self, rgb: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        """Infers the depth image from the rgb image.

        Args:
            rgb (np.ndarray): The rgb image to infer the depth from.

        Returns:
            np.ndarray: The inferred depth image.
        """
        raise NotImplementedError


@dataclass
class VLFMConfig:
    name: str = "HabitatITMPolicy"
    text_prompt: str = "Seems like there is a target_object ahead." # 看似有用
    # text_prompt: str = "target_object" # 看似有用
    pointnav_policy_path: str = "data/pointnav_weights.pth"
    depth_image_shape: Tuple[int, int] = (224, 224)
    pointnav_stop_radius: float = 0.9
    use_max_confidence: bool = False
    object_map_erosion_size: int = 5
    exploration_thresh: float = 0.0
    obstacle_map_area_threshold: float = 1.5  # in square meters
    min_obstacle_height: float = 0.61
    max_obstacle_height: float = 0.88
    hole_area_thresh: int = 100000
    use_vqa: bool = False
    vqa_prompt: str = "Is this "
    coco_threshold: float = 0.8
    non_coco_threshold: float = 0.4
    agent_radius: float = 0.18

    @classmethod  # type: ignore
    @property
    def kwaarg_names(cls) -> List[str]:
        # This returns all the fields listed above, except the name field
        return [f.name for f in fields(VLFMConfig) if f.name != "name"]


cs = ConfigStore.instance()
cs.store(group="policy", name="vlfm_config_base", node=VLFMConfig())
