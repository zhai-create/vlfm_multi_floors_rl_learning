# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

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

from vlfm.utils.geometry_utils import (
    extract_yaw,
    get_point_cloud,
    transform_points,
    within_fov_cone,
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

def merge_clusters_unique(cluster1: np.ndarray, cluster2: np.ndarray) -> np.ndarray:
    """
    高效融合两个点云簇，并去除重复的点 (优化版)。

    该函数使用 NumPy 的原生函数来提升性能。它首先将两个点云垂直堆叠，
    然后直接调用 np.unique(..., axis=0) 来找出所有唯一的行。
    这个操作在 NumPy 的底层 C 代码中执行，远快于在 Python 层面进行循环和类型转换。

    Args:
        cluster1 (np.ndarray): 第一个点云簇，一个 M x 2 的NumPy数组。
        cluster2 (np.ndarray): 第二个点云簇，一个 N x 2 的NumPy数组。

    Returns:
        np.ndarray: 一个融合且去重后的新点云簇 (K x 2)，其中 K 是唯一点的数量。
    """
    # 步骤 1: 将两个点云数组垂直堆叠成一个数组
    combined_array = np.vstack((cluster1, cluster2))

    # 步骤 2: 使用 np.unique 并指定 axis=0 直接找出唯一的行
    # 这是最高效且最简洁的方法
    merged_array = np.unique(combined_array, axis=0)

    return merged_array


class BaseObjectNavPolicy(BasePolicy):
    _target_object: str = ""
    _policy_info: Dict[str, Any] = {}
    _object_masks: Union[np.ndarray, Any] = None  # set by ._update_object_map()
    _stop_action: Union[Tensor, Any] = None  # MUST BE SET BY SUBCLASS
    _observations_cache: Dict[str, Any] = {}
    _non_coco_caption = ""
    _load_yolo: bool = True
    # _load_yolo: bool = False

    def __init__(
        self,
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


        self._use_vqa = use_vqa
        if use_vqa:
            self._vqa = BLIP2Client(port=int(os.environ.get("BLIP2_PORT", "12185")))
        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._object_map: ObjectPointCloudMap = ObjectPointCloudMap(erosion_size=object_map_erosion_size)
        self._depth_image_shape = tuple(depth_image_shape)
        self._pointnav_stop_radius = pointnav_stop_radius
        self._visualize = visualize
        self._vqa_prompt = vqa_prompt
        self._coco_threshold = coco_threshold
        self._non_coco_threshold = non_coco_threshold

        self._num_steps = 0
        self._did_reset = False
        self._last_goal = np.zeros(2)
        self._done_initializing = False
        self._called_stop = False
        self._compute_frontiers = compute_frontiers
        if compute_frontiers:
            self._obstacle_map = ObstacleMap(
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                area_thresh=obstacle_map_area_threshold,
                agent_radius=agent_radius,
                hole_area_thresh=hole_area_thresh,
            )


        self.last_target_map = None
        self.last_semantic_frontier_map = None
        self.last_value_map = None

        # frontier_revise
        self.false_frontier_ls = []

        # frontier_multi_navigate_revise
        self.false_frontier_multi_navigate_ls = []
        self.frontier_navigate_cnt_dict = {}
        # self.same_frontier_dis_thre = 0.15 # 原本0.5m
        self.same_frontier_dis_thre = 0.5 # 原本0.5m

        # new_object_revise_frontier
        self.new_object_frontier_ls = []
        self.total_object_frontier_ls = []

        # # is_need_closer
        # self.is_need_closer = False
        # self.closer_sub_goal = None
        # self.pos_buffer_ls = []
        # self.is_reverse = False
        # self.reverse_cnt = 0
        

    def _reset(self) -> None:
        self._target_object = ""
        self._pointnav_policy.reset() # 获得全0的pointnav_test_recurrent_hidden_states和pointnav_prev_actions
        self._object_map.reset() # 获得self.clouds为{}，并且self.last_target_coord为None
        self._last_goal = np.zeros(2)
        self._num_steps = 0
        self._done_initializing = False
        self._called_stop = False
        if self._compute_frontiers: # 默认为True
            self._obstacle_map.reset() # _navigable_map和explored_area全置为0，_frontiers_px和frontiers均置为[]
        self._did_reset = True

        self.last_target_map = None
        self.last_semantic_frontier_map = None
        self.last_value_map = None

        # frontier_revise
        self.false_frontier_ls = []

        # frontier_multi_navigate_revise
        self.false_frontier_multi_navigate_ls = []
        self.frontier_navigate_cnt_dict = {}
        # self.same_frontier_dis_thre = 0.15 # 原本0.5m
        self.same_frontier_dis_thre = 0.5 # 原本0.5m

        # new_object_revise_frontier
        self.new_object_frontier_ls = []
        self.total_object_frontier_ls = []

        # # is_need_closer
        # self.is_need_closer = False
        # self.closer_sub_goal = None
        # self._pointnav_stop_radius = 0.9
        # self.pos_buffer_ls = []
        # self.is_reverse = False
        # self.reverse_cnt = 0
        

    def _xy_to_px(self, points: np.ndarray) -> np.ndarray:
        """Converts an array of (x, y) coordinates to pixel coordinates.

        Args:
            points: The array of (x, y) coordinates to convert.

        Returns:
            The array of (x, y) pixel coordinates.
        """
        px = np.rint(points[:, ::-1] * 20) + np.array([500, 500])
        px[:, 0] = 1000 - px[:, 0]
        return px.astype(int)


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
        # get::当前帧的"frontier_sensor"信息
        self._pre_step(observations, masks) # 根据条件执行reset, 获得当前帧观测信息(存在_observations_cache中)，self._policy_info为空

        object_map_rgbd = self._observations_cache["object_map_rgbd"]
        detections = [
            self._update_object_map(rgb, depth, tf, min_depth, max_depth, fx, fy) # 完成当前rgb检测，累计记录当前帧rgb上检测得到的所有mask结果，将当前rgb观测+当前detect_mask对应的object_name对应的点云累计到self.clouds中+过滤掉clouds中“在当前fov及dis范围下， 不正常（随机数不为1）的点云”
            for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
        ] # get::当前帧的semantic_map
        # get::当前帧的机器人位置
        
        # new_object_revise_frontier (frontier_revise)
        new_right_object_frontier_ls = []
        for temp_i in range(len(self.new_object_frontier_ls)): # (N, 2)
            should_add = True
            for temp_j in range(len(self.false_frontier_ls)):
                temp_dis = ((self.new_object_frontier_ls[temp_i][0]-self.false_frontier_ls[temp_j][0])**2+(self.new_object_frontier_ls[temp_i][1]-self.false_frontier_ls[temp_j][1])**2)**0.5
                if temp_dis<0.1:
                    should_add = False
                    break
            if(should_add==True):
                new_right_object_frontier_ls.append(self.new_object_frontier_ls[temp_i])

        # new_object_revise_frontier(frontier_multi_navigate_revise)
        final_object_right_frontier_ls = []
        new_right_object_frontier_ls = np.array(new_right_object_frontier_ls)
        for temp_i in range(len(new_right_object_frontier_ls)):
            should_add = True
            for temp_frontier_key in self.frontier_navigate_cnt_dict:
                # if(self.frontier_navigate_cnt_dict[temp_frontier_key]>=15):
                if(self.frontier_navigate_cnt_dict[temp_frontier_key]>10):
                # if(self.frontier_navigate_cnt_dict[temp_frontier_key]>100000):
                    temp_dis = ((temp_frontier_key[0]-new_right_object_frontier_ls[temp_i][0])**2+(temp_frontier_key[1]-new_right_object_frontier_ls[temp_i][1])**2)**0.5
                    if(temp_dis<self.same_frontier_dis_thre):
                        should_add = False
                        break
            if(should_add==True):
                final_object_right_frontier_ls.append(new_right_object_frontier_ls[temp_i])
        final_object_right_frontier_ls = np.array(final_object_right_frontier_ls)

        if(len(final_object_right_frontier_ls)>0):
            if(len(self.total_object_frontier_ls)==0):
                self.total_object_frontier_ls = final_object_right_frontier_ls
            else:
                self.total_object_frontier_ls = merge_clusters_unique(self.total_object_frontier_ls, final_object_right_frontier_ls)

        # new_object_revise_frontier(delete_frontier)
        now_total_object_frontier_ls = copy.deepcopy(self.total_object_frontier_ls)
        real_total_object_frontier_ls = []
        tf_camera_to_episodic = object_map_rgbd[0][2]
        camera_coordinates = tf_camera_to_episodic[:3, 3]
        camera_yaw = extract_yaw(tf_camera_to_episodic)
        
        for temp_index in range(len(now_total_object_frontier_ls)):
            temp_dis = ((self._observations_cache["robot_xy"][0]-now_total_object_frontier_ls[temp_index][0])**2+(self._observations_cache["robot_xy"][1]-now_total_object_frontier_ls[temp_index][1])**2)**0.5
            
            temp_direction_y = now_total_object_frontier_ls[temp_index][1] - self._observations_cache["robot_xy"][1]
            temp_direction_x = now_total_object_frontier_ls[temp_index][0] - self._observations_cache["robot_xy"][0]
            temp_angle = np.arctan2(temp_direction_y, temp_direction_x)
            temp_angle_diff = np.mod(temp_angle - camera_yaw + np.pi, 2 * np.pi) - np.pi
            if(np.abs(temp_angle_diff) <= (79*np.pi/180) / 2) and (temp_dis<0.5):
                continue
            else:
                real_total_object_frontier_ls.append(now_total_object_frontier_ls[temp_index])

        self.total_object_frontier_ls = np.array(real_total_object_frontier_ls)
        if len(self.total_object_frontier_ls)>0:
            if(len(self._observations_cache["frontier_sensor"])==0):
                self._observations_cache["frontier_sensor"] = self.total_object_frontier_ls # 当前帧最终真正需要的frontier
            else:
                self._observations_cache["frontier_sensor"] = np.vstack((self._observations_cache["frontier_sensor"], self.total_object_frontier_ls))


        self._observations_cache["semantic_map"] = self._object_map.semantic_map
        robot_xy = self._observations_cache["robot_xy"]
        robot_yaw = self._observations_cache["robot_heading"]

        print("robot_xy:", robot_xy)
        print("robot_yaw:", robot_yaw)

        goal = self._get_target_object_location(robot_xy) # 如果看到了goal，则返回其最佳位置；否则返回None

        # _done_initializing: 没有转弯一直是False，转弯后是True

        # # is_need_closer
        # if (self.is_reverse == True):
        #     mode = "navigate"
        #     if(self.reverse_cnt==6):
        #         pointnav_action = torch.tensor([[1]], dtype=torch.long)
        #     elif(self.reverse_cnt==7):
        #         pointnav_action = torch.tensor([[0]], dtype=torch.long)
        #     else:
        #         pointnav_action = torch.tensor([[2]], dtype=torch.long)
        #     self.reverse_cnt += 1
        #     os.environ["DEBUG_INFO"] = "reverse_rethink"

        if not self._done_initializing:  # Initialize
            mode = "initialize"
            pointnav_action = self._initialize()
            os.environ["DEBUG_INFO"] = ""
        else:
            # # is_need_closer
            # if (self.is_need_closer==True):
            #     self.no_action_last_target_coord = None # object_no_action_revise
            #     mode = "navigate"
                    
            #     real_sub_goal_array = self.closer_sub_goal
            #     # if(os.environ["DEBUG_INFO"] != "no_action_navigate"):
            #     #     os.environ["DEBUG_INFO"] = "closer_navigate"

            #     os.environ["DEBUG_INFO"] = "closer_navigate"

            #     self.pos_buffer_ls.append(robot_xy) # 执行完动作以后，直接add位置
            #     now_closer_goal_dis = ((self.pos_buffer_ls[-1][0]-self.closer_sub_goal[0])**2+(self.pos_buffer_ls[-1][1]-self.closer_sub_goal[1])**2)**0.5
            #     last_closer_goal_dis = ((self.pos_buffer_ls[-2][0]-self.closer_sub_goal[0])**2+(self.pos_buffer_ls[-2][1]-self.closer_sub_goal[1])**2)**0.5

            #     if(now_closer_goal_dis>last_closer_goal_dis):
            #         pointnav_action = torch.tensor([[2]], dtype=torch.long)
            #         self.is_reverse = True
            #         self.reverse_cnt += 1
            #     else:
            #         pointnav_action = self._pointnav(real_sub_goal_array, stop=True)

            if ((len(self._observations_cache["frontier_sensor"])+np.sum(self._observations_cache["semantic_map"]))>0):
                # =====> 准备数据 <=====
                # frontier_map
                if(len(self._observations_cache["frontier_sensor"])>0):
                    now_frontier_map = np.zeros((1000, 1000))
                    xy_points = self._observations_cache["frontier_sensor"][:, :2]
                    pixel_points = self._xy_to_px(xy_points) # 点云的在map栅格上的像素坐标
                    now_frontier_map[pixel_points[:, 1], pixel_points[:, 0]] = 1 # 计算语义地图（1为相应类别）
                else:
                    now_frontier_map = np.zeros((1000, 1000))

                # semantic_map
                now_semantic_map = self._observations_cache["semantic_map"]

                # semantic_frontier_map
                semantic_frontier_map = np.zeros_like(now_semantic_map, dtype=np.float32)
                semantic_frontier_map[now_frontier_map == 1] = 0.5
                semantic_frontier_map[now_semantic_map == 1] = 1 # 1

                # value_map
                now_value_map = observations['value_map'][:, :, 0]

                # # robot_xy_map
                # robot_row = 1000-(int(-robot_xy[0] * 20) + 500)
                # robot_col = int(-robot_xy[1] * 20) + 500
                # col_map, row_map = np.meshgrid(np.arange(1000), np.arange(1000)) # 每列一个数，每行一个数
                # dist = np.sqrt((col_map - robot_col)**2 + (row_map - robot_row)**2)
                # now_xy_map = np.exp(-dist**2 / (2 * (1000/20)**2))

                # # sin_cos_map
                # dx = col_map - robot_col
                # dy = row_map - robot_row
                # abs_angle = np.arctan2(dy, dx)
                # rel_angle = abs_angle - robot_yaw
                # now_sin_map = np.sin(rel_angle)
                # now_cos_map = np.cos(rel_angle)

                # robot_state_map
                robot_row = 1000-(int(-robot_xy[0] * 20) + 500)
                robot_col = int(-robot_xy[1] * 20) + 500
                robot_state_map = np.zeros_like(now_semantic_map, dtype=np.float32)

                epsilon = 1e-5
                now_robot_yaw = normalize_angle(robot_yaw)
                if(now_robot_yaw<0.001):
                    now_robot_yaw += epsilon
                # assert now_robot_yaw>0 and now_robot_yaw<=1
                robot_state_map[robot_row][robot_col] = now_robot_yaw # 3


                if(self.last_value_map is None):
                    # =====> 第一帧启发式target_map <=====
                    target_map = np.zeros((1000, 1000), dtype=np.float32)
                    if(goal is None):
                        pointnav_action, best_frontier = self._explore(observations)
                        target_sub_goal = np.array([[best_frontier[0], best_frontier[1]]])
                    else:
                        target_sub_goal = np.array([[goal[0], goal[1]]])
                    xy_points = target_sub_goal
                    pixel_points = self._xy_to_px(xy_points) # 点云的在map栅格上的像素坐标
                    
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
                    self.last_semantic_frontier_map = copy.deepcopy(semantic_frontier_map)
                    self.last_value_map = copy.deepcopy(now_value_map)
                    self.last_target_map = copy.deepcopy(target_map)

                # resize_gai
                resize_last_semantic_frontier_map = compress_map_ultrafast_not_zero(self.last_semantic_frontier_map)
                resize_last_value_map = compress_map_ultrafast_not_zero(self.last_value_map)
                resize_last_target_map = compress_map_ultrafast_not_zero(self.last_target_map)
                resize_semantic_frontier_map = compress_map_ultrafast_not_zero(semantic_frontier_map)
                resize_now_value_map = compress_map_ultrafast_not_zero(now_value_map)
                resize_robot_state_map = compress_map_ultrafast_not_zero(robot_state_map)

                '''
                # 输入地图数据
                input_data = np.stack([
                    self.last_semantic_frontier_map, self.last_value_map, self.last_target_map,
                    semantic_frontier_map, now_value_map, robot_state_map
                ], axis=0)
                '''

                # resize_gai
                # 输入地图数据
                input_data = np.stack([
                    resize_last_semantic_frontier_map, resize_last_value_map, resize_last_target_map,
                    resize_semantic_frontier_map, resize_now_value_map, resize_robot_state_map
                ], axis=0)

                input_data = torch.FloatTensor(input_data)

                '''
                # 输入mask数据
                superimposed_mask = (semantic_frontier_map > 0).astype(np.float32)
                curr_mask = torch.FloatTensor(superimposed_mask)
                '''

                # resize_gai
                # 输入mask数据
                superimposed_mask = (resize_semantic_frontier_map > 0).astype(np.float32)
                curr_mask = torch.FloatTensor(superimposed_mask)

                input_data = input_data.unsqueeze(0).to("cuda")  # [1, seq_len]
                curr_mask = curr_mask.unsqueeze(0).to("cuda")  # [1, seq_len]


                if(need_replan==True):
                    VLFMTrainer.policy_model.eval()
                    with torch.no_grad():
                        preds = VLFMTrainer.policy_model(input_data)
                        # preds = torch.sigmoid(preds)
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
                        

                        '''
                        softmax_probs = F.softmax(torch.from_numpy(masked_pred_flat)/10, dim=0)
                        sampled_index = Categorical(softmax_probs).sample()
                        pred_row = sampled_index.item() // 1000
                        pred_col = sampled_index.item() % 1000
                        '''
                else:
                    '''
                    rows, cols = np.where(self.last_target_map == 1)
                    '''
                    # resize_gai
                    rows, cols = np.where(resize_last_target_map == 1)
                    pred_row, pred_col = rows[0], cols[0]

                '''
                pred_row, pred_col = int(pred_row), int(pred_col)
                pre_x, pre_y = (pred_row-500)/20, (500-pred_col)/20
                sub_goal_array = np.array([pre_x, pre_y])
                '''

                # resize_gai
                pred_row, pred_col = int(pred_row), int(pred_col)
                resize_pred_row, resize_pred_col = int(np.clip(pred_row*2, 0, 1000-1)), int(np.clip(pred_col*2, 0, 1000-1))
                pre_x, pre_y = (resize_pred_row-500)/20, (500-resize_pred_col)/20
                sub_goal_array = np.array([pre_x, pre_y])

                '''
                if(now_semantic_map[pred_row, pred_col]>0):
                '''
                # resize_gai
                if(resize_semantic_frontier_map[pred_row, pred_col]==1):
                    mode = "navigate"
                    navigate_target_cloud = self._object_map.get_target_cloud(self._target_object)
                    if(len(navigate_target_cloud)>0):
                        real_sub_goal_array, real_min_index = find_closest_vector(sub_goal_array, navigate_target_cloud[:, :2])
                        real_mean_center = find_dense_subcluster_dbscan(points=navigate_target_cloud[:, :2], real_sub_goal_array=real_sub_goal_array, eps=1.0, min_samples=1)
                        real_sub_goal_array, real_min_index = find_closest_vector(real_mean_center, navigate_target_cloud[:, :2])
                        
                        pointnav_action = self._pointnav(real_sub_goal_array, stop=True)
                        os.environ["DEBUG_INFO"] = ""
                    else:
                        navigate_target_cloud = None # is_need_closer
                        real_sub_goal_array = sub_goal_array
                        pointnav_action = self._pointnav(real_sub_goal_array, stop=True)
                        os.environ["DEBUG_INFO"] = "normal_navigate"

                    # # is_need_closer
                    # if(pointnav_action[0][0].cpu().numpy()==0) and (navigate_target_cloud is not None):
                    #     sorted_navigate_target_cloud = sort_by_euclidean_distance(robot_xy, navigate_target_cloud[:, :2])

                    #     # # 暂时修改
                    #     temp_real_sub_goal_array = sorted_navigate_target_cloud[0]
                    #     temp_pointnav_action = self._pointnav(temp_real_sub_goal_array, stop=True)
                    #     if(temp_pointnav_action[0][0].cpu().numpy()==0):
                    #         self.is_need_closer = False
                    #         self.closer_sub_goal = None
                    #     else:
                    #         self.is_need_closer = True
                    #         self.closer_sub_goal = temp_real_sub_goal_array
                    #         real_sub_goal_array = temp_real_sub_goal_array
                    #         pointnav_action = temp_pointnav_action
                    #         self._pointnav_stop_radius = 0.5

                    #         self.pos_buffer_ls.append(robot_xy)

                else:
                    mode = "explore"
                    explore_target_cloud = self._observations_cache["frontier_sensor"]
                    
                    if(len(explore_target_cloud)>0):    
                        real_sub_goal_array, real_min_index = find_closest_vector(sub_goal_array, explore_target_cloud[:, :2])
                        pointnav_action = self._pointnav(real_sub_goal_array, stop=False)
                    
                        # frontier_multi_navigate_revise
                        if(need_replan==True):
                            min_dis = 100000
                            min_key = None
                            for temp_key in self.frontier_navigate_cnt_dict:
                                temp_dis = ((temp_key[0]-real_sub_goal_array[0])**2+(temp_key[1]-real_sub_goal_array[1])**2)**0.5
                                if(temp_dis<self.same_frontier_dis_thre):
                                    if (temp_dis<min_dis):
                                        min_dis = temp_dis
                                        min_key = temp_key
                            if(min_key is not None):
                                self.frontier_navigate_cnt_dict[min_key] += 1
                            else:
                                self.frontier_navigate_cnt_dict[(real_sub_goal_array[0], real_sub_goal_array[1])] = 1

                        
                        # frontier_revise
                        if(pointnav_action[0][0].cpu().numpy()==0):
                            self.false_frontier_ls.append(real_sub_goal_array)
                            pointnav_action = torch.tensor([[2]], dtype=torch.long)

                        
                    else:
                        mode = "navigate"
                        pointnav_action = self._pointnav(sub_goal_array, stop=True)

                    # radius_px = int(0.5 * 20)
                    # now_best_value = pixel_value_within_radius(now_value_map, (pred_row, pred_col), radius_px)
                    # os.environ["DEBUG_INFO"] = f"Best value: {now_best_value*100:.2f}%"

                '''
                # =====> latst_target_map <=====
                last_target_map = np.zeros((1000, 1000), dtype=np.float32)
                last_target_map[pred_row, pred_col] = 1 # 计算语义地图（1为相应类别）
                # =====> latst_target_map <=====
                '''

                # resize_gai
                # =====> latst_target_map <=====
                last_target_map = np.zeros((1000, 1000), dtype=np.float32)
                last_target_map[resize_pred_row, resize_pred_col] = 1 # 计算语义地图（1为相应类别）
                # =====> latst_target_map <=====

                self.last_semantic_frontier_map = copy.deepcopy(semantic_frontier_map)
                self.last_value_map = copy.deepcopy(now_value_map)
                self.last_target_map = copy.deepcopy(last_target_map)
                
            else: # 如果action_space为空，则直接返回
                mode = "no action"
                pointnav_action = self._stop_action
                os.environ["DEBUG_INFO"] = ""
        
        action_numpy = pointnav_action.detach().cpu().numpy()[0]
        if len(action_numpy) == 1:
            action_numpy = action_numpy[0]
        print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")
        self._policy_info.update(self._get_policy_info(detections[0])) # 每act一次，有一个新的_policy_info
        self._num_steps += 1

        self._observations_cache = {} # 每act一次，有一个新的_observations_cache
        self._did_reset = False # 每act结束一次后，_did_reset都变为False

        print("==========> <==========")
        # print("\n\n\n\n\n")

        return pointnav_action, rnn_hidden_states, None


    # def act(
    #     self,
    #     observations: Dict, # 新的episode reset or 执行完上一个step
    #     rnn_hidden_states: Any,
    #     prev_actions: Any,
    #     masks: Tensor,
    #     deterministic: bool = False,
    # ) -> Any:
    #     """
    #     Starts the episode by 'initializing' and allowing robot to get its bearings
    #     (e.g., spinning in place to get a good view of the scene).
    #     Then, explores the scene until it finds the target object.
    #     Once the target object is found, it navigates to the object.
    #     """
    #     # get::当前帧的"frontier_sensor"信息
    #     self._pre_step(observations, masks) # 根据条件执行reset, 获得当前帧观测信息(存在_observations_cache中)，self._policy_info为空

    #     object_map_rgbd = self._observations_cache["object_map_rgbd"]
    #     detections = [
    #         self._update_object_map(rgb, depth, tf, min_depth, max_depth, fx, fy) # 完成当前rgb检测，累计记录当前帧rgb上检测得到的所有mask结果，将当前rgb观测+当前detect_mask对应的object_name对应的点云累计到self.clouds中+过滤掉clouds中“在当前fov及dis范围下， 不正常（随机数不为1）的点云”
    #         for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
    #     ] # get::当前帧的semantic_map
    #     # get::当前帧的机器人位置

    #     self._observations_cache["semantic_map"] = self._object_map.semantic_map

    #     robot_xy = self._observations_cache["robot_xy"]
        

    #     robot_yaw = self._observations_cache["robot_heading"]

    #     print("robot_xy:", robot_xy)
    #     print("robot_yaw:", robot_yaw)

    #     goal = self._get_target_object_location(robot_xy) # 如果看到了goal，则返回其最佳位置；否则返回None

    #     # _done_initializing: 没有转弯一直是False，转弯后是True

    #     if not self._done_initializing:  # Initialize
    #         mode = "initialize"
    #         pointnav_action = self._initialize()
    #         os.environ["DEBUG_INFO"] = ""
    #     else:
    #         if ((len(self._observations_cache["frontier_sensor"])+np.sum(self._observations_cache["semantic_map"]))>0):
    #             # =====> 准备数据 <=====
    #             # frontier_map
    #             if(len(self._observations_cache["frontier_sensor"])>0):
    #                 now_frontier_map = np.zeros((1000, 1000))
    #                 xy_points = self._observations_cache["frontier_sensor"][:, :2]
    #                 pixel_points = self._xy_to_px(xy_points) # 点云的在map栅格上的像素坐标
    #                 now_frontier_map[pixel_points[:, 1], pixel_points[:, 0]] = 1 # 计算语义地图（1为相应类别）
    #             else:
    #                 now_frontier_map = np.zeros((1000, 1000))

    #             # semantic_map
    #             now_semantic_map = self._observations_cache["semantic_map"]

    #             # semantic_frontier_map
    #             semantic_frontier_map = np.zeros_like(now_semantic_map, dtype=np.float32)
    #             semantic_frontier_map[now_frontier_map == 1] = 0.5
    #             semantic_frontier_map[now_semantic_map == 1] = 1 # 1

    #             # value_map
    #             now_value_map = observations['value_map'][:, :, 0]

    #             # # robot_xy_map
    #             # robot_row = 1000-(int(-robot_xy[0] * 20) + 500)
    #             # robot_col = int(-robot_xy[1] * 20) + 500
    #             # col_map, row_map = np.meshgrid(np.arange(1000), np.arange(1000)) # 每列一个数，每行一个数
    #             # dist = np.sqrt((col_map - robot_col)**2 + (row_map - robot_row)**2)
    #             # now_xy_map = np.exp(-dist**2 / (2 * (1000/20)**2))

    #             # # sin_cos_map
    #             # dx = col_map - robot_col
    #             # dy = row_map - robot_row
    #             # abs_angle = np.arctan2(dy, dx)
    #             # rel_angle = abs_angle - robot_yaw
    #             # now_sin_map = np.sin(rel_angle)
    #             # now_cos_map = np.cos(rel_angle)

    #             # robot_state_map
    #             robot_row = 1000-(int(-robot_xy[0] * 20) + 500)
    #             robot_col = int(-robot_xy[1] * 20) + 500
    #             robot_state_map = np.zeros_like(now_semantic_map, dtype=np.float32)

    #             epsilon = 1e-5
    #             now_robot_yaw = normalize_angle(robot_yaw)
    #             if(now_robot_yaw<0.001):
    #                 now_robot_yaw += epsilon
    #             # assert now_robot_yaw>0 and now_robot_yaw<=1
    #             robot_state_map[robot_row][robot_col] = now_robot_yaw # 3


    #             if(self.last_value_map is None):
    #                 # =====> 第一帧启发式target_map <=====
    #                 target_map = np.zeros((1000, 1000), dtype=np.float32)
    #                 if(goal is None):
    #                     pointnav_action, best_frontier = self._explore(observations)
    #                     target_sub_goal = np.array([[best_frontier[0], best_frontier[1]]])
    #                 else:
    #                     target_sub_goal = np.array([[goal[0], goal[1]]])
    #                 xy_points = target_sub_goal
    #                 pixel_points = self._xy_to_px(xy_points) # 点云的在map栅格上的像素坐标
                    
    #                 if(goal is None):
    #                     if (now_frontier_map[pixel_points[0, 1], pixel_points[0, 0]] == 1):
    #                         target_map[pixel_points[:, 1], pixel_points[:, 0]] = 1 # 计算语义地图（1为相应类别）
    #                     else:
    #                         now_frontier_map_one_position = np.argwhere(now_frontier_map == 1)
    #                         real_row_col_array = find_closest_vector(np.array([pixel_points[0, 1], pixel_points[0, 0]]), now_frontier_map_one_position)
                            
    #                         pixel_points = np.array([[real_row_col_array[1], real_row_col_array[0]]])
    #                         target_map[pixel_points[:, 1], pixel_points[:, 0]] = 1 # 计算语义地图（1为相应类别）
    #                 else:
    #                     if (now_semantic_map[pixel_points[0, 1], pixel_points[0, 0]] == 1):
    #                         target_map[pixel_points[:, 1], pixel_points[:, 0]] = 1 # 计算语义地图（1为相应类别）
    #                     else:
    #                         now_semantic_map_one_position = np.argwhere(now_semantic_map == 1)
    #                         real_row_col_array = find_closest_vector(np.array([pixel_points[0, 1], pixel_points[0, 0]]), now_semantic_map_one_position)
                            
    #                         pixel_points = np.array([[real_row_col_array[1], real_row_col_array[0]]])
    #                         target_map[pixel_points[:, 1], pixel_points[:, 0]] = 1 # 计算语义地图（1为相应类别）

    #                 # assert (len(pixel_points)==1)
    #                 # =====> 第一帧启发式target_map <=====
    #                 self.last_semantic_frontier_map = copy.deepcopy(semantic_frontier_map)
    #                 self.last_value_map = copy.deepcopy(now_value_map)
    #                 self.last_target_map = copy.deepcopy(target_map)

    #             # 输入地图数据
    #             input_data = np.stack([
    #                 self.last_semantic_frontier_map, self.last_value_map, self.last_target_map,
    #                 semantic_frontier_map, now_value_map, robot_state_map
    #             ], axis=0)

    #             input_data = torch.FloatTensor(input_data)

    #             # =====> new_add <=====
    #             if (goal is not None):
    #                 target_sub_goal = np.array([[goal[0], goal[1]]])
    #                 xy_points = target_sub_goal
    #                 pixel_points = self._xy_to_px(xy_points) # 点云的在map栅格上的像素坐标

    #                 if (semantic_frontier_map[pixel_points[0, 1], pixel_points[0, 0]] == 1):
    #                     pred_row = pixel_points[:, 1]
    #                     pred_col = pixel_points[:, 0]
    #                 else:
    #                     now_semantic_map_one_position = np.argwhere(semantic_frontier_map == 1)
    #                     real_row_col_array = find_closest_vector(np.array([pixel_points[0, 1], pixel_points[0, 0]]), now_semantic_map_one_position)
                        
    #                     pixel_points = np.array([[real_row_col_array[1], real_row_col_array[0]]])
    #                     pred_row = pixel_points[:, 1]
    #                     pred_col = pixel_points[:, 0]

    #             else:
    #                 # 输入mask数据
    #                 superimposed_mask = (semantic_frontier_map > 0).astype(np.float32)
    #                 curr_mask = torch.FloatTensor(superimposed_mask)

    #                 input_data = input_data.unsqueeze(0).to("cuda")  # [1, seq_len]
    #                 curr_mask = curr_mask.unsqueeze(0).to("cuda")  # [1, seq_len]

    #                 VLFMTrainer.policy_model.eval()
    #                 with torch.no_grad():
    #                     preds = VLFMTrainer.policy_model(input_data)
    #                     preds = torch.sigmoid(preds)
    #                     pred_map = preds[0, 0].detach().cpu().numpy()
    #                     mask_map = curr_mask[0].cpu().numpy()

    #                     masked_pred = pred_map.copy()
    #                     masked_pred[mask_map < 0.5] = 0

    #                     pred_row, pred_col = np.unravel_index(masked_pred.argmax(), masked_pred.shape)
    #             # =====> new_add <=====

    #             pred_row, pred_col = int(pred_row), int(pred_col)
    #             pre_x, pre_y = (pred_row-500)/20, (500-pred_col)/20
    #             sub_goal_array = np.array([pre_x, pre_y])

    #             if(now_semantic_map[pred_row, pred_col]>0):
    #                 mode = "navigate"
    #                 navigate_target_cloud = self._object_map.get_target_cloud(self._target_object)
    #                 real_sub_goal_array = find_closest_vector(sub_goal_array, navigate_target_cloud[:, :2])
    #                 pointnav_action = self._pointnav(real_sub_goal_array, stop=True)
    #                 os.environ["DEBUG_INFO"] = ""

    #             else:
    #                 mode = "explore"
    #                 explore_target_cloud = self._observations_cache["frontier_sensor"]
    #                 real_sub_goal_array = find_closest_vector(sub_goal_array, explore_target_cloud[:, :2])
    #                 pointnav_action = self._pointnav(real_sub_goal_array, stop=False)

    #                 radius_px = int(0.5 * 20)
    #                 now_best_value = pixel_value_within_radius(now_value_map, (pred_row, pred_col), radius_px)
    #                 os.environ["DEBUG_INFO"] = f"Best value: {now_best_value*100:.2f}%"


    #             # =====> latst_target_map <=====
    #             last_target_map = np.zeros((1000, 1000), dtype=np.float32)
    #             last_target_map[pred_row, pred_col] = 1 # 计算语义地图（1为相应类别）
    #             # =====> latst_target_map <=====
                
    #             self.last_semantic_frontier_map = copy.deepcopy(semantic_frontier_map)
    #             self.last_value_map = copy.deepcopy(now_value_map)
    #             self.last_target_map = copy.deepcopy(last_target_map)
                
    #         else: # 如果action_space为空，则直接返回
    #             mode = "no action"
    #             pointnav_action = self._stop_action
    #             os.environ["DEBUG_INFO"] = ""
        
    #     action_numpy = pointnav_action.detach().cpu().numpy()[0]
    #     if len(action_numpy) == 1:
    #         action_numpy = action_numpy[0]
    #     print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")
    #     self._policy_info.update(self._get_policy_info(detections[0])) # 每act一次，有一个新的_policy_info
    #     self._num_steps += 1

    #     self._observations_cache = {} # 每act一次，有一个新的_observations_cache
    #     self._did_reset = False # 每act结束一次后，_did_reset都变为False

    #     print("==========> <==========")
    #     # print("\n\n\n\n\n")

    #     return pointnav_action, rnn_hidden_states, None

    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None: # 根据条件执行reset, 获得当前帧观测信息(存在_observations_cache中)，self._policy_info为空
        
        assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        if not self._did_reset and masks[0] == 0: # reset之前，_did_reset为False
            self._reset()
            self._target_object = observations["objectgoal"]
        try:
            self._cache_observations(observations) # 根据当前观察，得到rgb，depth，frontiers，xy，yaw等信息
        
            # frontier_revise
            now_right_frontier_ls = []
            for temp_i in range(len(self._observations_cache["frontier_sensor"])): # (N, 2)
                should_add = True
                for temp_j in range(len(self.false_frontier_ls)):
                    temp_dis = ((self._observations_cache["frontier_sensor"][temp_i][0]-self.false_frontier_ls[temp_j][0])**2+(self._observations_cache["frontier_sensor"][temp_i][1]-self.false_frontier_ls[temp_j][1])**2)**0.5
                    if temp_dis<0.5:
                        should_add = False
                        break
                if(should_add==True):
                    now_right_frontier_ls.append(self._observations_cache["frontier_sensor"][temp_i])

            # frontier_multi_navigate_revise
            final_right_frontier_ls = []
            now_right_frontier_ls = np.array(now_right_frontier_ls)
            for temp_i in range(len(now_right_frontier_ls)):
                should_add = True
                for temp_frontier_key in self.frontier_navigate_cnt_dict:
                    # if(self.frontier_navigate_cnt_dict[temp_frontier_key]>=15):
                    if(self.frontier_navigate_cnt_dict[temp_frontier_key]>10):
                    # if(self.frontier_navigate_cnt_dict[temp_frontier_key]>100000):
                        temp_dis = ((temp_frontier_key[0]-now_right_frontier_ls[temp_i][0])**2+(temp_frontier_key[1]-now_right_frontier_ls[temp_i][1])**2)**0.5
                        if(temp_dis<self.same_frontier_dis_thre):
                            should_add = False
                            break
                if(should_add==True):
                    final_right_frontier_ls.append(now_right_frontier_ls[temp_i])
            self._observations_cache["frontier_sensor"] = np.array(final_right_frontier_ls) # 当前帧最终真正需要的frontier

        except IndexError as e:
            print(e)
            print("Reached edge of map, stopping.")
            raise StopIteration
        self._policy_info = {}

    def _initialize(self) -> Tensor:
        raise NotImplementedError

    def _explore(self, observations: "TensorDict") -> Tensor:
        raise NotImplementedError

    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]: # 如果看到了goal，则返回其最佳位置；否则返回None
        if self._object_map.has_object(self._target_object):
            return self._object_map.get_best_object(self._target_object, position) # 根据“连续多帧target点云的变化情况+机器人与target点云的距离”-->"最佳target点云"
        else:
            return None

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
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
        }

        if not self._visualize:
            return policy_info

        annotated_depth = self._observations_cache["object_map_rgbd"][0][1] * 255
        annotated_depth = cv2.cvtColor(annotated_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if self._object_masks.sum() > 0:
            # If self._object_masks isn't all zero, get the object segmentations and
            # draw them on the rgb and depth images
            contours, _ = cv2.findContours(self._object_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            annotated_rgb = cv2.drawContours(detections.annotated_frame, contours, -1, (255, 0, 0), 2)
            annotated_depth = cv2.drawContours(annotated_depth, contours, -1, (255, 0, 0), 2)
        else:
            annotated_rgb = self._observations_cache["object_map_rgbd"][0][0]
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

    def _update_object_map(
        self,
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
        # new_object_revise_frontier
        self.new_object_frontier_ls = []


        detections = self._get_object_detections(rgb) # 根据开集和闭集模型同时得到所有检测
        height, width = rgb.shape[:2]
        self._object_masks = np.zeros((height, width), dtype=np.uint8) # 累计记录当前帧rgb上检测得到的所有mask结果
        if np.array_equal(depth, np.ones_like(depth)) and detections.num_detections > 0: # 暂时不存在该情况
            depth = self._infer_depth(rgb, min_depth, max_depth)
            obs = list(self._observations_cache["object_map_rgbd"][0])
            obs[1] = depth
            self._observations_cache["object_map_rgbd"][0] = tuple(obs)
        for idx in range(len(detections.logits)):
            bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
            object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist()) # 当前rgb中的每个检测mask

            # If we are using vqa, then use the BLIP2 model to visually confirm whether
            # the contours are actually correct.
            
            # 不使用
            if self._use_vqa:
                contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255, 0, 0), 2)
                question = f"Question: {self._vqa_prompt}"
                if not detections.phrases[idx].endswith("ing"):
                    question += "a "
                question += detections.phrases[idx] + "? Answer:"
                answer = self._vqa.ask(annotated_rgb, question)
                if not answer.lower().startswith("yes"):
                    continue

            self._object_masks[object_mask > 0] = 1
            new_object_frontier = self._object_map.update_map( # ObjectPointCloudMap类型 # new_object_revise_frontier
                self._target_object,
                depth,
                object_mask,
                tf_camera_to_episodic,
                min_depth,
                max_depth,
                fx,
                fy,
            ) # 将当前rgb观测+当前detect_mask对应的object_name对应的点云累计到self.clouds中

            if(new_object_frontier is not None):
                self.new_object_frontier_ls.append(new_object_frontier)  # new_object_revise_frontier
        self.new_object_frontier_ls = np.array(self.new_object_frontier_ls) # new_object_revise_frontier

        cone_fov = get_fov(fx, depth.shape[1])
        self._object_map.update_explored(tf_camera_to_episodic, max_depth, cone_fov) # 过滤掉“在当前fov及dis范围下， 不正常（随机数不为1）的点云”

        self._object_map.get_semantic_map(self._target_object)

        return detections

    def _cache_observations(self, observations: "TensorDict") -> None:
        """Extracts the rgb, depth, and camera transform from the observations.

        Args:
            observations ("TensorDict"): The observations from the current timestep.
        """
        raise NotImplementedError

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
