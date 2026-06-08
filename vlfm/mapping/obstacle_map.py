# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Union

import cv2
import numpy as np
from frontier_exploration.frontier_detection import detect_frontier_waypoints
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war

from vlfm.mapping.base_map import BaseMap
from vlfm.utils.geometry_utils import extract_yaw, get_point_cloud, transform_points
from vlfm.utils.img_utils import fill_small_holes

from typing import Any, Union, Dict, Optional, List


class ObstacleMap(BaseMap):
    """Generates two maps; one representing the area that the robot has explored so far,
    and another representing the obstacles that the robot has seen so far.
    """

    # _map_dtype: np.dtype = np.dtype(bool)
    # _frontiers_px: np.ndarray = np.array([])
    # frontiers: np.ndarray = np.array([])
    # radius_padding_color: tuple = (100, 100, 100)

    def __init__(
        self,
        min_height: float,
        max_height: float,
        agent_radius: float,
        area_thresh: float = 3.0,  # square meters
        hole_area_thresh: int = 100000,  # square pixels
        size: int = 1000,
        pixels_per_meter: int = 20,
    ):
        super().__init__(size, pixels_per_meter)

        self._map_dtype: np.dtype = np.dtype(bool)
        self._frontiers_px: np.ndarray = np.array([])
        self.frontiers: np.ndarray = np.array([])
        self.radius_padding_color: tuple = (100, 100, 100)

        self.explored_area = np.zeros((size, size), dtype=bool)
        self._map = np.zeros((size, size), dtype=bool)
        self._navigable_map = np.zeros((size, size), dtype=bool)

        self._min_height = min_height
        self._max_height = max_height
        self._area_thresh_in_pixels = area_thresh * (self.pixels_per_meter**2)
        self._hole_area_thresh = hole_area_thresh
        kernel_size = self.pixels_per_meter * agent_radius * 2
        # round kernel_size to nearest odd number
        kernel_size = int(kernel_size) + (int(kernel_size) % 2 == 0)
        self._navigable_kernel = np.ones((kernel_size, kernel_size), np.uint8)


        # 爬楼梯相关
        self._strict_navigable_map = np.zeros((size, size), dtype=bool)
        self._movable_obstacle_map = np.zeros((size, size), dtype=bool)  # mainly for humans
        self._up_stair_map = np.zeros((size, size), dtype=bool)  # for upstairs
        self._down_stair_map = np.zeros((size, size), dtype=bool)  # for downstairs
        self._disabled_stair_map = np.zeros((size, size), dtype=bool)  # for disabled stairs
        self.agent_radius = agent_radius # * 1.3

        kernel_size += 2
        self._strict_navigable_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        self._has_up_stair = False # 确定有upstairs才为True
        self._has_down_stair = False # 确定有downstairs才为True（疑似也不行）
        self._done_initializing = False
        self._this_floor_explored = False

        self._up_stair_frontiers_px = np.array([])
        self._up_stair_frontiers = np.array([])
        self._down_stair_frontiers_px = np.array([])
        self._down_stair_frontiers = np.array([])

        self._up_stair_start = np.array([])
        # self._up_stair_centroid = np.array([])
        self._up_stair_end = np.array([])
        self._down_stair_start = np.array([])
        # self._down_stair_centroid = np.array([])
        self._down_stair_end = np.array([])

        self._carrot_goal_px = np.array([])
        self._explored_up_stair = False
        self._explored_down_stair = False

        self.stair_boundary = np.zeros((size, size), dtype=bool)
        self.stair_boundary_goal = np.zeros((size, size), dtype=bool)
        self._floor_num_steps = 0
        self._disabled_frontiers = set()
        self._disabled_frontiers_px =  np.array([], dtype=np.float64).reshape(0, 2) # np.array([])
        # self._temp_down_stair_map_frontiers_px = np.array([])
        # self._temp_stair_traj = np.array([])
        # self._search_down_stair = False
        self._climb_stair_paused_step = 0
        self._disable_end = False
        # self._look_for_downstair = True
        self._look_for_downstair_flag = False
        self._potential_stair_centroid_px = np.array([])
        self._potential_stair_centroid = np.array([])
        # 防止楼梯间
        self._reinitialize_flag = False
        self._tight_search_thresh = False
        self._best_frontier_selection_count = {}

        self.previous_frontiers = []  # 存储之前已经可视化过的 frontiers 的索引
        self.frontier_visualization_info = {}  # 存储每个 frontier 对应的 步数
        self._each_step_rgb = {} # 存储每一步对应的rgb, 仅供debug
        # self._each_step_rgb_hash = {} # 存储每一步对应的rgb hash
        self._each_step_rgb_phash = {} # 存储每一步对应的rgb phash
        self._finish_first_explore = False
        self._neighbor_search = False


    def reset(self) -> None:
        super().reset()

        # initialize class variable
        self._map_dtype = np.dtype(bool)
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])
        self.radius_padding_color = (100, 100, 100)

        self._navigable_map.fill(0)
        self.explored_area.fill(0)
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])

        # 爬楼梯相关
        self._strict_navigable_map.fill(0)
        self._movable_obstacle_map.fill(0) # for movable_obstacle_map

        self._up_stair_map.fill(0) # for upstairs_map
        self._down_stair_map.fill(0) # for downstairs_map
        self._disabled_stair_map.fill(0) # True for not possible for stair

        self.stair_boundary.fill(0)
        self.stair_boundary_goal.fill(0)

        self._has_up_stair = False
        self._has_down_stair = False
        self._explored_up_stair = False
        self._explored_down_stair = False
        self._done_initializing = False
        self._up_stair_frontiers_px = np.array([])
        self._up_stair_frontiers = np.array([])
        self._down_stair_frontiers_px = np.array([])
        self._down_stair_frontiers = np.array([])

        self._up_stair_start = np.array([])
        # self._up_stair_centroid = np.array([])
        self._up_stair_end = np.array([])
        self._down_stair_start = np.array([])
        # self._down_stair_centroid = np.array([])
        self._down_stair_end = np.array([])

        self._carrot_goal_px = np.array([])

        self._floor_num_steps = 0      
        self._disabled_frontiers = set()
        self._disabled_frontiers_px =  np.array([], dtype=np.float64).reshape(0, 2) # np.array([])
        # self._search_down_stair = False
        # self._temp_stair_traj = np.array([])
        self._climb_stair_paused_step = 0
        self._disable_end = False
        self._look_for_downstair_flag = False
        self._potential_stair_centroid_px = np.array([])
        self._potential_stair_centroid = np.array([])

        self._reinitialize_flag = False
        self._tight_search_thresh = False
        self._best_frontier_selection_count = {}

        self.previous_frontiers = []  # 存储之前已经可视化过的 frontiers 的索引
        self.frontier_visualization_info = {}  # 存储每个 frontier 对应的 RGB 图以及箭头标记
        self._each_step_rgb = {}
        # self._each_step_rgb_hash = {}
        self._each_step_rgb_phash = {}
        self._finish_first_explore = False
        self._neighbor_search = False



    # def update_map(
    #     self,
    #     depth: np.ndarray,
    #     tf_camera_to_episodic: np.ndarray,
    #     min_depth: float,
    #     max_depth: float,
    #     fx: float,
    #     fy: float,
    #     topdown_fov: float,
    #     stair_mask: List, # only a (480,640) mask, also for multiple stairs
    #     agent_pitch_angle: int,
    #     search_stair_over: bool,
    #     reach_stair: bool,
    #     climb_stair_flag: int,
    #     explore: bool = True,
    #     update_obstacles: bool = True,
    # ) -> None:
    #     # 1. 完成深度图像空洞修复
    #     # 2. 根据点云计算障碍地图self._map
    #     # 3. 根据障碍地图self._map计算可导航地图self._navigable_map
    #     # 4. 根据 障碍地图+可导航地图，计算explored_area(没有障碍区域)，并根据轮廓保证explored_area与机器人实时可连通
    #     # 5. 根据_navigable_map和explored_area，计算frontier

    #     """
    #     Adds all obstacles from the current view to the map. Also updates the area
    #     that the robot has explored so far.

    #     Args:
    #         depth (np.ndarray): The depth image to use for updating the object map. It
    #             is normalized to the range [0, 1] and has a shape of (height, width).

    #         tf_camera_to_episodic (np.ndarray): The transformation matrix from the
    #             camera to the episodic coordinate frame.
    #         min_depth (float): The minimum depth value (in meters) of the depth image.
    #         max_depth (float): The maximum depth value (in meters) of the depth image.
    #         fx (float): The focal length of the camera in the x direction.
    #         fy (float): The focal length of the camera in the y direction.
    #         topdown_fov (float): The field of view of the depth camera projected onto
    #             the topdown map.
    #         explore (bool): Whether to update the explored area.
    #         update_obstacles (bool): Whether to update the obstacle map.
    #     """
    #     if update_obstacles:
    #         if self._hole_area_thresh == -1:
    #             filled_depth = depth.copy()
    #             filled_depth[depth == 0] = 1.0 # 直接将所有空洞填补为1
    #         else: # 进入此条
    #             filled_depth = fill_small_holes(depth, self._hole_area_thresh) # 将原始深度图像中的小空洞填补为1
    #         scaled_depth = filled_depth * (max_depth - min_depth) + min_depth
            
    #         if(len(stair_mask)>0):
    #             if(len(stair_mask)==1):
    #                 fusion_stair_mask = stair_mask[0].astype(bool)
    #             else:
    #                 fusion_stair_mask = np.logical_or.reduce(stair_mask)
    #                 fusion_stair_mask = fusion_stair_mask.astype(bool)

    #             stair_depth = np.full_like(depth, max_depth)
    #             scaled_depth_stair = scaled_depth.copy()
    #             stair_depth[fusion_stair_mask] = scaled_depth_stair[fusion_stair_mask]

    #             stair_cloud_camera_frame = get_point_cloud(stair_depth, fusion_stair_mask, fx, fy)
    #             stair_cloud_episodic_frame = transform_points(tf_camera_to_episodic, stair_cloud_camera_frame)
    #             stair_xy_points = stair_cloud_episodic_frame[:, :2]
    #             stair_pixel_points = self._xy_to_px(stair_xy_points)
    #             # climb_stair_flag初始化为0
    #             if agent_pitch_angle >= 0 and climb_stair_flag != 2: # 机器人头平视或向上，并且没有在下楼 --> 可能是上楼的新检测楼梯 # 有可能是reverse_climb_stair
    #             # 遍历每个 stair_pixel_points 点，进行标记和清除
    #                 for x, y in stair_pixel_points:
    #                     # 在 _stair_map 上标记为确定的楼梯
    #                     if 0 <= x < self._up_stair_map.shape[1] and 0 <= y < self._up_stair_map.shape[0] and self._up_stair_map[y, x] == 0:
    #                         self._up_stair_map[y, x] = 1
    #                 self._map[self._up_stair_map == 1] = 1
    #             elif agent_pitch_angle < 0 and climb_stair_flag != 1: # 有可能是reverse_climb_stair
    #                 for x, y in stair_pixel_points:
    #                     # 在 _stair_map 上标记为确定的楼梯
    #                     if 0 <= x < self._down_stair_map.shape[1] and 0 <= y < self._down_stair_map.shape[0] and self._down_stair_map[y, x] == 0:
    #                         self._down_stair_map[y, x] = 1 
    #                 self._map[self._down_stair_map == 1] = 1 # 不可通行范围大一点，减少探索
            
    #         ## normal to look for downstair
    #         ## 反转深度，但发现对短楼梯不好使 
    #         # reach_stair初始化为False
    #         # 本质：在self._down_stair_map上标注向下的楼梯
    #         if agent_pitch_angle <= 0 and reach_stair == False: # 靠近楼梯的时候也要找，不然楼梯间的时候下楼误以为上楼了
    #             filled_depth_for_stair = fill_small_holes(depth, self._hole_area_thresh)
    #             inverted_depth_for_stair = max_depth - filled_depth_for_stair * (max_depth - min_depth)
    #             inverted_mask = inverted_depth_for_stair < 2 # 只要更远处的像素 # inverted_depth_for_stair < 2 # 3 <= true depth value < max_depth 
    #             inverted_point_cloud_camera_frame = get_point_cloud(inverted_depth_for_stair, inverted_mask, fx, fy)
    #             inverted_point_cloud_episodic_frame = transform_points(tf_camera_to_episodic, inverted_point_cloud_camera_frame)
    #             # below_ground_obstacle_cloud = filter_points_by_height_below_ground(inverted_point_cloud_episodic_frame)
    #             below_ground_obstacle_cloud_0 = filter_points_by_height_below_ground_0(inverted_point_cloud_episodic_frame) # 找到显著低于当前地面的点云
    #             below_ground_xy_points = below_ground_obstacle_cloud_0[:, :2] # below_ground_obstacle_cloud[:, :2]
    #             # 获取需要赋值的点的像素坐标
    #             below_ground_pixel_points = self._xy_to_px(below_ground_xy_points)
    #             self._down_stair_map[below_ground_pixel_points[:, 1], below_ground_pixel_points[:, 0]] = 1
            
    #         # 不爬楼梯的时候标注
    #         # search_stair_over默认为True
    #         if search_stair_over == True: # reach_stair == False:
    #             # 与原始vlfm相同，更新_map障碍区域
    #             mask = scaled_depth < max_depth
    #             point_cloud_camera_frame = get_point_cloud(scaled_depth, mask, fx, fy)
    #             point_cloud_episodic_frame = transform_points(tf_camera_to_episodic, point_cloud_camera_frame)
    #             obstacle_cloud = filter_points_by_height(point_cloud_episodic_frame, self._min_height, self._max_height)

    #             xy_points = obstacle_cloud[:, :2]
    #             pixel_points = self._xy_to_px(xy_points)

    #             self._map[pixel_points[:, 1], pixel_points[:, 0]] = 1

    #         self._up_stair_map = self._up_stair_map & (~self._disabled_stair_map) # "确定的up_stair"
    #         self._down_stair_map = self._down_stair_map & (~self._disabled_stair_map) # "确定的down_stair"
            
    #         stair_dilated_mask = (self._up_stair_map == 1) | (self._down_stair_map == 1)
    #         # stair_dilated_mask = self._up_stair_map == 1 | self._down_stair_map == 1 # ((self._map == 1) & (self._up_stair_map == 1)) | ((self._map == 1) & (self._down_stair_map == 1))
    #         self._map[stair_dilated_mask] = 0 # 确定的up_stair和down_stair在_map上均标记为0

    #         dilated_map = cv2.dilate(
    #             self._map.astype(np.uint8),
    #             self._navigable_kernel,
    #             iterations=1
    #         )
    #         dilated_map[stair_dilated_mask] = 1 # 楼梯不参与膨胀
    #         self._map[stair_dilated_mask] = 1 # 又加回来了
    #         # 不让楼梯膨胀
    #         self._navigable_map = 1 - dilated_map.astype(bool) # 计算可导航区域地图（0为被占用）

    #         strict_dilated_map = cv2.dilate(
    #             self._map.astype(np.uint8),
    #             self._strict_navigable_kernel,
    #             iterations=1
    #         )
    #         self._strict_navigable_map = 1 - strict_dilated_map.astype(bool) # 楼梯膨胀后对应的可导航区域（对应1的面积更小）

    #     if not explore:
    #         return

    #     # Update the explored area
    #     agent_xy_location = tf_camera_to_episodic[:2, 3]
    #     agent_pixel_location = self._xy_to_px(agent_xy_location.reshape(1, 2))[0] # 得到机器人在map栅格上的像素
    #     new_explored_area = reveal_fog_of_war(
    #         top_down_map=self._navigable_map.astype(np.uint8),
    #         current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),
    #         current_point=agent_pixel_location[::-1],
    #         current_angle=-extract_yaw(tf_camera_to_episodic),
    #         fov=np.rad2deg(topdown_fov),
    #         max_line_len=max_depth * self.pixels_per_meter,
    #     ) # 计算当前视角下新的已探索区域
    #     new_explored_area = cv2.dilate(new_explored_area, np.ones((3, 3), np.uint8), iterations=1)
        
    #     self.explored_area[new_explored_area > 0] = 1 # 累计已探索区域
    #     self.explored_area[self._navigable_map == 0] = 0 # 去除障碍区域 （只保留已探索的空闲区域为1）
        
        
        
    #     contours, _ = cv2.findContours(
    #         self.explored_area.astype(np.uint8),
    #         cv2.RETR_EXTERNAL,
    #         cv2.CHAIN_APPROX_SIMPLE,
    #     )

    #     #  保证已探索区域始终与机器人连通
    #     if len(contours) > 1:
    #         min_dist = np.inf
    #         best_idx = 0
    #         for idx, cnt in enumerate(contours):
    #             dist = cv2.pointPolygonTest(cnt, tuple([int(i) for i in agent_pixel_location]), True)
    #             if dist >= 0:
    #                 best_idx = idx
    #                 break
    #             elif abs(dist) < min_dist:
    #                 min_dist = abs(dist)
    #                 best_idx = idx
    #         new_area = np.zeros_like(self.explored_area, dtype=np.uint8)
    #         cv2.drawContours(new_area, contours, best_idx, 1, -1)  # type: ignore
    #         self.explored_area = new_area.astype(bool) #  保证已探索区域始终与机器人连通

    #     # Compute frontier locations
    #     self._frontiers_px = self._get_frontiers()
    #     if len(self._frontiers_px) == 0:
    #         self.frontiers = np.array([])
    #     else:
    #         self.frontiers = self._px_to_xy(self._frontiers_px) # 得到所有世界坐标系下的frontier

    #     # Compute stair frontier
            
    #     # 本质：处理累计多帧融合后的_down_stair_map和_up_stair_map
    #     if np.sum(self._down_stair_map == 1) > 20:
    #         # 填充楼梯区域内细小的空洞，使楼梯更完整
    #         self._down_stair_map = cv2.morphologyEx(self._down_stair_map.astype(np.uint8) , cv2.MORPH_CLOSE, self._navigable_kernel,) # 一条细线做先膨胀后腐蚀操作

    #         # 应该剔除小的，不连通的区域
    #         # 保留面积大的，剔除面积小的
    #         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self._down_stair_map, connectivity=8)
    #         min_area_threshold = 10  # 设定最小面积阈值为 10（即小于 10 个像素的区域被视为小连通域）
    #         filtered_map = np.zeros_like(self._down_stair_map)
    #         max_area = 0
    #         max_label = 1
    #         for i in range(1, num_labels):  # 从1开始，0是背景
    #             area = stats[i, cv2.CC_STAT_AREA]
    #             if area >= min_area_threshold:
    #                 filtered_map[labels == i] = 1  # 保留面积大于阈值的区域
    #                 # 更新最大面积区域的标签
    #                 if area > max_area:
    #                     max_area = area
    #                     max_label = i

    #         self._down_stair_map = filtered_map
    #         self._down_stair_frontiers_px = np.array([[centroids[max_label][0], centroids[max_label][1]]])

    #         self._down_stair_frontiers = self._px_to_xy(self._down_stair_frontiers_px) # 将最大的连通区域的“几何中心像素坐标”作为stair_frontier
    #         self._has_down_stair = True
    #         self._look_for_downstair_flag = False
    #         self._potential_stair_centroid_px = np.array([])
    #         self._potential_stair_centroid = np.array([])
    #     else:
    #         # self._down_stair_frontiers_px = np.array([])  # 没有楼梯区域时
    #         # self._has_down_stair = False
    #         if np.sum(self._down_stair_map == 1) > 0:
    #             # self._down_stair_map = cv2.morphologyEx(self._down_stair_map.astype(np.uint8) , cv2.MORPH_CLOSE, self._navigable_kernel,) # 一条细线做先膨胀后腐蚀操作
    #             num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self._down_stair_map.astype(np.uint8), connectivity=8)
    #             max_area = 0
    #             max_label = 1
    #             # 逐个找最大区域的质心，直到有向下楼梯
    #             for i in range(1, num_labels):  # 从1开始，0是背景
    #                 area = stats[i, cv2.CC_STAT_AREA]
    #                 if area > max_area:
    #                     max_area = area
    #                     max_label = i
    #             self._potential_stair_centroid_px = np.array([[centroids[max_label][0], centroids[max_label][1]]])
    #             self._potential_stair_centroid = self._px_to_xy(self._potential_stair_centroid_px)
    #             # self._down_stair_map = filtered_map
    #             # self._down_stair_frontiers_px = np.array([[centroids[max_label][0], centroids[max_label][1]]])
    #             # self._down_stair_frontiers = self._px_to_xy(self._down_stair_frontiers_px)
    #             # self._has_down_stair = True
    #             # self._look_for_downstair_flag = False

    #     if np.sum(self._up_stair_map == 1) > 20:
    #         self._up_stair_map = cv2.morphologyEx(self._up_stair_map.astype(np.uint8) , cv2.MORPH_CLOSE, self._navigable_kernel,) # 一条细线做先膨胀后腐蚀操作

    #         # 应该剔除小的，不连通的区域
    #         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self._up_stair_map, connectivity=8)
    #         min_area_threshold = 10  # 设定最小面积阈值为 10（即小于 10 个像素的区域被视为小连通域）
    #         filtered_map = np.zeros_like(self._up_stair_map)
    #         max_area = 0
    #         max_label = 1
    #         for i in range(1, num_labels):  # 从1开始，0是背景
    #             area = stats[i, cv2.CC_STAT_AREA]
    #             if area >= min_area_threshold:
    #                 filtered_map[labels == i] = 1  # 保留面积大于阈值的区域
    #                 # 更新最大面积区域的标签
    #                 if area > max_area:
    #                     max_area = area
    #                     max_label = i

    #         self._up_stair_map = filtered_map
    #         self._up_stair_frontiers_px = np.array([[centroids[max_label][0], centroids[max_label][1]]])

    #         self._up_stair_frontiers = self._px_to_xy(self._up_stair_frontiers_px)
    #         self._has_up_stair = True
    #     else:
    #         self._up_stair_frontiers_px = np.array([])  # 没有楼梯区域时
    #         self._has_up_stair = False

    #     if len(self._down_stair_frontiers) == 0 and np.sum(self._down_stair_map) > 0:
    #         # 标识，提示agent往这边导航
    #         self._look_for_downstair_flag = True # 若有疑似的downstairs，则_look_for_downstair_flag为True；若确实有较大面积的downstairs or 不存在downstairs，则_look_for_downstair_flag为False

    def update_map(
        self,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        topdown_fov: float,
        stair_mask: List, # only a (480,640) mask, also for multiple stairs
        agent_pitch_angle: int,
        search_stair_over: bool,
        reach_stair: bool,
        climb_stair_flag: int,
        explore: bool = True,
        update_obstacles: bool = True,
    ) -> None:
        # 1. 完成深度图像空洞修复
        # 2. 根据点云计算障碍地图self._map
        # 3. 根据障碍地图self._map计算可导航地图self._navigable_map
        # 4. 根据 障碍地图+可导航地图，计算explored_area(没有障碍区域)，并根据轮廓保证explored_area与机器人实时可连通
        # 5. 根据_navigable_map和explored_area，计算frontier

        """
        Adds all obstacles from the current view to the map. Also updates the area
        that the robot has explored so far.

        Args:
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).

            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.
            topdown_fov (float): The field of view of the depth camera projected onto
                the topdown map.
            explore (bool): Whether to update the explored area.
            update_obstacles (bool): Whether to update the obstacle map.
        """
        if update_obstacles:
            if self._hole_area_thresh == -1:
                filled_depth = depth.copy()
                filled_depth[depth == 0] = 1.0 # 直接将所有空洞填补为1
            else: # 进入此条
                filled_depth = fill_small_holes(depth, self._hole_area_thresh) # 将原始深度图像中的小空洞填补为1
            scaled_depth = filled_depth * (max_depth - min_depth) + min_depth
            
            # =====> stair_map <=====
            if(len(stair_mask)>0):
                if(len(stair_mask)==1):
                    fusion_stair_mask = stair_mask[0].astype(bool)
                else:
                    fusion_stair_mask = np.logical_or.reduce(stair_mask)
                    fusion_stair_mask = fusion_stair_mask.astype(bool)

                stair_depth = np.full_like(depth, max_depth)
                scaled_depth_stair = scaled_depth.copy()
                stair_depth[fusion_stair_mask] = scaled_depth_stair[fusion_stair_mask]

                stair_cloud_camera_frame = get_point_cloud(stair_depth, fusion_stair_mask, fx, fy)
                stair_cloud_episodic_frame = transform_points(tf_camera_to_episodic, stair_cloud_camera_frame)
                stair_xy_points = stair_cloud_episodic_frame[:, :2]
                stair_pixel_points = self._xy_to_px(stair_xy_points)
                # climb_stair_flag初始化为0
                if agent_pitch_angle >= 0 and climb_stair_flag != 2: # 机器人头平视或向上，并且没有在下楼 --> 可能是上楼的新检测楼梯 # 有可能是reverse_climb_stair
                # 遍历每个 stair_pixel_points 点，进行标记和清除
                    for x, y in stair_pixel_points:
                        # 在 _stair_map 上标记为确定的楼梯
                        if 0 <= x < self._up_stair_map.shape[1] and 0 <= y < self._up_stair_map.shape[0] and self._up_stair_map[y, x] == 0:
                            self._up_stair_map[y, x] = 1
                    # self._map[self._up_stair_map == 1] = 1
                elif agent_pitch_angle < 0 and climb_stair_flag != 1: # 有可能是reverse_climb_stair
                    for x, y in stair_pixel_points:
                        # 在 _stair_map 上标记为确定的楼梯
                        if 0 <= x < self._down_stair_map.shape[1] and 0 <= y < self._down_stair_map.shape[0] and self._down_stair_map[y, x] == 0:
                            self._down_stair_map[y, x] = 1 
                    # self._map[self._down_stair_map == 1] = 1 # 不可通行范围大一点，减少探索
            
            ## normal to look for downstair
            ## 反转深度，但发现对短楼梯不好使 
            # reach_stair初始化为False
            # 本质：在self._down_stair_map上标注向下的楼梯
            if agent_pitch_angle <= 0 and reach_stair == False: # 靠近楼梯的时候也要找，不然楼梯间的时候下楼误以为上楼了
                filled_depth_for_stair = fill_small_holes(depth, self._hole_area_thresh)
                inverted_depth_for_stair = max_depth - filled_depth_for_stair * (max_depth - min_depth)
                inverted_mask = inverted_depth_for_stair < 2 # 只要更远处的像素 # inverted_depth_for_stair < 2 # 3 <= true depth value < max_depth 
                inverted_point_cloud_camera_frame = get_point_cloud(inverted_depth_for_stair, inverted_mask, fx, fy)
                inverted_point_cloud_episodic_frame = transform_points(tf_camera_to_episodic, inverted_point_cloud_camera_frame)
                # below_ground_obstacle_cloud = filter_points_by_height_below_ground(inverted_point_cloud_episodic_frame)
                below_ground_obstacle_cloud_0 = filter_points_by_height_below_ground_0(inverted_point_cloud_episodic_frame) # 找到显著低于当前地面的点云
                below_ground_xy_points = below_ground_obstacle_cloud_0[:, :2] # below_ground_obstacle_cloud[:, :2]
                # 获取需要赋值的点的像素坐标
                below_ground_pixel_points = self._xy_to_px(below_ground_xy_points)
                self._down_stair_map[below_ground_pixel_points[:, 1], below_ground_pixel_points[:, 0]] = 1
            
            # # 不爬楼梯的时候标注
            # # search_stair_over默认为True
            # if search_stair_over == True: # reach_stair == False:
            #     # 与原始vlfm相同，更新_map障碍区域
            #     mask = scaled_depth < max_depth
            #     point_cloud_camera_frame = get_point_cloud(scaled_depth, mask, fx, fy)
            #     point_cloud_episodic_frame = transform_points(tf_camera_to_episodic, point_cloud_camera_frame)
            #     obstacle_cloud = filter_points_by_height(point_cloud_episodic_frame, self._min_height, self._max_height)

            #     xy_points = obstacle_cloud[:, :2]
            #     pixel_points = self._xy_to_px(xy_points)

            #     self._map[pixel_points[:, 1], pixel_points[:, 0]] = 1

            self._up_stair_map = self._up_stair_map & (~self._disabled_stair_map) # "确定的up_stair"
            self._down_stair_map = self._down_stair_map & (~self._disabled_stair_map) # "确定的down_stair"
            
            stair_dilated_mask = (self._up_stair_map == 1) | (self._down_stair_map == 1)
            # stair_dilated_mask = self._up_stair_map == 1 | self._down_stair_map == 1 # ((self._map == 1) & (self._up_stair_map == 1)) | ((self._map == 1) & (self._down_stair_map == 1))
            # self._map[stair_dilated_mask] = 0 # 确定的up_stair和down_stair在_map上均标记为0

            # =====> stair_map <=====

            # 不爬楼梯的时候标注
            # search_stair_over默认为True
            if search_stair_over == True: # reach_stair == False:
                # 与原始vlfm相同，更新_map障碍区域
                mask = scaled_depth < max_depth
                point_cloud_camera_frame = get_point_cloud(scaled_depth, mask, fx, fy)
                point_cloud_episodic_frame = transform_points(tf_camera_to_episodic, point_cloud_camera_frame)
                obstacle_cloud = filter_points_by_height(point_cloud_episodic_frame, self._min_height, self._max_height)

                xy_points = obstacle_cloud[:, :2]
                pixel_points = self._xy_to_px(xy_points)
                self._map[pixel_points[:, 1], pixel_points[:, 0]] = 1

            '''
            dilated_map = cv2.dilate(
                self._map.astype(np.uint8),
                self._navigable_kernel,
                iterations=1
            )
            dilated_map[stair_dilated_mask] = 1 # 楼梯不参与膨胀
            self._map[stair_dilated_mask] = 1 # 又加回来了
            # 不让楼梯膨胀
            self._navigable_map = 1 - dilated_map.astype(bool) # 计算可导航区域地图（0为被占用）

            strict_dilated_map = cv2.dilate(
                self._map.astype(np.uint8),
                self._strict_navigable_kernel,
                iterations=1
            )
            self._strict_navigable_map = 1 - strict_dilated_map.astype(bool) # 楼梯膨胀后对应的可导航区域（对应1的面积更小）
            '''

            self._navigable_map = 1 - cv2.dilate(
                self._map.astype(np.uint8),
                self._navigable_kernel,
                iterations=1,
            ).astype(bool) # 计算可导航区域地图（0为被占用）

        if not explore:
            return

        # Update the explored area
        agent_xy_location = tf_camera_to_episodic[:2, 3]
        agent_pixel_location = self._xy_to_px(agent_xy_location.reshape(1, 2))[0] # 得到机器人在map栅格上的像素
        new_explored_area = reveal_fog_of_war(
            top_down_map=self._navigable_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),
            current_point=agent_pixel_location[::-1],
            current_angle=-extract_yaw(tf_camera_to_episodic),
            fov=np.rad2deg(topdown_fov),
            max_line_len=max_depth * self.pixels_per_meter,
        ) # 计算当前视角下新的已探索区域
        new_explored_area = cv2.dilate(new_explored_area, np.ones((3, 3), np.uint8), iterations=1)
        
        self.explored_area[new_explored_area > 0] = 1 # 累计已探索区域
        self.explored_area[self._navigable_map == 0] = 0 # 去除障碍区域 （只保留已探索的空闲区域为1）
        
        
        
        contours, _ = cv2.findContours(
            self.explored_area.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        #  保证已探索区域始终与机器人连通
        if len(contours) > 1:
            min_dist = np.inf
            best_idx = 0
            for idx, cnt in enumerate(contours):
                dist = cv2.pointPolygonTest(cnt, tuple([int(i) for i in agent_pixel_location]), True)
                if dist >= 0:
                    best_idx = idx
                    break
                elif abs(dist) < min_dist:
                    min_dist = abs(dist)
                    best_idx = idx
            new_area = np.zeros_like(self.explored_area, dtype=np.uint8)
            cv2.drawContours(new_area, contours, best_idx, 1, -1)  # type: ignore
            self.explored_area = new_area.astype(bool) #  保证已探索区域始终与机器人连通

        # Compute frontier locations
        self._frontiers_px = self._get_frontiers()
        if len(self._frontiers_px) == 0:
            self.frontiers = np.array([])
        else:
            self.frontiers = self._px_to_xy(self._frontiers_px) # 得到所有世界坐标系下的frontier

        # Compute stair frontier
            
        # 本质：处理累计多帧融合后的_down_stair_map和_up_stair_map
        if np.sum(self._down_stair_map == 1) > 20:
            # 填充楼梯区域内细小的空洞，使楼梯更完整
            self._down_stair_map = cv2.morphologyEx(self._down_stair_map.astype(np.uint8) , cv2.MORPH_CLOSE, self._navigable_kernel,) # 一条细线做先膨胀后腐蚀操作

            # 应该剔除小的，不连通的区域
            # 保留面积大的，剔除面积小的
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self._down_stair_map, connectivity=8)
            min_area_threshold = 10  # 设定最小面积阈值为 10（即小于 10 个像素的区域被视为小连通域）
            filtered_map = np.zeros_like(self._down_stair_map)
            max_area = 0
            max_label = 1
            for i in range(1, num_labels):  # 从1开始，0是背景
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_area_threshold:
                    filtered_map[labels == i] = 1  # 保留面积大于阈值的区域
                    # 更新最大面积区域的标签
                    if area > max_area:
                        max_area = area
                        max_label = i

            self._down_stair_map = filtered_map
            self._down_stair_frontiers_px = np.array([[centroids[max_label][0], centroids[max_label][1]]])

            self._down_stair_frontiers = self._px_to_xy(self._down_stair_frontiers_px) # 将最大的连通区域的“几何中心像素坐标”作为stair_frontier
            self._has_down_stair = True
            self._look_for_downstair_flag = False
            self._potential_stair_centroid_px = np.array([])
            self._potential_stair_centroid = np.array([])
        else:
            # self._down_stair_frontiers_px = np.array([])  # 没有楼梯区域时
            # self._has_down_stair = False
            if np.sum(self._down_stair_map == 1) > 0:
                # self._down_stair_map = cv2.morphologyEx(self._down_stair_map.astype(np.uint8) , cv2.MORPH_CLOSE, self._navigable_kernel,) # 一条细线做先膨胀后腐蚀操作
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self._down_stair_map.astype(np.uint8), connectivity=8)
                max_area = 0
                max_label = 1
                # 逐个找最大区域的质心，直到有向下楼梯
                for i in range(1, num_labels):  # 从1开始，0是背景
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area > max_area:
                        max_area = area
                        max_label = i
                self._potential_stair_centroid_px = np.array([[centroids[max_label][0], centroids[max_label][1]]])
                self._potential_stair_centroid = self._px_to_xy(self._potential_stair_centroid_px)
                # self._down_stair_map = filtered_map
                # self._down_stair_frontiers_px = np.array([[centroids[max_label][0], centroids[max_label][1]]])
                # self._down_stair_frontiers = self._px_to_xy(self._down_stair_frontiers_px)
                # self._has_down_stair = True
                # self._look_for_downstair_flag = False

        if np.sum(self._up_stair_map == 1) > 20:
            self._up_stair_map = cv2.morphologyEx(self._up_stair_map.astype(np.uint8) , cv2.MORPH_CLOSE, self._navigable_kernel,) # 一条细线做先膨胀后腐蚀操作

            # 应该剔除小的，不连通的区域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self._up_stair_map, connectivity=8)
            min_area_threshold = 10  # 设定最小面积阈值为 10（即小于 10 个像素的区域被视为小连通域）
            filtered_map = np.zeros_like(self._up_stair_map)
            max_area = 0
            max_label = 1
            for i in range(1, num_labels):  # 从1开始，0是背景
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_area_threshold:
                    filtered_map[labels == i] = 1  # 保留面积大于阈值的区域
                    # 更新最大面积区域的标签
                    if area > max_area:
                        max_area = area
                        max_label = i

            self._up_stair_map = filtered_map
            self._up_stair_frontiers_px = np.array([[centroids[max_label][0], centroids[max_label][1]]])

            self._up_stair_frontiers = self._px_to_xy(self._up_stair_frontiers_px)
            self._has_up_stair = True
        else:
            self._up_stair_frontiers_px = np.array([])  # 没有楼梯区域时
            self._has_up_stair = False

        if len(self._down_stair_frontiers) == 0 and np.sum(self._down_stair_map) > 0:
            # 标识，提示agent往这边导航
            self._look_for_downstair_flag = True # 若有疑似的downstairs，则_look_for_downstair_flag为True；若确实有较大面积的downstairs or 不存在downstairs，则_look_for_downstair_flag为False




    '''
    def _get_frontiers(self) -> np.ndarray:
        """Returns the frontiers of the map."""
        # Dilate the explored area slightly to prevent small gaps between the explored
        # area and the unnavigable area from being detected as frontiers.
        explored_area = cv2.dilate(
            self.explored_area.astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        frontiers = detect_frontier_waypoints(
            self._navigable_map.astype(np.uint8),
            explored_area,
            self._area_thresh_in_pixels,
        )
        return frontiers
    '''

    def _get_frontiers(self) -> np.ndarray:
        """Returns the frontiers of the map."""
        # Dilate the explored area slightly to prevent small gaps between the explored
        # area and the unnavigable area from being detected as frontiers.
        explored_area = cv2.dilate(
            self.explored_area.astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        # 如果有楼梯间，那么需要更仔细地搜索到另一个楼梯
        # 或者探索完没发现楼梯
        if self._tight_search_thresh: # _tight_search_thresh初始化为False
            frontiers = detect_frontier_waypoints(
                self._navigable_map.astype(np.uint8),
                explored_area,
                -1,
            ) # 更加仔细的搜索frontier，不会滤除
        else: # 原版代码
            frontiers = detect_frontier_waypoints(
                self._navigable_map.astype(np.uint8),
                explored_area,
                self._area_thresh_in_pixels,
            )
        return frontiers




    def visualize(self) -> np.ndarray:
        """Visualizes the map."""
        vis_img = np.ones((*self._map.shape[:2], 3), dtype=np.uint8) * 255 # 纯白色的RGB图像
        # Draw explored area in light green
        vis_img[self.explored_area == 1] = (200, 255, 200)
        # Draw unnavigable areas in gray
        vis_img[self._navigable_map == 0] = self.radius_padding_color
        # Draw obstacles in black
        vis_img[self._map == 1] = (0, 0, 0)
        # Draw frontiers in blue (200, 0, 0)
        for frontier in self._frontiers_px:
            cv2.circle(vis_img, tuple([int(i) for i in frontier]), 5, (200, 0, 0), 2)

        vis_img = cv2.flip(vis_img, 0)

        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                vis_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

        return vis_img


def filter_points_by_height(points: np.ndarray, min_height: float, max_height: float) -> np.ndarray:
    return points[(points[:, 2] >= min_height) & (points[:, 2] <= max_height)]

def filter_points_by_height_below_ground_0(points: np.ndarray) -> np.ndarray:
    data = points[(points[:, 2] < 0)] # 0.2 是机器人的max_climb
    return data
