# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Dict, Union

import cv2
import numpy as np
import open3d as o3d

from vlfm.utils.geometry_utils import (
    extract_yaw,
    get_point_cloud,
    transform_points,
    within_fov_cone,
)

from vlfm.utils.img_utils import (
    monochannel_to_inferno_rgb,
    pixel_value_within_radius,
    place_img_in_img,
    rotate_image,
)

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# zhou_percept
from scipy.spatial import cKDTree # 导入 cKDTree
import copy

# zhou_percept
class HabitatAction:
    """
        static class for habitat action process.
        Attributes
        ----------
        count_steps: all step num.
        front_steps: front step num.
        walk_path_meter: The path meter of robot walking.
        this_episode_short_dis: init min_distance to goal.
    """
    name_val = 0
    detect_name_val = 0

    has_vis_object_goal = 0
    all_object_goal_gps_pos = None
    real_sub_goal_array = None
    

    @staticmethod
    def reset():
        """
            Reset the static attributes.
            :param habitat_env
        """
        HabitatAction.name_val = 0
        HabitatAction.detect_name_val = 0

        HabitatAction.has_vis_object_goal = 0
        HabitatAction.all_object_goal_gps_pos = None
        HabitatAction.real_sub_goal_array = None

# zhou_percept
class ObjectPCDetect:
    def __init__(self, object_name, confidence_score, pc):
        self.object_name = object_name
        self.confidence_score = confidence_score
        self.pc = pc
        self.pc_num = None
        self.is_check_now = False
        self.name = str(HabitatAction.detect_name_val) # 在graph update时赋值, str
        HabitatAction.detect_name_val += 1

    def __eq__(self, other):
        if isinstance(other, ObjectPCDetect):
            return self.name == other.name
        return False

    def __hash__(self):
        # 返回self.value的哈希值
        return hash(self.name)

# zhou_percept
class ObjectPCCluster:
    def __init__(self):
        self.object_dict_per_cluster = {} # label: ObjectPCDetect
        self.all_pc = None
        self.name = str(HabitatAction.name_val) # 在graph update时赋值, str
        HabitatAction.name_val += 1

    def __eq__(self, other):
        if isinstance(other, ObjectPCCluster):
            return self.name == other.name
        return False

    def __hash__(self):
        # 返回self.value的哈希值
        return hash(self.name)

    def get_best_matched_class(self): # 得到当前cluster中，“点云数*置信分数”最大的类别
        max_score = 0
        max_class = None
        for temp_class in self.object_dict_per_cluster:
            temp_score = self.object_dict_per_cluster[temp_class].pc_num*self.object_dict_per_cluster[temp_class].confidence_score
            if(temp_score>max_score):
                max_score = temp_score
                max_class = temp_class
        return max_class

    def get_intersect_num_numpy_pro(self, test_pc: np.ndarray, scale_factor: int = 20):
        """
        计算重合点 (纯NumPy高级优化版)。
        
        此版本将二维栅格坐标“压平”为一维视图，
        然后使用NumPy内置的 np.unique 和 np.isin 函数来查找交集，
        完全避免了Python循环，在处理大规模数据时速度最快。
        """
        if self.all_pc.size == 0 or test_pc.size == 0:
            return 0

        # 步骤 1: 向量化转换栅格坐标
        grid_cells1 = (self.all_pc[:, :2] * scale_factor).astype(np.int32)
        grid_cells2 = (test_pc[:, :2] * scale_factor).astype(np.int32)
        
        # 步骤 2: 将二维坐标数组转换为一维的结构化数组（void view）
        # 这是实现高效唯一化和比较的关键技巧
        # 我们将每行 (x, y) 视为一个单独的、不可分割的字节块
        c1_view = np.ascontiguousarray(grid_cells1).view(
            np.dtype((np.void, grid_cells1.dtype.itemsize * grid_cells1.shape[1]))
        )
        c2_view = np.ascontiguousarray(grid_cells2).view(
            np.dtype((np.void, grid_cells2.dtype.itemsize * grid_cells2.shape[1]))
        )

        # 步骤 3: 找到第一个点云中的唯一栅格位置
        unique_cells1 = np.unique(c1_view)
        
        # 步骤 4: 使用 np.isin 检查第二个点云中有多少点落在这些唯一位置上
        # np.isin 是一个高度优化的向量化查找函数
        is_overlapping = np.isin(c2_view, unique_cells1)
        
        # 步骤 5: 计算总数
        return np.sum(is_overlapping)

# zhou_percept
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

# zhou_percept
def check_points_in_fov(
    global_point_cloud: np.ndarray,
    tf_camera_to_episodic: np.ndarray,
    z_near: float = 0.0,
    z_far: float = 5.0,
    x_limit_ratio: float = 0.84,
    # y_limit_ratio: float = 0.6, # Docstring中提到但未使用，如果需要可以加入
) -> bool:
    """
    高效检查全局点云中是否有任何点在机器人的FOV内 (优化版)。

    优化点:
    1. 避免了通过布尔索引创建中间点云数组，减少了内存分配和数据复制。
    2. 将所有筛选条件(mask)直接在完整的局部点云上计算并合并。
    3. 使用 np.any() 来快速检查是否存在任何满足条件的点，一旦找到立即返回，无需完整计算。
    """

    # 1. 计算逆矩阵 (这部分开销很小，保持不变)
    T_local_from_global = np.linalg.inv(tf_camera_to_episodic)

    # 2. 将全局点云转换到机器人局部坐标系
    local_points = transform_points(T_local_from_global, global_point_cloud)

    # 3. 在一个表达式中计算所有筛选条件
    # 局部坐标系: +X向前, +Y向左, +Z向上
    
    # 条件1: 深度范围 (z_near < X < z_far)
    depth_mask = (local_points[:, 0] > z_near) & (local_points[:, 0] < z_far)
    
    # 条件2: 水平视野范围 (|Y/X| < ratio  =>  |Y| < X * ratio)
    # 注意: 我们只在深度符合条件的点上评估更复杂的条件，以利用逻辑与的优势
    # 虽然NumPy的'&'不会短路，但将最简单的检查放在前面是良好实践。
    horizontal_mask = np.abs(local_points[:, 1]) < local_points[:, 0] * x_limit_ratio
    
    # (可选) 条件3: 垂直视野范围
    # vertical_mask = np.abs(local_points[:, 2]) < local_points[:, 0] * y_limit_ratio

    # 合并所有掩码
    final_mask = depth_mask & horizontal_mask # & vertical_mask
    
    # 4. 使用 np.any() 直接判断结果
    is_in_fov = np.any(final_mask)
    
    return is_in_fov

# zhou_percept
def find_nearest_in_radius(
    point_cloud_A: np.ndarray, 
    point_cloud_B: np.ndarray, 
    radius: float
):
    """
    高效地为点云B中的每个点，在点云A中找到指定半径内的最近邻点 (优化版)。

    优化点:
    1. 使用 cKDTree 替代 KDTree，前者是C语言实现，速度更快。
    2. 完全移除了Python for循环，使用NumPy的向量化布尔掩码来处理查询结果。
       这使得筛选过程在NumPy的C底层执行，速度极快。
    """

    # 如果任一点云为空，直接返回空数组
    if point_cloud_A.shape[0] == 0 or point_cloud_B.shape[0] == 0:
        return np.array([])
        
    # 1. 为点云A构建cKDTree (通常比KDTree更快)
    tree_A = cKDTree(point_cloud_A)
    M = point_cloud_A.shape[0]

    # 2. 查询树 (这部分保持不变，它已经很快了)
    distances, indices = tree_A.query(
        point_cloud_B, 
        k=1, 
        distance_upper_bound=radius
    )

    # 3. 向量化处理结果，完全替代for循环
    # 创建一个布尔掩码，当索引不等于M时为True (表示找到了邻居)
    found_mask = (indices != M)
    
    # 直接使用掩码从point_cloud_B中选取所有找到了邻居的点
    results = point_cloud_B[found_mask]
    
    return results

class ObjectPointCloudMap:
    clouds: Dict[str, np.ndarray] = {}
    use_dbscan: bool = True
    _all_object_cluster_ls = [] # zhou_percept

    def __init__(self, erosion_size: float, size: int = 1000) -> None:
        self._erosion_size = erosion_size
        self.last_target_coord: Union[np.ndarray, None] = None

        # =====> new_add <=====
        self.semantic_map = np.zeros((size, size), np.uint8)
        self._episode_pixel_origin = np.array([size // 2, size // 2])
        self.pixels_per_meter = 20

        self.last_target_cluster = None # zhou_percept
        self.voxel_size = 0.02

    def reset(self, size: int = 1000) -> None:
        self.clouds = {}
        self.last_target_coord = None

        # =====> new_add <=====
        self.semantic_map = np.zeros((size, size), np.uint8)

        # zhou_percept
        self._all_object_cluster_ls = []
        self.last_target_coord = None
        self.last_target_cluster = None
        HabitatAction.reset()

    # def has_object(self, target_class: str) -> bool:
    #     return target_class in self.clouds and len(self.clouds[target_class]) > 0

    # zhou_percept
    def has_object(self, target_class: str) -> bool:
        for temp_cluster in self._all_object_cluster_ls:
            if(target_class in temp_cluster.object_dict_per_cluster):
                if(len(temp_cluster.object_dict_per_cluster[target_class].pc)>0):
                    return True
        return False

    '''
    def update_map(
        self,
        object_name: str,
        depth_img: np.ndarray,
        object_mask: np.ndarray, # 当前rgb中的每个检测mask
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> None: # 将当前rgb观测中，当前detect_mask对应的object_name对应的点云累计到self.clouds中
        """Updates the object map with the latest information from the agent."""
        local_cloud = self._extract_object_cloud(depth_img, object_mask, min_depth, max_depth, fx, fy) # 得到每个object_detect_mask对应的点云
        if len(local_cloud) == 0:
            return

        # For second-class, bad detections that are too offset or out of range, we
        # assign a random number to the last column of its point cloud that can later
        # be used to identify which points came from the same detection.
        if too_offset(object_mask): # 检查mask对应的bbox是否太偏
            within_range = np.ones_like(local_cloud[:, 0]) * np.random.rand() # (N, )的随机数序列
        else:
            # Mark all points of local_cloud whose distance from the camera is too far
            # as being out of range
            within_range = (local_cloud[:, 0] <= max_depth * 0.95) * 1.0  # 5% margin # (N, )的bool序列
            # All values of 1 in within_range will be considered within range, and all
            # values of 0 will be considered out of range; these 0s need to be
            # assigned with a random number so that they can be identified later.
            within_range = within_range.astype(np.float32) # (N, )的float序列
            within_range[within_range == 0] = np.random.rand() # 过远的点云分配为随机数，正常的点云为1
        global_cloud = transform_points(tf_camera_to_episodic, local_cloud) # 得到episodic_frame坐标系下的点云
        global_cloud = np.concatenate((global_cloud, within_range[:, None]), axis=1) # (N, 4)

        curr_position = tf_camera_to_episodic[:3, 3]
        closest_point = self._get_closest_point(global_cloud, curr_position) # 返回距离机器人最近的点云
        dist = np.linalg.norm(closest_point[:3] - curr_position)
        if dist < 1.0: # 过滤掉太近的object
            # Object is too close to trust as a valid object
            return

        # 将object_name对应的点云累计到self.clouds中
        if object_name in self.clouds:
            self.clouds[object_name] = np.concatenate((self.clouds[object_name], global_cloud), axis=0)
        else:
            self.clouds[object_name] = global_cloud
    '''

    # zhou_percept
    def update_map(
        self,
        multiclass_cloud,
        confidence_score,
        object_name: str,
        depth_img: np.ndarray,
        object_mask: np.ndarray, # 当前rgb中的每个检测mask
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> None: # 将当前rgb观测中，当前detect_mask对应的object_name对应的点云累计到self.clouds中
        """Updates the object map with the latest information from the agent."""
        local_cloud = self._extract_object_cloud(depth_img, object_mask, min_depth, max_depth, fx, fy) # 得到每个object_detect_mask对应的点云
        if len(local_cloud) == 0:
            return

        # For second-class, bad detections that are too offset or out of range, we
        # assign a random number to the last column of its point cloud that can later
        # be used to identify which points came from the same detection.
        if too_offset(object_mask): # 检查mask对应的bbox是否太偏
            within_range = np.ones_like(local_cloud[:, 0]) * np.random.rand() # (N, )的随机数序列
        else:
            # Mark all points of local_cloud whose distance from the camera is too far
            # as being out of range
            within_range = (local_cloud[:, 0] <= max_depth * 0.95) * 1.0  # 5% margin # (N, )的bool序列
            # All values of 1 in within_range will be considered within range, and all
            # values of 0 will be considered out of range; these 0s need to be
            # assigned with a random number so that they can be identified later.
            within_range = within_range.astype(np.float32) # (N, )的float序列
            within_range[within_range == 0] = np.random.rand() # 过远的点云分配为随机数，正常的点云为1
        global_cloud = transform_points(tf_camera_to_episodic, local_cloud) # 得到episodic_frame坐标系下的点云
        global_cloud = np.concatenate((global_cloud, within_range[:, None]), axis=1) # (N, 4)

        curr_position = tf_camera_to_episodic[:3, 3]
        closest_point = self._get_closest_point(global_cloud, curr_position) # 返回距离机器人最近的点云
        dist = np.linalg.norm(closest_point[:3] - curr_position)
        if dist < 1.0: # 过滤掉太近的object
            # Object is too close to trust as a valid object
            return

        # # 将object_name对应的点云累计到self.clouds中
        # if object_name in self.clouds:
        #     self.clouds[object_name] = np.concatenate((self.clouds[object_name], global_cloud), axis=0)
        # else:
        #     self.clouds[object_name] = global_cloud

        temp_object_pc_detect = ObjectPCDetect(object_name=object_name, confidence_score=confidence_score, pc=global_cloud)

        if object_name in multiclass_cloud:
            multiclass_cloud[object_name].append(temp_object_pc_detect)
        else:
            multiclass_cloud[object_name] = [temp_object_pc_detect]

    # zhou_percept
    def get_cluster_best_object(self, best_matched_cluster, target_class: str, curr_position: np.ndarray) -> np.ndarray:
        target_cloud = self.get_cluster_target_cloud(best_matched_cluster, target_class)
        closest_point_2d = self._get_closest_point(target_cloud, curr_position)[:2] # 返回距离机器人最近的点云 # 不需要改

        if self.last_target_cluster is None:
            self.last_target_cluster = best_matched_cluster
            self.last_target_coord = closest_point_2d
        else:
            if(self.last_target_cluster.name != best_matched_cluster.name):
                self.last_target_cluster = best_matched_cluster
                self.last_target_coord = closest_point_2d
            else: # 与上一次在同一个cluster内
                delta_dist = np.linalg.norm(closest_point_2d - self.last_target_coord)
                if delta_dist < 0.1:
                    # closest point is only slightly different
                    return self.last_target_coord
                elif delta_dist < 0.5 and np.linalg.norm(curr_position - closest_point_2d) > 2.0: # 由于与目标的距离过远而没有很好的在位置上match好
                    # closest point is a little different, but the agent is too far for
                    # the difference to matter much
                    return self.last_target_coord
                else:
                    self.last_target_coord = closest_point_2d

        return self.last_target_coord

    def _xy_to_px(self, points: np.ndarray) -> np.ndarray:
        """Converts an array of (x, y) coordinates to pixel coordinates.

        Args:
            points: The array of (x, y) coordinates to convert.

        Returns:
            The array of (x, y) pixel coordinates.
        """
        px = np.rint(points[:, ::-1] * self.pixels_per_meter) + self._episode_pixel_origin
        px[:, 0] = self.semantic_map.shape[0] - px[:, 0]
        return px.astype(int)

    '''
    def get_semantic_map(self, object_name, size: int = 1000):
        self.semantic_map = np.zeros((size, size), np.uint8)
        if(object_name in self.clouds) and (len(self.clouds[object_name])>0):
            xy_points = self.clouds[object_name][:, :2]
            pixel_points = self._xy_to_px(xy_points) # 点云的在map栅格上的像素坐标
            self.semantic_map[pixel_points[:, 1], pixel_points[:, 0]] = 1 # 计算语义地图（1为相应类别）
            # self.semantic_map = cv2.dilate(self.semantic_map.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    '''

    # zhou_percept
    def get_best_matched_cluster_new_sort(self, cluster_ls, target_class): # 进来的cluster_ls可能是reliable的，也可能是suspect的
        max_score = 0
        max_cluster = None

        for temp_cluster in cluster_ls:
            temp_score = temp_cluster.object_dict_per_cluster[target_class].confidence_score
            if(temp_score>max_score):
                max_score = temp_score
                max_cluster = temp_cluster

        return max_cluster

    # # zhou_percept
    # def get_semantic_map(self, cluster_ls, object_name, size: int = 1000):
    #     self.semantic_map = np.zeros((size, size), np.uint8)

    #     best_matched_cluster = self.get_best_matched_cluster_new_sort(cluster_ls, object_name)
    #     if (best_matched_cluster is not None):
    #         navigate_target_cloud = self.get_cluster_target_cloud(best_matched_cluster, object_name)

    #         if(len(navigate_target_cloud)>0):
    #             xy_points = navigate_target_cloud[:, :2]
    #             pixel_points = self._xy_to_px(xy_points) # 点云的在map栅格上的像素坐标
    #             self.semantic_map[pixel_points[:, 1], pixel_points[:, 0]] = 1 # 计算语义地图（1为相应类别）


    # zhou_percept （新考虑）
    def get_semantic_map(self, cluster_ls, object_name, size: int = 1000):
        best_matched_cluster = self.get_best_matched_cluster_new_sort(cluster_ls, object_name)
        if (best_matched_cluster is not None):
            navigate_target_cloud = self.get_cluster_target_cloud(best_matched_cluster, object_name)

            if(len(navigate_target_cloud)>0):
                xy_points = navigate_target_cloud[:, :2]
                pixel_points = self._xy_to_px(xy_points) # 点云的在map栅格上的像素坐标
                self.semantic_map[pixel_points[:, 1], pixel_points[:, 0]] = 1 # 计算语义地图（1为相应类别）

    
    def visualize(
        self
    ) -> np.ndarray:
        """Return an image representation of the map"""
        # Must negate the y values to get the correct orientation
        reduced_map = self.semantic_map.copy()
        map_img = np.flipud(reduced_map)
        # Make all 0s in the value map equal to the max value, so they don't throw off
        # the color mapping (will revert later)
        zero_mask = map_img == 0
        map_img[zero_mask] = np.max(map_img)
        map_img = monochannel_to_inferno_rgb(map_img)
        # Revert all values that were originally zero to white
        map_img[zero_mask] = (255, 255, 255)

        return map_img    
    
    # zhou_percept
    ''' 
    def get_best_object(self, target_class: str, curr_position: np.ndarray) -> np.ndarray: # 根据“连续多帧target点云的变化情况+机器人与target点云的距离”-->"最佳target点云"
        target_cloud = self.get_target_cloud(target_class) # 得到与target_object对应的有效的点云

        closest_point_2d = self._get_closest_point(target_cloud, curr_position)[:2] # 返回距离机器人最近的点云

        if self.last_target_coord is None:
            self.last_target_coord = closest_point_2d
        else:
            # Do NOT update self.last_target_coord if:
            # 1. the closest point is only slightly different
            # 2. the closest point is a little different, but the agent is too far for
            #    the difference to matter much
            delta_dist = np.linalg.norm(closest_point_2d - self.last_target_coord)
            if delta_dist < 0.1:
                # closest point is only slightly different
                return self.last_target_coord
            elif delta_dist < 0.5 and np.linalg.norm(curr_position - closest_point_2d) > 2.0:
                # closest point is a little different, but the agent is too far for
                # the difference to matter much
                return self.last_target_coord
            else:
                self.last_target_coord = closest_point_2d

        return self.last_target_coord
    '''

    # zhou_percept
    '''
    def update_explored(self, tf_camera_to_episodic: np.ndarray, max_depth: float, cone_fov: float) -> None:
        # 过滤掉“在当前fov及dis范围下， 不正常（随机数不为1）的点云”
        """
        This method will remove all point clouds in self.clouds that were originally
        detected to be out-of-range, but are now within range. This is just a heuristic
        that suppresses ephemeral false positives that we now confirm are not actually
        target objects.

        Args:
            tf_camera_to_episodic: The transform from the camera to the episode frame.
            max_depth: The maximum distance from the camera that we consider to be
                within range.
            cone_fov: The field of view of the camera.
        """
        camera_coordinates = tf_camera_to_episodic[:3, 3]
        camera_yaw = extract_yaw(tf_camera_to_episodic)

        for obj in self.clouds:
            within_range = within_fov_cone(
                camera_coordinates,
                camera_yaw,
                cone_fov,
                max_depth * 0.5,
                self.clouds[obj],
            ) # 提取在当前fov及dis范围下的点云
            range_ids = set(within_range[..., -1].tolist())
            for range_id in range_ids:
                if range_id == 1:
                    # Detection was originally within range
                    continue
                # Remove all points from self.clouds[obj] that have the same range_id
                self.clouds[obj] = self.clouds[obj][self.clouds[obj][..., -1] != range_id]
    '''

    # zhou_percept
    def update_other_explored(self, multiclass_cloud, tf_camera_to_episodic: np.ndarray, max_depth: float, cone_fov: float) -> None:
        # 过滤掉“在当前fov及dis范围下， 不正常（随机数不为1）的点云”
        camera_coordinates = tf_camera_to_episodic[:3, 3]
        camera_yaw = extract_yaw(tf_camera_to_episodic)

        copy_multiclass_cloud = copy.deepcopy(multiclass_cloud)

        for obj in copy_multiclass_cloud:
            copy_detect_ls = copy.deepcopy(copy_multiclass_cloud[obj])
            for temp_index in range(len(copy_detect_ls)):
                temp_detect = copy_detect_ls[temp_index]
                within_range = within_fov_cone(
                    camera_coordinates,
                    camera_yaw,
                    cone_fov,
                    max_depth * 0.5,
                    temp_detect.pc,
                ) # 提取在当前fov及dis范围下的点云
                range_ids = set(within_range[..., -1].tolist())
                if(np.all(range_ids==1)==True):
                    continue
                for range_id in range_ids:
                    if range_id == 1:
                        # Detection was originally within range
                        continue
                    temp_detect.pc = temp_detect.pc[temp_detect.pc[..., -1] != range_id]

                if(len(temp_detect.pc)==0):
                    multiclass_cloud[obj].remove(temp_detect)

            if(len(multiclass_cloud[obj])==0):
                del multiclass_cloud[obj]

    # zhou_percept
    def dowm_sample_pc(self, test_pc):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(test_pc)
        downsampled_pcd = pcd.voxel_down_sample(self.voxel_size)
        res_pc = np.asarray(downsampled_pcd.points)
        return res_pc

    # zhou_percept
    def _extract_depth_cloud(
        self,
        depth: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> np.ndarray: # 得到每个object_detect_mask对应的点云
        all_depth_mask = np.ones((depth.shape[0], depth.shape[1]))
        final_mask = all_depth_mask * 255  # [0, 255]

        valid_depth = depth.copy()
        # valid_depth[valid_depth == 0] = 1  # set all holes (0) to just be far (1)
        valid_depth = valid_depth * (max_depth - min_depth) + min_depth
        cloud = get_point_cloud(valid_depth, final_mask, fx, fy) # 深度图-->当前机器人相机坐标系下的mask区域对应的点云
        # cloud = get_random_subarray(cloud, 5000) # 从当前点云中随机采样5000个
        return cloud

    # zhou_percept
    def confuse_object_to_cluster(self, multiclass_cloud, tf_camera_to_episodic, depth_img, min_depth, max_depth, fx, fy):
        for temp_object_pc_cluster in self._all_object_cluster_ls:
            for temp_key in temp_object_pc_cluster.object_dict_per_cluster:
                temp_object_pc_cluster.object_dict_per_cluster[temp_key].is_check_now = False

        for temp_label in multiclass_cloud:
            for temp_object_pc_detect in multiclass_cloud[temp_label]: # 遍历当前RGB帧的中每一个检测的pc
                # 对于某一个特定的检测
                temp_object_pc_detect.pc = temp_object_pc_detect.pc[:, :3]
                max_intersect_num = 0
                max_object_pc_cluster = None
                for temp_i, temp_object_pc_cluster in enumerate(self._all_object_cluster_ls): # 遍历每一个cluster
                    temp_intersect_num = temp_object_pc_cluster.get_intersect_num_numpy_pro(temp_object_pc_detect.pc)

                    if(temp_intersect_num>max_intersect_num):
                        max_intersect_num = temp_intersect_num
                        max_object_pc_cluster = temp_object_pc_cluster
                

                temp_object_pc_detect.pc = self.dowm_sample_pc(temp_object_pc_detect.pc) # 点云降采样
                if(max_intersect_num==0): # 如果当前detect的点云没有对应的cluster, 则新建cluster，并将新建的cluster加入到_all_object_cluster_ls中
                    new_object_pc_cluster = ObjectPCCluster() # name从0开始
                    new_object_pc_cluster.object_dict_per_cluster[temp_label] = temp_object_pc_detect
                    # 体积融合
                    new_object_pc_cluster.object_dict_per_cluster[temp_label].pc_num = len(temp_object_pc_detect.pc)

                    # cluster的all_pc更新
                    new_object_pc_cluster.all_pc = temp_object_pc_detect.pc

                    # is_check更新
                    new_object_pc_cluster.object_dict_per_cluster[temp_label].is_check_now = True

                    self._all_object_cluster_ls.append(new_object_pc_cluster)
                else: # 若检测到对应的cluster，则将当前观测融合到对应的cluster中
                    if(temp_label in max_object_pc_cluster.object_dict_per_cluster):
                        # 体积融合
                        temp_new_pc_num = len(temp_object_pc_detect.pc)
                        temp_last_pc_num = max_object_pc_cluster.object_dict_per_cluster[temp_label].pc_num
                        res_confused_pc_num = temp_new_pc_num+temp_last_pc_num
                        max_object_pc_cluster.object_dict_per_cluster[temp_label].pc_num = res_confused_pc_num 

                        # 点云融合
                        res_confused_pc = merge_clusters_unique(max_object_pc_cluster.object_dict_per_cluster[temp_label].pc, temp_object_pc_detect.pc)
                        res_confused_pc = self.dowm_sample_pc(res_confused_pc) # 点云降采样
                        max_object_pc_cluster.object_dict_per_cluster[temp_label].pc = res_confused_pc

                        # 分数融合
                        last_score = max_object_pc_cluster.object_dict_per_cluster[temp_label].confidence_score
                        new_score = temp_object_pc_detect.confidence_score
                        max_object_pc_cluster.object_dict_per_cluster[temp_label].confidence_score = last_score*temp_last_pc_num/res_confused_pc_num+new_score*temp_new_pc_num/res_confused_pc_num

                        # is_check更新
                        max_object_pc_cluster.object_dict_per_cluster[temp_label].is_check_now = True

                    else:
                        max_object_pc_cluster.object_dict_per_cluster[temp_label] = temp_object_pc_detect
                        # 体积融合
                        max_object_pc_cluster.object_dict_per_cluster[temp_label].pc_num = len(temp_object_pc_detect.pc)
                    
                        # is_check更新
                        max_object_pc_cluster.object_dict_per_cluster[temp_label].is_check_now = True

                    # cluster的all_pc更新
                    max_object_pc_cluster.all_pc = merge_clusters_unique(max_object_pc_cluster.all_pc, temp_object_pc_detect.pc)
                    max_object_pc_cluster.all_pc = self.dowm_sample_pc(max_object_pc_cluster.all_pc) # 点云降采样

        # 处理is_recheck==False的情况
        for temp_object_pc_cluster in self._all_object_cluster_ls:
            for temp_key in temp_object_pc_cluster.object_dict_per_cluster:
                if (temp_object_pc_cluster.object_dict_per_cluster[temp_key].is_check_now == False):
                        if check_points_in_fov(temp_object_pc_cluster.object_dict_per_cluster[temp_key].pc, tf_camera_to_episodic, min_depth, max_depth)==True:
                            local_depth_cloud = self._extract_depth_cloud(depth_img, min_depth, max_depth, fx, fy)
                            local_depth_cloud = self.dowm_sample_pc(local_depth_cloud)
                            global_depth_cloud = transform_points(tf_camera_to_episodic, local_depth_cloud) # 得到episodic_frame坐标系下的点云
                            nearest_pc = find_nearest_in_radius(global_depth_cloud, temp_object_pc_cluster.object_dict_per_cluster[temp_key].pc, radius=self.voxel_size)

                            # 体积融合
                            temp_new_pc_num = len(nearest_pc)
                            temp_last_pc_num = temp_object_pc_cluster.object_dict_per_cluster[temp_key].pc_num
                            res_confused_pc_num = temp_new_pc_num+temp_last_pc_num

                            temp_object_pc_cluster.object_dict_per_cluster[temp_key].pc_num = res_confused_pc_num

                            # 分数融合
                            last_score = temp_object_pc_cluster.object_dict_per_cluster[temp_key].confidence_score
                            new_score = 0
                            temp_object_pc_cluster.object_dict_per_cluster[temp_key].confidence_score = last_score*temp_last_pc_num/res_confused_pc_num+new_score*temp_new_pc_num/res_confused_pc_num
                            
                            # is_check更新
                            temp_object_pc_cluster.object_dict_per_cluster[temp_key].is_check_now = True

    # zhou_percept
    '''
    def get_target_cloud(self, target_class: str) -> np.ndarray: # 得到与target_object对应的有效的点云
        target_cloud = self.clouds[target_class].copy()
        # Determine whether any points are within range
        within_range_exists = np.any(target_cloud[:, -1] == 1)
        if within_range_exists:
            # Filter out all points that are not within range
            target_cloud = target_cloud[target_cloud[:, -1] == 1]
        return target_cloud
    '''
    # zhou_percept
    def get_cluster_target_cloud(self, best_matched_cluster, target_class: str) -> np.ndarray:
        target_cloud = best_matched_cluster.object_dict_per_cluster[target_class].pc.copy()
        return target_cloud

    def _extract_object_cloud(
        self,
        depth: np.ndarray,
        object_mask: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> np.ndarray: # 得到每个object_detect_mask对应的点云
        final_mask = object_mask * 255  # [0, 255]
        final_mask = cv2.erode(final_mask, None, iterations=self._erosion_size)  # type: ignore

        valid_depth = depth.copy()
        valid_depth[valid_depth == 0] = 1  # set all holes (0) to just be far (1)
        valid_depth = valid_depth * (max_depth - min_depth) + min_depth
        cloud = get_point_cloud(valid_depth, final_mask, fx, fy) # 深度图-->当前机器人相机坐标系下的mask区域对应的点云
        cloud = get_random_subarray(cloud, 5000) # 从当前点云中随机采样5000个
        if self.use_dbscan:
            cloud = open3d_dbscan_filtering(cloud) # 得到最大的cluster中的所有点云

        return cloud

    def _get_closest_point(self, cloud: np.ndarray, curr_position: np.ndarray) -> np.ndarray: # 返回距离机器人最近的点云
        ndim = curr_position.shape[0]
        if self.use_dbscan:
            # Return the point that is closest to curr_position, which is 2D
            closest_point = cloud[np.argmin(np.linalg.norm(cloud[:, :ndim] - curr_position, axis=1))]
        else:
            # Calculate the Euclidean distance from each point to the reference point
            if ndim == 2:
                ref_point = np.concatenate((curr_position, np.array([0.5])))
            else:
                ref_point = curr_position
            distances = np.linalg.norm(cloud[:, :3] - ref_point, axis=1)

            # Use argsort to get the indices that would sort the distances
            sorted_indices = np.argsort(distances)

            # Get the top 20% of points
            percent = 0.25
            top_percent = sorted_indices[: int(percent * len(cloud))]
            try:
                median_index = top_percent[int(len(top_percent) / 2)]
            except IndexError:
                median_index = 0
            closest_point = cloud[median_index]
        return closest_point


def open3d_dbscan_filtering(points: np.ndarray, eps: float = 0.2, min_points: int = 100) -> np.ndarray:
    # 得到最大的cluster中的所有点云
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Perform DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(eps, min_points))

    # Count the points in each cluster
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    # Exclude noise points, which are given the label -1
    non_noise_labels_mask = unique_labels != -1
    non_noise_labels = unique_labels[non_noise_labels_mask]
    non_noise_label_counts = label_counts[non_noise_labels_mask]

    if len(non_noise_labels) == 0:  # only noise was detected
        return np.array([])

    # Find the label of the largest non-noise cluster
    largest_cluster_label = non_noise_labels[np.argmax(non_noise_label_counts)]

    # Get the indices of points in the largest non-noise cluster
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]

    # Get the points in the largest non-noise cluster
    largest_cluster_points = points[largest_cluster_indices]

    return largest_cluster_points


def visualize_and_save_point_cloud(point_cloud: np.ndarray, save_path: str) -> None:
    """Visualizes an array of 3D points and saves the visualization as a PNG image.

    Args:
        point_cloud (np.ndarray): Array of 3D points with shape (N, 3).
        save_path (str): Path to save the PNG image.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    ax.scatter(x, y, z, c="b", marker="o")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.savefig(save_path)
    plt.close()


def get_random_subarray(points: np.ndarray, size: int) -> np.ndarray:
    # 从当前点云中随机采样5000个
    """
    This function returns a subarray of a given 3D points array. The size of the
    subarray is specified by the user. The elements of the subarray are randomly
    selected from the original array. If the size of the original array is smaller than
    the specified size, the function will simply return the original array.

    Args:
        points (numpy array): A numpy array of 3D points. Each element of the array is a
            3D point represented as a numpy array of size 3.
        size (int): The desired size of the subarray.

    Returns:
        numpy array: A subarray of the original points array.
    """
    if len(points) <= size:
        return points
    indices = np.random.choice(len(points), size, replace=False)
    return points[indices]


def too_offset(mask: np.ndarray) -> bool: # 检查mask对应的bbox是否太偏
    """
    This will return true if the entire bounding rectangle of the mask is either on the
    left or right third of the mask. This is used to determine if the object is too far
    to the side of the image to be a reliable detection.

    Args:
        mask (numpy array): A 2D numpy array of 0s and 1s representing the mask of the
            object.
    Returns:
        bool: True if the object is too offset, False otherwise.
    """
    # Find the bounding rectangle of the mask
    x, y, w, h = cv2.boundingRect(mask)

    # Calculate the thirds of the mask
    third = mask.shape[1] // 3

    # Check if the entire bounding rectangle is in the left or right third of the mask
    if x + w <= third:
        # Check if the leftmost point is at the edge of the image
        # return x == 0
        return x <= int(0.05 * mask.shape[1])
    elif x >= 2 * third:
        # Check if the rightmost point is at the edge of the image
        # return x + w == mask.shape[1]
        return x + w >= int(0.95 * mask.shape[1])
    else:
        return False
