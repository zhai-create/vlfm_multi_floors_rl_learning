from typing import Dict, Tuple, Any, Union, List, Optional
from habitat_baselines.common.tensor_dict import TensorDict
import numpy as np
import cv2
import os
from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.mapping.value_map import ValueMap

from torch.autograd import Variable as V
from torch.nn import functional as F
from vlfm.utils.geometry_utils import get_fov, rho_theta
from PIL import Image
from copy import deepcopy
from vlfm.vlm.coco_classes import COCO_CLASSES

from torchvision import transforms as trn

PROMPT_SEPARATOR = "|"

class Map_Controller:
    """
    MapController 接口，负责管理地图。
    StairNavigationHandler 将通过此接口与地图交互。
    """
    def __init__(self, text_prompt,
                    object_map_erosion_size,
                    min_obstacle_height,
                    max_obstacle_height,
                    obstacle_map_area_threshold,
                    agent_radius,
                    hole_area_thresh,
                    use_max_confidence,
                    coco_threshold,
                    non_coco_threshold):

        self._text_prompt = text_prompt
        self._object_map_erosion_size = object_map_erosion_size

        self.min_obstacle_height = min_obstacle_height
        self.max_obstacle_height = max_obstacle_height
        self.obstacle_map_area_threshold = obstacle_map_area_threshold
        self.agent_radius = agent_radius
        self.hole_area_thresh = hole_area_thresh
        self.use_max_confidence = use_max_confidence

        self._coco_threshold = coco_threshold
        self._non_coco_threshold = non_coco_threshold
        text_prompt_channels = len(self._text_prompt.split(PROMPT_SEPARATOR))

        # 初始化所有楼层的地图列表 (统一处理，避免重复逻辑)
        # 楼层越高越往后
        self._object_map_list: List[ObjectPointCloudMap] = []
        self._obstacle_map_list: List[ObstacleMap] = []
        self._value_map_list: List[ValueMap] = []

        self._object_map_list.append(ObjectPointCloudMap(
            erosion_size=self._object_map_erosion_size, size=self.MAP_SIZE
        ))
        self._obstacle_map_list.append(ObstacleMap(
            min_height=self.min_obstacle_height, max_height=self.max_obstacle_height,
            area_thresh=self.obstacle_map_area_threshold, agent_radius=self.agent_radius,
            hole_area_thresh=self.hole_area_thresh, size=self.MAP_SIZE,
        ))
        self._value_map_list.append(ValueMap(
            value_channels=text_prompt_channels,
            use_max_confidence=self.use_max_confidence,
            obstacle_map=None, size=self.MAP_SIZE,
        ))
        
        self.floor_num = len(self._obstacle_map_list[env])


    def _update_obstacle_map(self, observations_cache: List[Dict], red_semantic_pred_list: List[np.array], pitch_angle: List[int]) -> None:
        for env in range(self._num_envs):
            robot_xy = observations_cache[env]["robot_xy"]
            robot_px = self._obstacle_map[env]._xy_to_px(np.atleast_2d(robot_xy))
            
            # ✅ 新增：被动检测楼梯（仅在非爬楼梯状态时）
            # self._climb_stair_over[env]: 初始化和reset后均为True
            # self._climb_stair_flag[env]: 初始化和reset后均为0, up是1, down是2
            if self._climb_stair_over[env] and self._climb_stair_flag[env] == 0:
                self._detect_passive_stair_entry(env, robot_px)
                """
                检测机器人是否被动进入楼梯区域。
                如果机器人在楼梯区域内停留超过阈值步数，自动触发爬楼梯模式。
                """
            
            # 原有的爬楼梯状态处理（如果正在爬楼梯，则进入此处）
            if not self._climb_stair_over[env]:
                stair_map_to_use = None
                if self._climb_stair_flag[env] == 1:
                    stair_map_to_use = self._obstacle_map[env]._up_stair_map
                elif self._climb_stair_flag[env] == 2:
                    stair_map_to_use = self._obstacle_map[env]._down_stair_map

                if stair_map_to_use is not None:
                    if not self._stair_dilate_flag[env]:
                        self._temp_stair_map[env] = cv2.dilate(
                            stair_map_to_use.astype(np.uint8),
                            (7, 7),
                            iterations=1,
                        )
                        self._stair_dilate_flag[env] = True
                    else:
                        self._temp_stair_map[env] = stair_map_to_use
                    
                    # 处理楼梯状态
                    # 处理机器人达到或离开楼梯的状态逻辑。
                    self._process_stair_climb_state(env, robot_xy, robot_px, self._temp_stair_map[env], self._climb_stair_flag[env])


            self._obstacle_map[env].update_map(
                observations_cache[env]["depth"],
                observations_cache[env]["tf_camera_to_episodic"],
                observations_cache[env]["min_depth"],
                observations_cache[env]["max_depth"],
                observations_cache[env]["fx"],
                observations_cache[env]["fy"],
                observations_cache[env]["camera_fov"],
                self._object_map[env].movable_clouds,
                self._person_masks[env],
                self._stair_masks[env],
                red_semantic_pred_list[env],
                pitch_angle[env],
                self._climb_stair_over[env],
                self._reach_stair[env],
                self._climb_stair_flag[env],
            )
            frontiers = self._obstacle_map[env].frontiers # 楼梯相关的frontier不在这个里面
            self._obstacle_map[env].update_agent_traj(observations_cache[env]["robot_xy"], observations_cache[env]["robot_heading"])
            observations_cache[env]["frontier_sensor"] = frontiers

            # 附加：处理新楼层地图的创建
            # self._cur_floor_index[env]默认为0，self._object_map_list[env]中默认为一个list，该list中有一个map
            if self._obstacle_map[env]._has_up_stair and self._cur_floor_index[env] + 1 >= len(self._object_map_list[env]):
                self._add_floor_map(env, len(self._object_map_list[env]))
            if self._obstacle_map[env]._has_down_stair and self._cur_floor_index[env] == 0: # self._cur_floor_index[env] == 0-->只有之前没有下去过，才会新建下面那层的map
                self._add_floor_map(env, 0)
                self._cur_floor_index[env] += 1 # 当前不是最底层了
            
            self.floor_num[env] = len(self._obstacle_map_list[env])
            self._obstacle_map[env].project_frontiers_to_rgb_hush(observations_cache[env]["rgb"]) # 将新建的frontier投影到当前帧RGB上
        
        
        
        
        
        


    