# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
from vlfm.arguments import args as main_args

import os
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from torch import Tensor

from vlfm.mapping.frontier_map import FrontierMap
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.utils.geometry_utils import closest_point_within_threshold
from vlfm.vlm.blip2itm import BLIP2ITMClient
# from vlfm.vlm.blip2 import BLIP2Client
from vlfm.vlm.detections import ObjectDetections

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

PROMPT_SEPARATOR = "|"

from vlfm.utils.vlfm_trainer import VLFMTrainer
from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix
from depth_camera_filtering import filter_depth

def new_normalize_angle(x):
    x = np.mod(x + np.pi, 2 * np.pi) - np.pi  # 约束到 [-π, π]
    return x


class BaseITMPolicy(BaseObjectNavPolicy):
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    # _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _selected__frontier_color: Tuple[int, int, int] = (255, 0, 0)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5
    
    '''
    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(2)
    '''

    @staticmethod
    def _vis_reduce_fn(i: np.ndarray) -> np.ndarray:
        return np.max(i, axis=-1)

    def __init__(
        self,
        text_prompt: str,
        use_max_confidence: bool = True,
        sync_explored_areas: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(text_prompt, use_max_confidence, sync_explored_areas, *args, **kwargs)
        # self._itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12082")))
        '''
        self._itm = BLIP2ITMClient(port=int(main_args.blip_port))
        '''
        # self._itm = BLIP2Client(port=int(os.environ.get("BLIP2_PORT", "12185")))
        

        '''
        self._text_prompt = text_prompt
        self._value_map: ValueMap = ValueMap(
            value_channels=len(text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map if sync_explored_areas else None,
        )
        self._acyclic_enforcer = AcyclicEnforcer()
        '''

    '''
    def _reset(self) -> None:
        super()._reset()
        self._value_map.reset()
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)
    '''
    '''
    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        frontiers = self._observations_cache["frontier_sensor"]
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("No frontiers found during exploration, stopping.")
            return self._stop_action, None
        best_frontier, best_value = self._get_best_frontier(observations, frontiers)
        # os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%" # 临时
        print(f"Best value: {best_value*100:.2f}%")
        pointnav_action = self._pointnav(best_frontier, stop=False)

        return pointnav_action, best_frontier
    '''

    '''
    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
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
        robot_xy = self._observations_cache["robot_xy"]
        sorted_pts, sorted_values = self._sort_frontiers_by_value(robot_xy, observations, frontiers)
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])

        # os.environ["DEBUG_INFO"] = "" # 临时
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
                    # os.environ["DEBUG_INFO"] += "Sticking to last point. " # 临时
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
            # os.environ["DEBUG_INFO"] += "All frontiers are cyclic. " # 临时
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier
        # os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%" # 临时

        return best_frontier, best_value
    '''

    def binary_matrix_to_rgb(self, matrix):
        """
        将0-1二维矩阵转换为RGB图像
        参数:
            matrix: 二维numpy数组，元素为0或1
        返回:
            rgb_image: RGB图像矩阵，0显示为白色，1显示为红色
        """
        # 创建全白RGB图像（所有通道值为255）
        height, width = matrix.shape
        rgb_image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # 将值为1的位置设为红色（R=255, G=0, B=0）
        rgb_image[matrix == 1] = [0, 0, 255]
        
        return rgb_image

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)

        if not self._visualize:
            return policy_info

        markers = []

        # Draw frontiers on to the cost map
        frontiers = self._all_floor_frontier_list[self._cur_floor_index]
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            markers.append((frontier[:2], marker_kwargs))

        if not np.array_equal(self._last_goal, np.zeros(2)):
            # Draw the pointnav goal on to the cost map
            if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                color = self._selected__frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal, marker_kwargs))

        temp_value_map_img, _camera_positions, _last_camera_yaw = self._value_map.visualize(markers, reduce_fn=self._vis_reduce_fn)
        policy_info["value_map"] = cv2.cvtColor(
            temp_value_map_img,
            cv2.COLOR_BGR2RGB,
        )

        temp_object_map = self._object_map.visualize()
        temp_object_map = self._value_map._traj_vis.draw_trajectory(temp_object_map, _camera_positions, _last_camera_yaw)
        policy_info["semantic_map"] = cv2.cvtColor(
            temp_object_map,
            cv2.COLOR_BGR2RGB,
        )
        return policy_info

    '''
    def _update_value_map(self) -> None:

        for temp_index in range(4):
            if(temp_index==0):
                all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
                cosines = [
                    [
                        self._itm.cosine(
                            rgb,
                            p.replace("target_object", self._target_object.replace("|", "/")),
                        )
                        for p in self._text_prompt.split(PROMPT_SEPARATOR)
                    ]
                    for rgb in all_rgb
                ]
                for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
                    cosines, self._observations_cache["value_map_rgbd"]
                ):
                    self._value_map.update_map(np.array(cosine), depth, tf, min_depth, max_depth, fov) # 用blip分数更新

            else:
                all_rgb = [VLFMTrainer.rgb_ls[temp_index-1].cpu().numpy()]
                cosines = [
                    [
                        self._itm.cosine(
                            rgb,
                            p.replace("target_object", self._target_object.replace("|", "/")),
                        )
                        for p in self._text_prompt.split(PROMPT_SEPARATOR)
                    ]
                    for rgb in all_rgb
                ]

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

                other_angle_observations_cache = [(VLFMTrainer.rgb_ls[temp_index-1].cpu().numpy(), other_angle_depth, other_angle_tf_camera_to_episodic, self._observations_cache["value_map_rgbd"][0][3], self._observations_cache["value_map_rgbd"][0][4], self._observations_cache["value_map_rgbd"][0][5])]

                for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
                    cosines, other_angle_observations_cache
                ):
                    self._value_map.update_map(np.array(cosine), depth, tf, min_depth, max_depth, fov) # 用blip分数更新

        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )
    '''

    '''
    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        raise NotImplementedError
    '''

class ITMPolicy(BaseITMPolicy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frontier_map: FrontierMap = FrontierMap()

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        self._pre_step(observations, masks)
        if self._visualize:
            self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _reset(self) -> None:
        super()._reset()
        self._frontier_map.reset()

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        rgb = self._observations_cache["object_map_rgbd"][0][0]
        text = self._text_prompt.replace("target_object", self._target_object)
        self._frontier_map.update(frontiers, rgb, text)  # type: ignore
        return self._frontier_map.sort_waypoints()


class ITMPolicyV2(BaseITMPolicy):
    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:

        '''
        self._pre_step(observations, masks)
        self._update_value_map() # get::当前帧的value_map   

        # 与操作
        explored_numeric = observations["explored_area"].astype(np.float32)
        now_value_map = (explored_numeric * self._value_map._value_map.squeeze())
        now_value_map[observations["navigable_map"]==0] = -1
        now_value_map[observations["obstacle_map"]==1] = -1
        now_value_map = now_value_map[:, :, np.newaxis]
        self._value_map._value_map_for_vis = now_value_map
        observations['value_map'] = self._value_map._value_map_for_vis    
        '''

        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def select_action(self):
        return super().select_action()


    '''
    def _sort_frontiers_by_value(
        self, robot_xy, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(robot_xy, frontiers, 0.5)
        return sorted_frontiers, sorted_values
    '''

    '''
    def select_action(self):
        return super().select_action()
    '''

class ITMPolicyV3(ITMPolicyV2):
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._exploration_thresh = exploration_thresh

        def visualize_value_map(arr: np.ndarray) -> np.ndarray:
            # Get the values in the first channel
            first_channel = arr[:, :, 0]
            # Get the max values across the two channels
            max_values = np.max(arr, axis=2)
            # Create a boolean mask where the first channel is above the threshold
            mask = first_channel > exploration_thresh
            # Use the mask to select from the first channel or max values
            result = np.where(mask, first_channel, max_values)

            return result

        self._vis_reduce_fn = visualize_value_map  # type: ignore

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5, reduce_fn=self._reduce_values)

        return sorted_frontiers, sorted_values

    def _reduce_values(self, values: List[Tuple[float, float]]) -> List[float]:
        """
        Reduce the values to a single value per frontier

        Args:
            values: A list of tuples of the form (target_value, exploration_value). If
                the highest target_value of all the value tuples is below the threshold,
                then we return the second element (exploration_value) of each tuple.
                Otherwise, we return the first element (target_value) of each tuple.

        Returns:
            A list of values, one per frontier.
        """
        target_values = [v[0] for v in values]
        max_target_value = max(target_values)

        if max_target_value < self._exploration_thresh:
            explore_values = [v[1] for v in values]
            return explore_values
        else:
            return [v[0] for v in values]
