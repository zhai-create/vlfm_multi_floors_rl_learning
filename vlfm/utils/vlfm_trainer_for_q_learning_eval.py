# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = '2'

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

from vlfm.policy.unet_model import HistoryAwareUNet


def extract_scalars_from_info(info: Dict[str, Any]) -> Dict[str, float]:
    info_filtered = {k: v for k, v in info.items() if not isinstance(v, list)}
    return extract_scalars_from_info_habitat(info_filtered)


@baseline_registry.register_trainer(name="vlfm")
class VLFMTrainer(PPOTrainer):
    envs: VectorEnv

    # # load训练好的模型
    # policy_model = HistoryAwareUNet(in_channels=6).to("cuda")
    # pre_policy = 'checkpoints_single_v3_easy_data/epoch_5.pt'
    # checkpoint = torch.load(pre_policy, map_location="cuda")['model_state']
    # policy_model.load_state_dict(checkpoint)


    policy_model = HistoryAwareUNet(in_channels=6).to("cuda")
    policy_model.load_state_dict(torch.load("Models_train_PPO/policy/init_q_learning/47_actor"))
    policy_model.eval()

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
        # logger_file_name = "./log_files_dino/log_"+date_time
        # logger_file_name = "./log_files/log_0.2m_"+date_time+"_init_46"
        # logger_file_name = "./log_files/log_0.2m_il_policy_5w_data_eval_"+date_time
        # logger_file_name = "./log_files/log_0.2m_il_policy_5w_easy_data_eval_"+date_time+"_init_115"
        # logger_file_name = "./log_files/log_0.2m_il_policy_5w_easy_data_eval_model_5_"+date_time
        logger_file_name = "./log_files/log_for_q_learning_eval_"+date_time+"_init_600" 
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
            config.habitat_baselines.eval.split = "val"
            config.habitat.dataset.split = config.habitat_baselines.eval.split

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


        self._init_envs(config, is_eval=True) # 初始化env(暂时不管)

        self._agent = self._create_agent(None) # 初始化agent(暂时不管)
        action_shape, discrete_actions = get_action_space_info(self._agent.policy_action_space)  # 获取action_space形状+action_space是否离散

        # 加载智能体状态：待检查
        if self._agent.actor_critic.should_load_agent_state:
            self._agent.load_state_dict(ckpt_dict)

        observations = self.envs.reset() # 重置所有并行环境并获取初始观察

        
        # 待检查：多个act的执行顺序
        # 待检查，batch和observations
        batch = batch_obs(observations, device=self.device) # 将来自多个环境的观察批量处理成单个张量，并将其移动到指定的 self.device
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore # 对批量观察应用任何观察转换（例如，归一化、调整大小）

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

        # 从601集开始
        for i in range(600):
            self.envs.reset()
            print("=====> episode: {} <=====".format(i+2))


        # 开始评估：遍历每一个episode（evals_per_ep默认为1， self.envs.num_envs为1）
        while len(stats_episodes)+600 < (number_of_eval_episodes * evals_per_ep) and self.envs.num_envs > 0:
        # while len(stats_episodes) < (number_of_eval_episodes * evals_per_ep) and self.envs.num_envs > 0:
            
            current_episodes_info = self.envs.current_episodes() #  获取关于每个环境中当前回合的元数据

            with inference_mode(): # 禁用梯度计算
                action_data, all_save_dict_ls = self._agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                ) # 关键的一步 (actor_critic是HabitatITMPolicyV2类型,实际调用HabitatMixin类中的act)

                # 没用到，暂时不管 （将选择的动作 ID 记录到一个文件中，用于调试）
                if "VLFM_RECORD_ACTIONS_DIR" in os.environ:
                    action_id = action_data.actions.cpu()[0].item()
                    filepath = os.path.join(
                        os.environ["VLFM_RECORD_ACTIONS_DIR"],
                        "actions.txt",
                    )
                    # If the file doesn't exist, create it
                    if not os.path.exists(filepath):
                        open(filepath, "w").close()
                    with open(filepath, "a") as f:
                        f.write(f"{action_id}\n")

                # 更新循环状态和先前动作
                # 赋值test_recurrent_hidden_states和prev_actions
                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = action_data.rnn_hidden_states
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    for i, should_insert in enumerate(action_data.should_inserts):
                        if should_insert.item():
                            test_recurrent_hidden_states[i] = action_data.rnn_hidden_states[i]
                            prev_actions[i].copy_(action_data.actions[i])  # type: ignore



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
            outputs = self.envs.step(step_data) 
            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            policy_infos = self._agent.actor_critic.get_extra(action_data, infos, dones)
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i]) # 将策略特定的信息合并到环境信息中

            
            # 新的 observations 被批量处理和转换，类似于初始设置，为下一次智能体的 act 调用准备 batch
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
            ) 
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            ) # 根据来自环境的 dones 标志进行更新

            rewards = torch.tensor(rewards_l, dtype=torch.float, device="cpu").unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)
                elif int(next_episodes_info[i].episode_id) == 123123123:
                    envs_to_pause.append(i)

                if len(self.config.habitat_baselines.eval.video_option) > 0:
                    hab_vis.collect_data(batch, infos, action_data.policy_info)

                # episode ended
                if not not_done_masks[i].item(): # 如果环境 i 中的一个回合已结束
                    pbar.update() # 更新进度条
                    episode_stats = {"reward": current_episode_reward[i].item()} # 创建一个字典来存储此回合的最终统计数据
                    episode_stats.update(extract_scalars_from_info(infos[i])) # 用于从 infos 字典中提取标量指标（例如，success、spl、distance_to_goal）
                    current_episode_reward[i] = 0 # 重置此环境的奖励累加器
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
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

                    real_index = num_total+600
                    # real_index = num_total
                    # episode_stats: dict_keys(['reward', 'distance_to_goal', 'success', 'spl', 'soft_spl', 'distance_to_goal_reward', 'traveled_stairs', 'yaw', 'target_detected', 'stop_called'])
                    writer.add_scalar("Result/reward", episode_stats["reward"], real_index)
                    writer.add_scalar("Result/distance_to_goal", episode_stats["distance_to_goal"], real_index)
                    writer.add_scalar("Result/success_num", num_successes, real_index)
                    writer.add_scalar("Result/spl_per_episode", episode_stats["spl"], real_index)
                    writer.add_scalar("Result/stop_called", episode_stats["stop_called"], real_index)
                    writer.add_scalar("Result/spl_mean", np.mean(all_spl_ls), real_index)
                    writer.add_scalar("Result/ne_mean", np.mean(all_ne_ls), real_index)

                    

                    from vlfm.utils.episode_stats_logger import (
                        log_episode_stats,
                    )


                    # 确定failure_cause
                    try:
                        failure_cause = log_episode_stats(
                            current_episodes_info[i].episode_id,
                            current_episodes_info[i].scene_id,
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
                            current_episodes_info[i].episode_id,
                        )

            not_done_masks = not_done_masks.to(device=self.device) # 将 not_done_masks 张量移动到目标设备 (self.device)，因为它将在下一次迭代中作为智能体的输入
            
            
            # 从活动的 VectorEnv (self.envs) 中移除暂停的环境
            # 从 test_recurrent_hidden_states、not_done_masks、current_episode_reward、prev_actions、batch（观察）和 rgb_frames 中过滤掉相应的条目，以便这些张量/列表仅包含仍处于活动状态的环境的数据
            # 确保了智能体仅对尚未完成其评估配额的环境进行操作并从中接收数据
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

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

        self.envs.close()
