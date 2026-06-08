# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
from vlfm.arguments import args as main_args

import os


os.environ["CUDA_VISIBLE_DEVICES"] = str(main_args.card_select)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = str(main_args.card_select)

import sys


if ('/home/zhaishichao/Data/VLN/dependencies/habitat-lab/examples' in sys.path):
    sys.path.remove('/home/zhaishichao/Data/VLN/dependencies/habitat-lab/examples')
    sys.path.remove('/home/zhaishichao/Data/VLN')
    sys.path.remove('/home/zhaishichao/Data/VLN/dependencies/habitat-lab/habitat-lab')
    sys.path.remove('/home/zhaishichao/Data/VLN/dependencies')

sys.path.append("/home/zhaishichao/Data/vlfm_multi_floors_rl_learning")

# sys.path.append('/home/zhaishichao/vlfm_multi_floors_learning/habitat-lab/habitat-lab')
# sys.path.append('/home/zhaishichao/MobileSAM')
# sys.path.append('/home/zhaishichao/vlfm_multi_floors_learning/habitat-lab/habitat-baselines')
# sys.path.append('/home/zhaishichao/vlfm_multi_floors_learning/frontier_exploration')
# sys.path.append('/home/zhaishichao/depth_camera_filtering')




# The following imports require habitat to be installed, and despite not being used by
# this script itself, will register several classes and make them discoverable by Hydra.
# This run.py script is expected to only be used when habitat is installed, thus they
# are hidden here instead of in an __init__.py file. This avoids import errors when used
# in an environment without habitat, such as when doing real-world deployment. noqa is
# used to suppress the unused import and unsorted import warnings by ruff.
import frontier_exploration  # noqa
import hydra  # noqa
from habitat import get_config  # noqa
from habitat.config import read_write
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.run import execute_exp
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import DictConfig



import vlfm.measurements.traveled_stairs  # noqa: F401
import vlfm.obs_transformers.resize  # noqa: F401
import vlfm.policy.action_replay_policy  # noqa: F401
import vlfm.policy.habitat_policies  # noqa: F401
import vlfm.utils.vlfm_trainer  # noqa: F401

import datetime




class HabitatConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="habitat", path="config/")


register_hydra_plugin(HabitatConfigPlugin)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="experiments/vlfm_objectnav_hm3d", # /home/zhaishichao/vlfm/config/experiments/vlfm_objectnav_hm3d.yaml
)
def main(cfg: DictConfig) -> None:
    assert os.path.isdir("data"), "Missing 'data/' directory!"
    
    if not os.path.isfile("data/dummy_policy.pth"):
        print("Dummy policy weights not found! Please run the following command first:")
        print("python -m vlfm.utils.generate_dummy_policy")
        exit(1)
    
    cfg = patch_config(cfg)
    with read_write(cfg):
        try:
            print("hello")
            # cfg.habitat.simulator.agents.main_agent.sim_sensors.pop("semantic_sensor")
            # cfg.habitat_baselines.tensorboard_dir = "/home/zhaishichao/vlfm/log_files/log_"+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        except KeyError:
            pass
    execute_exp(cfg, "eval" if cfg.habitat_baselines.evaluate else "train")


if __name__ == "__main__":
    main()
