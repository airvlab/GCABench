# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to record demonstrations with Isaac Lab environments using human teleoperation.

This script allows users to record demonstrations operated by human teleoperation for a specified task.
The recorded demonstrations are stored as episodes in a hdf5 file. Users can specify the task, teleoperation
device, dataset directory, and environment stepping rate through command-line arguments.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --teleop_device           Device for interacting with environment. (default: keyboard)
    --dataset_file            File path to export recorded demos. (default: "./datasets/dataset.hdf5")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Number of demonstrations to record. (default: 0)
    --num_success_steps       Number of continuous steps with task success for concluding a demo as successful. (default: 10)
"""

"""Launch Isaac Sim Simulator first."""

# Standard library imports
# import isaaclab_tasks 
import argparse
import contextlib
import matplotlib.pyplot as plt
# Third-party imports
import gymnasium as gym
import numpy as np
import os
import time
import torch
# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
# parser.add_argument("--task", type=str, default= 'Grasp-Kinova-J2N75300-IK-Rel-img', help="Name of the task.")
parser.add_argument("--task", type=str, default= 'Grasp-Franka-IK-Rel-img', help="Name of the task.")
# parser.add_argument("--task", type=str, default= 'Grasp-UR10-Long-Suction-IK-Rel-img', help="Name of the task.")
# parser.add_argument("--task", type=str, default= 'Grasp-UR10-Short-Suction-IK-Rel-img', help="Name of the task.")
parser.add_argument("--teleop_device", type=str, default="SpaceMouse", help="Device for interacting with environment.")
parser.add_argument(
    "--dataset_dir", type=str, default="./datasets", help="dir path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos", type=int, default=20, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=20,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--task_id",
    type=str,
    default= 'd01',
    help="id of the task in task config",
)
parser.add_argument(
    "--num_objects",
    type=int,
    default= 4,
    help="id of the task in task config",
)
parser.add_argument(
    "--test",
    type=bool,
    default=False,
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.num_envs = 1 
# args_cli.device = "cpu"
app_launcher_args = vars(args_cli)
# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app



# Additional Isaac Lab imports that can only be imported after the simulator is running
from isaaclab.devices import OpenXRDevice, Se3Keyboard, Se3SpaceMouse


# import datacollection  
import grasp_env
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from grasp_env.custom_recorder import MultiModalRecorderCfg  
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
# from isaaclab.envs.mdp.actions import RigidObjectCollectionCfg
from utils.load_utils import TaskBuilder


def main():
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task
    # set the objects for task
    task_builder = TaskBuilder(task_id=args_cli.task_id)
    TG_CFGs = task_builder.create_scene_objects(num_objs=args_cli.num_objects)
    # TG_CFGs= create_scene_objects(task_id=args_cli.task_id, num_objs=args_cli.num_objects, base_pos=[0.5, 0, 0.01])
    env_cfg.scene.objects = TG_CFGs
    #if task_id start with a, then the table scale should be 0.5
    if args_cli.task_id.startswith("a"):
        env_cfg.scene.table.spawn.scale = [0.7, 1.0, 1.0] 

    if args_cli.task_id.startswith("g"):
        env_cfg.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.6, 0.1, 0], rot=[0.0, 0, 0, 1.0]),      
            spawn=UsdFileCfg(usd_path="/home/hanyi/code/Grasp-with-trajectory/models/env_shelves/cabinet_b01/cabinet_b01.usd", scale=(1.0, 1.0, 1.0),),
        )
        env_cfg.scene.plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0.05]),
        spawn=GroundPlaneCfg(),
        )
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    # env_cfg.recorders.obs_keys = ["policy", "Image_info"] 
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    # env.recorder_manager._dataset_file_handler.add_env_args({'num_objects': len(env_cfg.scene.objects.rigid_objects), 'task_id': args_cli.task_id})



    env.reset()
    wait_action = torch.zeros((1, 7), device=env.device)
    for i in range(5):
        obs, _, _, _, _ = env.step(wait_action)
            # Match your GraspVLA packaging style: use third_view as front, wrist/side as side
        front_image_np = obs["policy"]["table_cam"][0].cpu().numpy() # (H, W, 3)
        wrist_image_np = obs["policy"]["wrist_cam"][0].cpu().numpy() # (H, W, 3)
        ee_pos = obs["policy"]["eef_pos"][0].cpu().numpy() # (1,3)
        ee_quat = obs["policy"]["eef_quat"][0].cpu().numpy() # (1,4)
        ee_gripper = obs["policy"]["gripper_pos"][0].cpu().numpy() 

        # resize the image into 256x256
        front_image_np = (front_image_np - front_image_np.min()) / (front_image_np.max() - front_image_np.min())* 255
        wrist_image_np = (wrist_image_np - wrist_image_np.min()) / (wrist_image_np.max() - wrist_image_np.min())* 255
        wrist_image_np = wrist_image_np.astype(np.uint8)
        front_image_np = front_image_np.astype(np.uint8)
        print(ee_pos, ee_quat, ee_gripper)
        # show the rgb images
        plt.imshow(front_image_np)
        plt.show()
        plt.imshow(wrist_image_np)
        plt.show()
    env.reset()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
