# ur10_grasp_img_rel_cfg.py

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from grasp_env import mdp

from .grasp_img_env_cfg import GraspImgEnvCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import SurfaceGripper

from isaaclab.assets import SurfaceGripperCfg
# from .assets.universal_robots import UR10_SHORT_SUCTION_CFG
from isaaclab_assets.robots.universal_robots import UR10_SHORT_SUCTION_CFG
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.devices.device_base import DevicesCfg
from .grasp_env_rel_cfg import GraspEnvRelCfg

WIDTH = 224
HEIGHT = 224


@configclass
# class UR10ImageInfoCfg(ObsGroup):
#     front_cam = ObsTerm(
#         func=mdp.image,
#         params={"sensor_cfg": SceneEntityCfg("camera_1"), "data_type": "rgb"},
#     )
#     wrist_cam = ObsTerm(
#         func=mdp.image,
#         params={"sensor_cfg": SceneEntityCfg("camera_0"), "data_type": "rgb"},
#     )

#     def __post_init__(self):
#         self.enable_corruption = False
#         self.concatenate_terms = False


@configclass
class UR10suctionGraspEnvRelCfg(GraspEnvRelCfg):
    def __post_init__(self):
        super().__post_init__()
     # Suction grippers currently require CPU simulation
     
        self.device = "cpu"
       
        
        # self.observations.Image_info = UR10ImageInfoCfg()

        
        robot_cfg = UR10_SHORT_SUCTION_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )
        
        robot_cfg.init_state.joint_pos.update(
            {
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.57,
                "elbow_joint": 1.57,
                "wrist_1_joint": -1.57,
                "wrist_2_joint": -1.57,
                "wrist_3_joint": -1.57,
            }
        )
        self.scene.robot = robot_cfg
        

        
        self.scene.surface_gripper = SurfaceGripperCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ee_link/SurfaceGripper",
            max_grip_distance=0.05,
            # max_grip_distance=0.05,
            shear_force_limit=500.0,
            coaxial_force_limit=500.0,
            retry_interval=0.05,
        )

        
        self.scene.camera_0 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ee_link/Camera/wrist_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane", "normals", "instance_segmentation_fast"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 10),
            ),
            offset=CameraCfg.OffsetCfg(pos=(0, 0, 0), rot=(1.0, 0.0, 0.0, 0.0), convention="opengl"),
            # offset=CameraCfg.OffsetCfg(pos=(-0.011, 0.038, 0.1964), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
            update_latest_camera_pose=True,
        )
# (-2,57.55,1.5),(-0.6691306, 0.7431448, 0, 0)
        
        self.scene.camera_1 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/front_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane","normals", "instance_segmentation_fast"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 10)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.35, 0, 0.53), rot=[0.382,-0.596,-0.596,0.382], convention="ros"            ),
            update_latest_camera_pose= True,
        )
            # [0.5,-0.5,-0.5,0.5]based on this rotate around z -arctan(0.4/0.69) then -90 degree around x
        self.scene.camera_2 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/side_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane","normals", "instance_segmentation_fast"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 10)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.5, 0.69, 0.5), rot= [0, 0, -0.8453, 0.5343], convention="ros"            ),
            update_latest_camera_pose= True,
        )

        
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[".*_joint"],
        body_name="ee_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=1.0,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -0.159]),
      )
        self.actions.gripper_action = mdp.SurfaceGripperBinaryActionCfg(
            asset_name="surface_gripper",
            open_command=-1.0,
            close_command=1.0,
        )

       
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.001, 0.001, 0.001)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.02,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.02,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.05, 0.05],
                    ),
                ),
            ],
        )

        
        self.image_obs_list = ["front_cam", "wrist_cam"]