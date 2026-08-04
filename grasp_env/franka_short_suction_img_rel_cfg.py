# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from re import T
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg, RigidObjectCollectionCfg
from grasp_env import mdp
from .grasp_img_env_cfg import GraspImgEnvCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

from isaaclab.assets import SurfaceGripper
from isaaclab.assets import SurfaceGripperCfg
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from robot_assets.franka_cfg import FRANKA_PANDA_SHORT_SUCTION_HIGH_PD_CFG
import os

WIDTH = 224
HEIGHT = 224


@configclass
class FrankaShortSuctionGraspImgRelEnvCfg(GraspImgEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # self.task_info = {'task_name': "single_obj", 'object_name': "cracker_box"}
        # self.task_id = 101# Set Franka as robot
        self.scene.robot = FRANKA_PANDA_SHORT_SUCTION_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 2.531,
            # "panda_joint7": 0.741,
            "panda_joint7": 0.0,
            # "panda_finger_joint.*": 0.04,
        },
    ),)
        self.scene.surface_gripper = SurfaceGripperCfg(
            prim_path="{ENV_REGEX_NS}/Robot/franka_instanceable/suction_gripper/SurfaceGripper",
            max_grip_distance=0.05,
            shear_force_limit=500.0,
            coaxial_force_limit=500.0,
            retry_interval=0.05,
        )
            # focal length change from 24 to 10 - realsense d415, clipping range change from 2 to 10
        self.scene.camera_0 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/franka_instanceable/suction_gripper/Camera/wrist_cam",
            update_period=0.0,
            height=HEIGHT,
            width=WIDTH,
            data_types=["rgb", "distance_to_image_plane","normals", "instance_segmentation_fast"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 10)
            ),
            offset=CameraCfg.OffsetCfg(
                # pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"
                pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="opengl"
            ),
            update_latest_camera_pose= True,
        )

        self.scene.camera_1 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/front_cam",
            update_period=0.0,
            height=HEIGHT,
            width=WIDTH,
            data_types=["rgb", "distance_to_image_plane","normals", "instance_segmentation_fast"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 10)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.35, 0, 0.53), rot=[0.382,-0.596,-0.596,0.382], convention="ros"            ),
            update_latest_camera_pose= True,
        )



        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="suction_gripper",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            # scale=0.5,
            scale=1,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
            # body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.153]),
        )

        self.actions.gripper_action = mdp.SurfaceGripperBinaryActionCfg(
            asset_name="surface_gripper",
            open_command=-1.0,
            close_command=1.0,
        )

        # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["panda_finger.*"],
        #     open_command_expr={"panda_finger_.*": 0.04},
        #     close_command_expr={"panda_finger_.*": 0.0},
        # )
        # self.gripper_joint_names = ["panda_finger_.*"]
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/franka_instanceable/panda_link0",

            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/franka_instanceable/suction_gripper",
                    name="end_effector",
                    offset=OffsetCfg(
                        # pos=[0.0, 0.0, 0.1534],
                        pos=[0.0, 0.0, 0.04],
                        # pos=[0.0, 0.0, 0.06],
                    ),
                ),
            ],
        )

        # List of image observations in policy observations
        self.image_obs_list = ["table_cam", "wrist_cam"]

