# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
# from isaaclab_assets.robots.franka import FRANKA_ROBOTIQ_GRIPPER_CFG
from robot_assets.franka_cfg import (
    FRANKA_ROBOTIQ_GRIPPER_CFG,
    ROBOTIQ_GRIPPER_ACTION_JOINTS,
    ROBOTIQ_GRIPPER_CLOSE,
    ROBOTIQ_GRIPPER_OPEN,
)
import os

WIDTH = 224
HEIGHT = 224

# FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.rigid_props.disable_gravity = True
# FRANKA_ROBOTIQ_GRIPPER_CFG .actuators["panda_shoulder"].stiffness = 400.0
# FRANKA_ROBOTIQ_GRIPPER_CFG .actuators["panda_shoulder"].damping = 80.0
# FRANKA_ROBOTIQ_GRIPPER_CFG .actuators["panda_forearm"].stiffness = 400.0
# FRANKA_ROBOTIQ_GRIPPER_CFG .actuators["panda_forearm"].damping = 80.0
# @configclass
class FrankaRobotiqGraspImgRelEnvCfg(GraspImgEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # self.task_info = {'task_name': "single_obj", 'object_name': "cracker_box"}
        # self.task_id = 101# Set Franka as robot
        self.scene.robot = FRANKA_ROBOTIQ_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
             "panda_joint1": 0.0,
             "panda_joint2": -0.569,
             "panda_joint3": 0.0,
             "panda_joint4": -2.810,
             "panda_joint5": 0.0,
             "panda_joint6": 2.531,
            # "panda_joint6": 3.037,
             "panda_joint7": 0.741,
             "finger_joint":-0.0,
             ".*_inner_finger_joint": 0.0,
             ".*_inner_finger_knuckle_joint": 0.0,
             ".*_outer_.*_joint": 0.0,
        },
    ),)
            # focal length change from 24 to 10 - realsense d415, clipping range change from 2 to 10
        self.scene.camera_0 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            # prim_path="{ENV_REGEX_NS}/Robot/Robotiq_2F_85_config/Robotiq_2F_85/base_link/wrist_cam",
            update_period=0.0,
            height=HEIGHT,
            width=WIDTH,
            data_types=["rgb", "distance_to_image_plane","normals", "instance_segmentation_fast"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 10)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"
            ),
            update_latest_camera_pose= True,
        )
        # original camera settings
        # self.scene.camera_1 = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/table_cam",
        #     update_period=0.0,
        #     height=480,
        #     width=640,
        #     data_types=["rgb", "distance_to_image_plane","normals", "instance_segmentation_fast"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
        #     ),
        #     offset=CameraCfg.OffsetCfg(
        #         pos=(1.6, 0, 0.8), rot=(0.35355, -0.61237, -0.61237, 0.35355), convention="ros"            ),
        #     update_latest_camera_pose= True,
        # )
        # grapvla camera setting
        # [0.5,-0.5,-0.5,0.5] towads the x. rotate -arctan(0.53/1.15) = -25.26 degree around z axis the quaternion is [0.382,-0.596,-0.596,0.382]
        # [0.0, 0.0, -0.70710678, 0.70710678] towads the -y]
        # pos=(1.35, 0, 0.53), rot=[0.382,-0.596,-0.596,0.382]
        # pos=(0.5, 0.69, 0.5), rot= [0, 0, -0.8453, 0.5343]

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
                pos=(1.35, 0, 0.53), rot=[0.382,-0.596,-0.596,0.382], convention="ros"),
            update_latest_camera_pose= True,
        )
            # [0.5,-0.5,-0.5,0.5]based on this rotate around z -arctan(0.4/0.69) then -90 degree around x
        # self.scene.camera_2 = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/side_cam",
        #     update_period=0.0,
        #     height=HEIGHT,
        #     width=WIDTH,
        #     data_types=["rgb", "distance_to_image_plane","normals", "instance_segmentation_fast"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=18, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 10)
        #     ),
        #     offset=CameraCfg.OffsetCfg(
        #         pos=(0.5, 0.69, 0.5), rot= [0, 0, -0.8453, 0.5343], convention="ros"            ),
        #     update_latest_camera_pose= True,
        # )


    #     self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
    #     asset_name="robot",
    #     joint_names=[".*_joint"],
    #     body_name="panda_hand",
    #     controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    #     scale=1.0,
    #     body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -0.159]),
    #   )


        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            # body_name="panda_hand",
            # body_name="Robotiq_2F_85_config",
            body_name="base_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            # scale=0.5,
            scale=1,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
            # body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.153]),
        )

        # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=ROBOTIQ_GRIPPER_ACTION_JOINTS,
        #     open_command_expr=ROBOTIQ_GRIPPER_OPEN,
        #     close_command_expr=ROBOTIQ_GRIPPER_CLOSE,
        # )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint":0.0},
            close_command_expr={"finger_joint": -0.80},
        )
        self.gripper_joint_names = [".*_inner_finger_joint"]
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (1, 1, 1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",

            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    # prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    prim_path="{ENV_REGEX_NS}/Robot/Robotiq_2F_85_config/Robotiq_2F_85/base_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        # pos=[0.0, 0.0, 0.1534],
                        pos=[0.0, 0.0, 0.1034],
                        # pos=[0.0, 0.0, 0.06],
                    ),
                ),
            ],
        )

        # List of image observations in policy observations
        self.image_obs_list = ["table_cam", "wrist_cam"]