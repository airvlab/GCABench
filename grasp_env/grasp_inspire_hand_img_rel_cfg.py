# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from grasp_env import mdp
from .grasp_img_env_cfg import GraspImgEnvCfg
import isaaclab.sim as sim_utils
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from robot_assets.franka_cfg import FRANKA_INSPIRE_HAND_CFG

WIDTH = 224
HEIGHT = 224

INSPIRE_HAND_JOINT_NAMES = [
    "L_index_intermediate_joint",
    "L_index_proximal_joint",
    "L_middle_intermediate_joint",
    "L_middle_proximal_joint",
    "L_pinky_intermediate_joint",
    "L_pinky_proximal_joint",
    "L_ring_intermediate_joint",
    "L_ring_proximal_joint",
    "L_thumb_distal_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_proximal_yaw_joint",
]

INSPIRE_HAND_OPEN = {
    "L_index_intermediate_joint": 0.0,
    "L_index_proximal_joint": 0.0,
    "L_middle_intermediate_joint": 0.0,
    "L_middle_proximal_joint": 0.0,
    "L_pinky_intermediate_joint": 0.0,
    "L_pinky_proximal_joint": 0.0,
    "L_ring_intermediate_joint": 0.0,
    "L_ring_proximal_joint": 0.0,
    "L_thumb_distal_joint": 0.0,
    "L_thumb_proximal_pitch_joint": 0.0,
    "L_thumb_proximal_yaw_joint": 0.0,
}

INSPIRE_HAND_CLOSE = {
    "L_index_proximal_joint": 1.2,
    "L_index_intermediate_joint": 1.2,
    "L_middle_proximal_joint": 1.2,
    "L_middle_intermediate_joint": 1.2,
    "L_ring_proximal_joint": 1.2,
    "L_ring_intermediate_joint": 1.2,
    "L_pinky_proximal_joint": 1.2,
    "L_pinky_intermediate_joint": 1.2,
    "L_thumb_proximal_yaw_joint": 1.0,
    "L_thumb_proximal_pitch_joint": 0.5,
    "L_thumb_distal_joint": 0.3,
}


@configclass
class FrankaInspireHandGraspImgRelEnvCfg(GraspImgEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # gripper_pos only supports parallel 2-finger grippers
        self.observations.policy.gripper_pos = None

        self.scene.robot = FRANKA_INSPIRE_HAND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=INSPIRE_HAND_OPEN | {
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.569,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.810,
                    "panda_joint5": 0.0,
                    "panda_joint6": 2.531,
                    "panda_joint7": 0.741,
                },
            ),
        )

        self.scene.camera_0 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/inspire_left_hand/L_hand_base_link/wrist_cam",
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
            body_name="L_hand_base_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=1,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=INSPIRE_HAND_JOINT_NAMES,
            open_command_expr=INSPIRE_HAND_OPEN,
            close_command_expr=INSPIRE_HAND_CLOSE,
        )
        self.gripper_joint_names = INSPIRE_HAND_JOINT_NAMES

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (1, 1, 1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/inspire_left_hand/L_hand_base_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        # pos=[0.0, 0.0, 0.1534],
                        pos=[0.0, 0.0, 0.1034],
                        # pos=[0.0, 0.0, 0.06],
                    ),
                ),
            ],
        )

        self.image_obs_list = ["table_cam", "wrist_cam"]
