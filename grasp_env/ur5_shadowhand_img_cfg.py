# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.markers.config import FRAME_MARKER_CFG

from grasp_env import mdp
from .grasp_img_env_cfg import GraspImgEnvCfg

import math
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.devices.device_base import DevicesCfg

WIDTH = 224
HEIGHT = 224

# ------------------------------------------------------------------
# Path to your combined UR5 + ShadowHand USD (single articulation)
# ------------------------------------------------------------------
UR5_SHADOW_USD = "/home/ian/MultiGripper_Grasping_isaaclab/robot_assets/ur5_shadowhand.usd"


# ==================================================================
# Articulation Cfg for UR5 + ShadowHand
# ==================================================================
UR5_SHADOWHAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=UR5_SHADOW_USD,
        activate_contact_sensors=False,
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,       
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            # ---- UR5 arm (6 DOF) ----
            "shoulder_pan_joint":  0.0,
            "shoulder_lift_joint": -1.57,
            "elbow_joint":          1.57,
            "wrist_1_joint":       -1.57,
            "wrist_2_joint":       -1.57,
            "wrist_3_joint":        0.0,
            # ---- ShadowHand (24 DOF) ----
            "robot0_WRJ.*": 0.0,
            "robot0_FFJ.*": 0.0,
            "robot0_MFJ.*": 0.0,
            "robot0_RFJ.*": 0.0,
            "robot0_LFJ.*": 0.0,
            "robot0_THJ.*": 0.0,
        },
    ),
    actuators={
        # UR5 arm
        "ur5_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
            ],
            velocity_limit_sim=3.14,
            effort_limit_sim=150.0,
            stiffness=800.0,
            damping=40.0,
        ),
        # ShadowHand wrist (2 DOF)
        "shadow_wrist": ImplicitActuatorCfg(
            joint_names_expr=["robot0_WRJ.*"],
            effort_limit_sim=10.0, #10
            velocity_limit_sim=3.14,
            stiffness=50.0,       # ← 5 → 50
            damping=5.0,          # ← 0.5 → 5
        ),
        # ShadowHand fingers + thumb
        "shadow_fingers": ImplicitActuatorCfg(
            joint_names_expr=[
                "robot0_FFJ.*", "robot0_MFJ.*", "robot0_RFJ.*",
                "robot0_LFJ.*", "robot0_THJ.*",
            ],
            effort_limit_sim=10.0,
            velocity_limit_sim=3.14,
            stiffness=30.0,       # ← 1 → 20
            damping=3.0,          # ← 0.1 → 1
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


# ==================================================================
# Env Cfg
# ==================================================================
@configclass
class UR5ShadowHandGraspImgEnvCfg(GraspImgEnvCfg):
    """UR5 + ShadowHand grasping env (name kept as 'Franka...' for registry back-compat)."""

    def __post_init__(self):
        super().__post_init__()

        # ----------------------------------------------------------
        # Robot
        # ----------------------------------------------------------
        self.observations.policy.gripper_pos = None
        self.scene.robot = UR5_SHADOWHAND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.02,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.2,  
                    rot_sensitivity=0.2,  
                    sim_device=self.sim.device,
                ),
            }
        )
        # ----------------------------------------------------------
        # Cameras
        # ----------------------------------------------------------
        # Wrist cam — mounted on the hand palm, looking down/out toward fingertips
        self.scene.camera_0 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/shadow_hand/robot0_palm/wrist_cam",
            update_period=0.0,
            height=HEIGHT, width=WIDTH,
            data_types=["rgb", "distance_to_image_plane", "normals", "instance_segmentation_fast"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18, focus_distance=400.0,
                horizontal_aperture=20.955, clipping_range=(0.1, 10),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.06, 0.0, 0.03),
                rot=(0.5, -0.5, 0.5, -0.5),   
                convention="ros",
            ),
            update_latest_camera_pose=True,
        )

        # Front / table cam
        self.scene.camera_1 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/front_cam",
            update_period=0.0,
            height=HEIGHT, width=WIDTH,
            data_types=["rgb", "distance_to_image_plane", "normals", "instance_segmentation_fast"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18, focus_distance=400.0,
                horizontal_aperture=20.955, clipping_range=(0.1, 10),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.35, 0, 0.53),
                rot=[0.382, -0.596, -0.596, 0.382],
                convention="ros",
            ),
            update_latest_camera_pose=True,
        )

       
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
            ],
            body_name="wrist_3_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.25]),
        )

        # Gripper: binary open/close on all finger flex joints.
        # Close = fingers curl (~1.4 rad on J2/J3), thumb opposes.
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "robot0_FFJ.*", "robot0_MFJ.*", "robot0_RFJ.*",
                "robot0_LFJ.*", "robot0_THJ.*",
            ],
            open_command_expr={
                "robot0_FFJ.*": 0.0, "robot0_MFJ.*": 0.0,
                "robot0_RFJ.*": 0.0, "robot0_LFJ.*": 0.0,
                "robot0_THJ.*": 0.0,
            },
            close_command_expr={
                # 4 fingers curl
                "robot0_FFJ[0-2]": 1.4, "robot0_FFJ3": 0.0,
                "robot0_MFJ[0-2]": 1.4, "robot0_MFJ3": 0.0,
                "robot0_RFJ[0-2]": 1.4, "robot0_RFJ3": 0.0,
                "robot0_LFJ[0-2]": 1.4, "robot0_LFJ[3-4]": 0.0,
                # thumb oppose + curl
                "robot0_THJ4": 1.0, "robot0_THJ3": 0.5,
                "robot0_THJ2": 0.2, "robot0_THJ1": 0.5, "robot0_THJ0": 0.5,
            },
        )
        self.gripper_joint_names = [
            "robot0_FFJ.*", "robot0_MFJ.*", "robot0_RFJ.*",
            "robot0_LFJ.*", "robot0_THJ.*",
        ]

        # ----------------------------------------------------------
        # EE Frame (for logging / visualization)
        # ----------------------------------------------------------
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/shadow_hand/robot0_palm",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )

        # Image obs list
        self.image_obs_list = ["table_cam", "wrist_cam"]