# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from torch._prims import prim
from torch.nn import init
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

from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.devices.device_base import DevicesCfg

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.kinova import KINOVA_JACO2_N7S300_CFG

WIDTH = 224
HEIGHT = 224

@configclass
class KinovaImageInfoCfg(ObsGroup):
    front_cam = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("camera_1"), "data_type": "rgb"},
    )
    wrist_cam = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("camera_0"), "data_type": "rgb"},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False

@configclass
class KinovaGraspImgRelEnvCfg(GraspImgEnvCfg):
    def __post_init__(self):

        super().__post_init__()
        #self.device = "cuda"     
        #self.sim.device = "cuda" 

        self.observations.Image_info = KinovaImageInfoCfg()

        self.observations.policy.gripper_pos = None

        KINOVA_JACO2_N7S300_CFG.spawn.rigid_props.disable_gravity = True

        self.scene.robot = KINOVA_JACO2_N7S300_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "j2n7s300_joint_1": 0.0,
            "j2n7s300_joint_2": 2.76,#2.76
            "j2n7s300_joint_3": 0.0,
            "j2n7s300_joint_4": 2.0,#2.0
            "j2n7s300_joint_5": 2.0,#2.0
            "j2n7s300_joint_6": 0.0,
            "j2n7s300_joint_7": 0.0,
            "j2n7s300_joint_finger_[1-3]": 0.2,  # close: 1.2, open: 0.2
            "j2n7s300_joint_finger_tip_[1-3]": 0.2,
        },
    ),
        actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=[".*_joint_[1-7]"],
                    effort_limit_sim={
                        ".*_joint_[1-2]": 80.0,
                        ".*_joint_[3-4]": 40.0,
                        ".*_joint_[5-7]": 20.0,
                    },
                    stiffness={
                        ".*_joint_[1-4]": 400.0,  # 40.0
                        ".*_joint_[5-7]": 200.0,  # 15.0 
                    },
                    damping={
                        ".*_joint_[1-4]": 40.0,   
                        ".*_joint_[5-7]": 20.0,   
                    },
                ),
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=[".*_finger_[1-3]", ".*_finger_tip_[1-3]"],
                    effort_limit_sim=500.0,       # 2.0 
                    stiffness=2000.0,             # 1.2
                    damping=100.0,                # 0.01
                ),
            }
        )
        
        self.scene.camera_0 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/j2n7s300_end_effector/wrist_cam",
            update_period=0.0,
            height=HEIGHT,
            width=WIDTH,
            data_types=["rgb", "distance_to_image_plane", "normals", "instance_segmentation_fast"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 10),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.08),
                rot=(0.0, 0.0, 1.0, 0.0),
                convention="ros",
            ),
            update_latest_camera_pose=True,
        )

        self.scene.camera_1 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/front_cam",
            update_period=0.0,
            height=HEIGHT,
            width=WIDTH,
            data_types=["rgb", "distance_to_image_plane", "normals", "instance_segmentation_fast"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 10),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.35, 0.0, 0.53),
                rot=(0.382, -0.596, -0.596, 0.382),
                convention="ros",
            ),
            update_latest_camera_pose=True,
        )
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["j2n7s300_joint_[1-7]"],
            body_name="j2n7s300_end_effector",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=0.6,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.12],
            ),
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "j2n7s300_joint_finger_[1-3]",
                "j2n7s300_joint_finger_tip_[1-3]",
            ],
            open_command_expr={
                "j2n7s300_joint_finger_[1-3]": 0.2,
                "j2n7s300_joint_finger_tip_[1-3]": 0.2,
            },
            close_command_expr={
                "j2n7s300_joint_finger_[1-3]": 1.2,
                "j2n7s300_joint_finger_tip_[1-3]": 1.2,
            },
        )

        
        self.gripper_joint_names = [
            "j2n7s300_joint_finger_[1-3]",
            "j2n7s300_joint_finger_tip_[1-3]"
        ]

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
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
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/j2n7s300_link_base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/j2n7s300_end_effector",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.12],
                    ),
                ),
            ],
        )
        self.image_obs_list = ["front_cam", "wrist_cam"]