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
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG 
from isaaclab_assets.robots.universal_robots  import UR10e_ROBOTIQ_GRIPPER_CFG  # isort: skip
import os

WIDTH = 224
HEIGHT = 224


        # self.scene.object = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.2], rot=[1, 0, 0, 0]),
        #     spawn=UsdFileCfg(
        #         usd_path= STACK_PATH,  # Use the orbit object USD file
        #         # usd_path=f"{ISAAC_NUCLEUsssssS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        #         scale=(1, 1, 1),
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #     ),
        # )



@configclass
class UR10ParallelGraspImgRelEnvCfg(GraspImgEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # self.task_info = {'task_name': "single_obj", 'object_name': "cracker_box"}
        # self.task_id = 101# Set Franka as robot
    #     self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
    #     init_state=ArticulationCfg.InitialStateCfg(
    #     joint_pos={
    #         "panda_joint1": 0.0,
    #         "panda_joint2": -0.569,
    #         "panda_joint3": 0.0,
    #         "panda_joint4": -2.810,
    #         "panda_joint5": 0.0,
    #         "panda_joint6": 2.531,
    #         "panda_joint7": 0.741,
    #         "panda_finger_joint.*": 0.04,
    #     },
    # ),)
        self.scene.robot = UR10e_ROBOTIQ_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "shoulder_pan_joint": 0.0,
                    "shoulder_lift_joint": -1.57,
                    "elbow_joint": 1.57,
                    "wrist_1_joint": -1.57,
                    "wrist_2_joint": -1.57,
                    "wrist_3_joint": -1.57,
                }
            ),
        )
        
        # robot_cfg.init_state.joint_pos.update(
        #     {
        #         "shoulder_pan_joint": 0.0,
        #         "shoulder_lift_joint": -1.57,
        #         "elbow_joint": 1.57,
        #         "wrist_1_joint": -1.57,
        #         "wrist_2_joint": -1.57,
        #         "wrist_3_joint": -1.57,
        #     }
        # )
            # focal length change from 24 to 10 - realsense d415, clipping range change from 2 to 10
        self.scene.camera_0 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link/wrist_cam",
            update_period=0.0,
            height=HEIGHT,
            width=WIDTH,
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


        
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[".*_joint"],
        body_name="robotiq_base_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=1.0,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -0.159]),
      )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.0},
            close_command_expr={"finger_joint": 1.0},
        )
        self.gripper_joint_names = [".*_inner_finger_joint"]
        # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["panda_finger.*"],
        #     open_command_expr={"panda_finger_.*": 0.04},
        #     close_command_expr={"panda_finger_.*": 0.0},
        # )
        # self.gripper_joint_names = ["panda_finger_.*"]
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/base_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.05],
                    ),
                ),
            ],
        )

        # List of image observations in policy observations
        self.image_obs_list = ["table_cam", "wrist_cam"]

