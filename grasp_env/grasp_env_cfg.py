# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from re import I

from grasp_env.mdp.rewards import avoid_collision_score
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg, RigidObjectCollectionCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from . import mdp
import os
from pathlib import Path
# from .element_cfg import *
from isaaclab.sensors import CameraCfg
from grasp_env.mdp import *
# Scene definition
##
nums = [f"{i:03}" for i in range(0, 83)]
HOME_PATH = Path(os.getcwd())
MODEL_PATH = os.path.join(HOME_PATH, "models", "models_ifl") # path to the models
STACK_PATH = os.path.join(MODEL_PATH, "005", "orbit_obj.usd") # path to the objects
ACRONYM_PATH = 'models/asset_release/benchmark_objects/'
MGN_PATHS = [os.path.join(MODEL_PATH, a, "orbit_obj.usd") for a in nums] # path to the objects
test_path = 'models/asset_release/benchmark_objects/Book/1d493a57a21833f2d92c7cdc3939488b/mesh.usd'
num_objs = 4
# objects = sample_class_objects(num_objs, 'Book')  
# MGN_CFGs = [
#     RigidObjectCfg(
#         spawn=UsdFileCfg(
#             usd_path=mgn, #f"{obj['asset_root']}/{obj['usd_file']}",
#             rigid_props=RigidBodyPropertiesCfg(
#                     solver_position_iteration_count=16,
#                     solver_velocity_iteration_count=1,
#                     max_angular_velocity=1000.0,
#                     max_linear_velocity=1000.0,
#                     max_depenetration_velocity=5.0,
#                     disable_gravity=False,
#                 ),
#             mass_props = sim_utils.MassPropertiesCfg(density=5.0),
#             articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#                 articulation_enabled=False,
#             ),
#         #     semantic_tags=[("class", f"{mgn.split('/')[-2]}"), ("color", "red")],
#         ),
#         init_state=RigidObjectCfg.InitialStateCfg(
#             # pos=ROBOT_POS,
#             pos=(0.5, 0, 0.2),
#             rot=(1, 0.0, 0.0, 0),
#         ),
#         collision_group = 0
#     )
#     for mgn in MGN_PATHS
# ]


# DIFF = 0.2
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    camera_0: CameraCfg = MISSING
    camera_1: CameraCfg = MISSING
    camera_2: CameraCfg = MISSING
    # objects: RigidObjectCollectionCfg = TG_CFGs
    objects: RigidObjectCollectionCfg = MISSING
    # tray: AssetBaseCfg = MISSING
    # task_info: dict = {'task_name': "single_obj", 'object_name': "cracker_box"}

    # objects: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
    #         rigid_objects={
    #             f"obj_{i}": MGN_CFGs[i].replace(
    #                 init_state=MGN_CFGs[i].init_state.replace(pos=(0.5, 0+i*DIFF*0.2, 0.01+ i*DIFF)),
    #                 prim_path="{ENV_REGEX_NS}/obj_"+str(i)
    #             )
    #             for i in range(num_objs)
    #         },
    #         )

    # if TASK = 'SINGLE_OBJ':
        
    

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd", scale=(1.0, 1.0, 1.0),),
        # spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd", scale=(0.5, 1.0, 1.0),),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )


    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )





@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class ImageCfg(ObsGroup):
        rgb = ObsTerm(func=mdp.rgb_capture)
        # normals = ObsTerm(func=mdp.normal_capture)
        # instance_segmentation_fast = ObsTerm(func=mdp.inst_capture)
        distance_to_image_plane = ObsTerm(func=mdp.depth_capture)
        pcd = ObsTerm(func=mdp.pcd_capture)

        # self.gripper_pose = ObsTerm(func=mdp.gripper_pose_capture)
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False


    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        cam_pos = ObsTerm(func=mdp.cam_pos)
        ee_pose_with_gripper = ObsTerm(func=mdp.ee_pose_with_gripper)
        # rgb = ObsTerm(func=mdp.rgb_capture)
        # # normals = ObsTerm(func=mdp.normal_capture)
        # # instance_segmentation_fast = ObsTerm(func=mdp.inst_capture)
        # distance_to_image_plane = ObsTerm(func=mdp.depth_capture)
        # pcd = ObsTerm(func=mdp.pcd_capture)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    Image_info: ImageCfg=ImageCfg()
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_robot_to_default, mode="reset")

    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uni,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.05, 0.05), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("objects"),
    #     },
    # )
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_set,
        mode="reset",
        params={
            # "pose_range": {"x": (-0.05, 0.05), "y": (-0.2, 0.2), "z": (0.0, 0.0)},
            # if push b06 change the range to x(-0.0.02, 0.02)
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)}, # easy mode
            # "pose_range": {"x": (-0.02, 0.02), "y": (-0.05, 0.05), "z": (0.0, 0.0)}, # easy mode
            # "pose_range": {"x": (-0.00000001, 0.00000001), "y": (-0.00000001, 0.00000001), "z": (0.0, 0.0)}, # easy mode
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("objects"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)

    # object_goal_tracking = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=16.0,
    # )

    # object_goal_tracking_fine_grained = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=5.0,
    # )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    balance_score = RewTerm(func=mdp.balance_score, weight=1.0)
    avoid_collision_score = RewTerm(func=avoid_collision_score, weight=1.0)
    # balance_score = RewTerm(func=mdp.balance_score, weight=1.0)
    smoothness_score = RewTerm(func=mdp.get_current_smoothness, weight=1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # success = DoneTerm(func=mdp.any_object_goal_success, params={"minimal_height": 0.1})
    success = DoneTerm(func=mdp.any_object_goal_success, params={"minimal_height": 0.3})
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum_objcollection, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("objects")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##


@configclass
class GraspEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


