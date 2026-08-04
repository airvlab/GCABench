# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control
* :obj:`FRANKA_ROBOTIQ_GRIPPER_CFG`: Franka robot with Robotiq_2f_85 gripper

Reference: https://github.com/frankaemika/franka_ros
"""


import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

_ROBOT_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))

# Gripper joints commanded by gripper_action (right_outer_knuckle_joint is USD mimic-only).
ROBOTIQ_GRIPPER_ACTION_JOINTS = [
    # "finger_joint",
    # "left_inner_finger_joint",
    # "right_inner_finger_joint",
    # "left_inner_finger_knuckle_joint",
    # "right_inner_finger_knuckle_joint",
    "finger_joint",
    "right_outer_knuckle_joint",
    "right_outer_finger_joint",
    "right_inner_finger_joint",
    "right_inner_finger_knuckle_joint",
    "left_outer_finger_joint",
    "left_inner_finger_knuckle_joint",
    "left_inner_finger_joint",
]
ROBOTIQ_GRIPPER_OPEN = {name: 0.0 for name in ROBOTIQ_GRIPPER_ACTION_JOINTS}
ROBOTIQ_GRIPPER_CLOSE = {
    # "finger_joint": 0.78,
    # "left_inner_finger_joint": -0.785398163,
    # "right_inner_finger_joint": 0.785398163,
    # "left_inner_finger_knuckle_joint": -0.785398163,
    # "right_inner_finger_knuckle_joint": -0.785398163,
    "finger_joint": 0.785398163,
    "right_outer_knuckle_joint": 0.785398163,
    "right_outer_finger_joint": 0.0,
    "right_inner_finger_joint": 0.785398163,
    "right_inner_finger_knuckle_joint": -0.785398163,
    "left_outer_finger_joint": 0.0 ,
    "left_inner_finger_knuckle_joint": -0.785398163,
    "left_inner_finger_joint": -0.785398163,
}

##
# Configuration
##

FRANKA_PANDA_VACUUM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_ROBOT_ASSETS_DIR, "franka_cobot_suction.usd"),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            # "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=80.0,
            damping=4.0,
        ),
        # "panda_hand": ImplicitActuatorCfg(
        #     joint_names_expr=["panda_finger_joint.*"],
        #     effort_limit_sim=200.0,
        #     stiffness=2e3,
        #     damping=1e2,
        # ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


FRANKA_PANDA_VACUUM_HIGH_PD_CFG = FRANKA_PANDA_VACUUM_CFG.copy()

# UR10_LONG_SUCTION_CFG.spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10/ur10.usd"
FRANKA_PANDA_VACUUM_HIGH_PD_CFG.spawn.variants = {"Gripper": "Vacuum"}
FRANKA_PANDA_VACUUM_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
# FRANKA_PANDA_VACUUM_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_VACUUM_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_PANDA_VACUUM_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_PANDA_VACUUM_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_PANDA_VACUUM_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
FRANKA_PANDA_SHORT_SUCTION_HIGH_PD_CFG = FRANKA_PANDA_VACUUM_HIGH_PD_CFG.copy()
FRANKA_PANDA_SHORT_SUCTION_HIGH_PD_CFG.spawn.usd_path = os.path.join(
    _ROBOT_ASSETS_DIR, "franka_cobot_suction _short.usd"
)


FRANKA_ROBOTIQ_GRIPPER_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()
# FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
# FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.variants = {"Gripper": "Robotiq_2F_85"}
FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.usd_path = os.path.join(_ROBOT_ASSETS_DIR, "franka_robotiq.usd")
FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "finger_joint": 0.0,
    ".*_inner_finger_joint": 0.0,
    ".*_inner_finger_knuckle_joint": 0.0,
    ".*_outer_.*_joint": 0.0,
}
# FRANKA_ROBOTIQ_GRIPPER_CFG.init_state.pos = (-0.85, 0, 0.76)
FRANKA_ROBOTIQ_GRIPPER_CFG.actuators = {
    "panda_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"],
        effort_limit_sim=5200.0,
        velocity_limit_sim=2.175,
        stiffness=1100.0,
        damping=80.0,
    ),
    "panda_forearm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"],
        effort_limit_sim=720.0,
        velocity_limit_sim=2.61,
        stiffness=1000.0,
        damping=80.0,
    ),
    "gripper_drive": ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"],  # "right_outer_knuckle_joint" is its mimic joint
        effort_limit_sim=1650,
        velocity_limit_sim=10.0,
        stiffness=17,
        damping=0.02,
    ),
    # enable the gripper to grasp in a parallel manner
    # "gripper_finger": ImplicitActuatorCfg(
    #     joint_names_expr=[".*_inner_finger_joint"],
    #     effort_limit_sim=50,
    #     velocity_limit_sim=10.0,
    #     stiffness=0.0,
    #     # damping=0.001,
    #     damping=0.0001,
    # ),
    # # set PD to zero for passive joints in close-loop gripper
    # "gripper_passive": ImplicitActuatorCfg(
    #     joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
    #     effort_limit_sim=1.0,
    #     velocity_limit_sim=10.0,
    #     stiffness=0.0,
    #     damping=0.0,
    # ),
}
# FRANKA_ROBOTIQ_GRIPPER_CFG.actuators = {
#     "panda_shoulder": ImplicitActuatorCfg(
#         joint_names_expr=["panda_joint[1-4]"],
#         effort_limit_sim=5200.0,
#         velocity_limit_sim=2.175,
#         stiffness=1100.0,
#         damping=80.0,
#     ),
#     "panda_forearm": ImplicitActuatorCfg(
#         joint_names_expr=["panda_joint[5-7]"],
#         effort_limit_sim=720.0,
#         velocity_limit_sim=2.61,
#         stiffness=1000.0,
#         damping=80.0,
#     ),
#     "gripper_drive": ImplicitActuatorCfg(
#         joint_names_expr=["finger_joint"],  # "right_outer_knuckle_joint" is its mimic joint
#         effort_limit_sim=1650,
#         velocity_limit_sim=10.0,
#         # stiffness=17,
#         # damping=0.02,
#         stiffness=80.0,
#         damping=0.05,

#     ),
#     # enable the gripper to grasp in a parallel manner
#     "gripper_finger": ImplicitActuatorCfg(
#         joint_names_expr=[".*_inner_finger_joint"],
#         effort_limit_sim=50,
#         velocity_limit_sim=10.0,
#         stiffness=1.0,
#         damping=0.05,
#         # stiffness=0.0,  # Sim uses ~0.05 for UR compatibility
#         # damping=5000.0  # Sim typically uses <10000.0
#     ),
#     "gripper_knuckle": ImplicitActuatorCfg(
#         joint_names_expr=[".*_inner_finger_knuckle_joint"],
#         effort_limit_sim=50.0,
#         velocity_limit_sim=10.0,
#         stiffness=20.0,
#         damping=0.5,
#         # stiffness=0.0,  # Sim uses ~0.05 for UR compatibility
#         # damping=5000.0  # Sim typically uses <10000.0
#     ),
#     # mimic joint — must stay passive (stiffness fights USD mimic → rotation)
#     "gripper_passive": ImplicitActuatorCfg(
#         joint_names_expr=["right_outer_knuckle_joint"],
#         effort_limit_sim=1.0,
#         velocity_limit_sim=10.0,
#         stiffness=80.0,
#         damping=0.0,
#         # stiffness=0.0,  # Sim uses ~0.05 for UR compatibility
#         # damping=5000.0  # Sim typically uses <10000.0
#     ),
#     # set PD to zero for passive joints in close-loop gripper
#     # "gripper_passive": ImplicitActuatorCfg(
#     # joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
#     # #         joint_names_expr=[
#     # #     ".*_inner_finger_knuckle_joint",
#     # #     ".*_outer_knuckle_joint",      # covers left AND right
#     # #     ".*_outer_finger_joint",         # if these exist as DOFs
#     # # ],
#     #     effort_limit_sim=1.0,
#     #     velocity_limit_sim=10.0,
#     #     stiffness=0.0,
#     #     damping=0.0,
#     # ),

# }

#  'finger_joint', 
#  'right_outer_knuckle_joint', 
 
#  'left_inner_knuckle_joint', 
#  'left_outer_finger_joint', 
#  'right_outer_finger_joint', 
#  'left_inner_finger_joint', 
#  'right_inner_finger_joint',
#  'right_inner_finger_knuckle_joint'


"""Configuration of Franka Emika Panda robot with Robotiq_2f_85 gripper."""
FRANKA_INSPIRE_HAND_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()
# FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
# FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.variants = {"Gripper": "Robotiq_2F_85"}
FRANKA_INSPIRE_HAND_CFG.spawn.usd_path = os.path.join(_ROBOT_ASSETS_DIR, "franka_inspire_hand.usd")
FRANKA_INSPIRE_HAND_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_INSPIRE_HAND_CFG.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False,
    solver_position_iteration_count=32,
    solver_velocity_iteration_count=1,
)
FRANKA_INSPIRE_HAND_CFG.init_state.joint_pos = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
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
FRANKA_INSPIRE_HAND_CFG.actuators = {
    "panda_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"],
        effort_limit_sim=5200.0,
        velocity_limit_sim=2.175,
        stiffness=1100.0,
        damping=80.0,
    ),
    "panda_forearm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"],
        effort_limit_sim=720.0,
        velocity_limit_sim=2.61,
        stiffness=1000.0,
        damping=80.0,
    ),
    "left-hand": ImplicitActuatorCfg(
        joint_names_expr=["L_.*"],
        effort_limit_sim=30.0,
        velocity_limit_sim=10.0,
        stiffness=40.0,
        damping=4.0,
        armature=0.001,
    ),
}
"""Configuration of Franka Emika Panda robot with Inspire Hand."""

