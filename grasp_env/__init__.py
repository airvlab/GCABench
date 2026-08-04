# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os
from . import agents
from .grasp_rel_cfg import FrankaGraspEnvCfg
from isaaclab.envs import ManagerBasedEnv
from .grasp_env import GraspRLEnv
from .grasp_img_rel_cfg import FrankaGraspImgRelEnvCfg
from .grasp_img_env_cfg import GraspImgEnvCfg
from .ur10_short_grasp_img_rel_cfg import UR10ShortSuctionGraspImgRelEnvCfg
from .ur10_long_grasp_img_rel_cfg import UR10LongSuctionGraspImgRelEnvCfg
from .J2N7S300_grasp_img_rel_cfg import KinovaGraspImgRelEnvCfg
from .franka_vacuum_img_rel_cfg import FrankaVacuumGraspImgRelEnvCfg
from .franka_short_suction_img_rel_cfg import FrankaShortSuctionGraspImgRelEnvCfg
from .ur10_parallel_img_rel_cfg import UR10ParallelGraspImgRelEnvCfg
from .franka_robotiq_img_rel_cfg import FrankaRobotiqGraspImgRelEnvCfg
from .ur10_rel_cfg import UR10suctionGraspEnvRelCfg
from .ur5_shadowhand_img_cfg import UR5ShadowHandGraspImgEnvCfg
from .grasp_inspire_hand_img_rel_cfg import FrankaInspireHandGraspImgRelEnvCfg
# from .franka_robotiq_rel import FrankaRobotiqImgRelEnvCfg
##
# Register Gym environments.
##

##
# Joint Position Control
##

# gym.register(
#     id="Isaac-Lift-Cube-Franka-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubeLiftEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-Lift-Cube-Franka-Play-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubeLiftEnvCfg_PLAY",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
#     },
#     disable_env_checker=True,
# )

# ##
# # Inverse Kinematics - Absolute Pose Control
# ##

# gym.register(
#     id="Isaac-Lift-Cube-Franka-IK-Abs-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaCubeLiftEnvCfg",
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaTeddyBearLiftEnvCfg",
#     },
#     disable_env_checker=True,
# )

# ##
# # Inverse Kinematics - Relative Pose Control
# ##

# gym.register(
#     id="Isaac-Lift-Cube-Franka-IK-Rel-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:FrankaCubeLiftEnvCfg",
#         "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Grasp-Franka-IK-Rel-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": FrankaGraspEnvCfg, #FrankaCubeLiftEnvCfg,
#         # "env_cfg_entry_point": FrankaCubeLiftEnvCfg,
#         "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
#     },
#     disable_env_checker=True,
# )


gym.register(
    id="Grasp-Franka-IK-Rel-v0",
    entry_point="grasp_env:GraspRLEnv",
    kwargs={
        "env_cfg_entry_point": FrankaGraspEnvCfg, #FrankaCubeLiftEnvCfg,
        # "env_cfg_entry_point": FrankaCubeLiftEnvCfg,
        # "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
    },
    disable_env_checker=True,
)


gym.register(
    id="Grasp-Franka-IK-Rel-img",
    entry_point="grasp_env:GraspRLEnv",
    kwargs={
        "env_cfg_entry_point": FrankaGraspImgRelEnvCfg, #FrankaCubeLiftEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_84.json"),
        # "env_cfg_entry_point": FrankaCubeLiftEnvCfg,
        # "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
    },
    disable_env_checker=True,
)
gym.register(
    id="Grasp-UR10-Short-Suction-IK-Rel-img",
    entry_point="grasp_env:GraspRLEnv",
    kwargs={
        "env_cfg_entry_point": UR10ShortSuctionGraspImgRelEnvCfg, #FrankaCubeLiftEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_84.json"),
        # "env_cfg_entry_point": FrankaCubeLiftEnvCfg,
        # "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Grasp-UR10-Long-Suction-IK-Rel-img",
    entry_point="grasp_env:GraspRLEnv",
    kwargs={
        "env_cfg_entry_point": UR10LongSuctionGraspImgRelEnvCfg, #FrankaCubeLiftEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_84.json"),
        # "env_cfg_entry_point": FrankaCubeLiftEnvCfg,
        # "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Grasp-Kinova-J2N75300-IK-Rel-img",
    entry_point="grasp_env:GraspRLEnv",
    kwargs={
        "env_cfg_entry_point": KinovaGraspImgRelEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0],"robomimic/bc_rnn_image_84.json"),
    
    },
    disable_env_checker=True,
)

gym.register(
    id="Grasp-Franka-Vacuum-IK-Rel-img",
    entry_point="grasp_env:GraspRLEnv",
    kwargs={
        "env_cfg_entry_point": FrankaVacuumGraspImgRelEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_84.json"),
    },
    disable_env_checker=True,
)
gym.register(
    id="Grasp-Franka-Short-Suction-IK-Rel-img",
    entry_point="grasp_env:GraspRLEnv",
    kwargs={
        "env_cfg_entry_point": FrankaShortSuctionGraspImgRelEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_84.json"),
    },
    disable_env_checker=True,
)
gym.register(
    id="Grasp-UR10-Parallel-IK-Rel-img",
    entry_point="grasp_env:GraspRLEnv",
    kwargs={
        "env_cfg_entry_point": UR10ParallelGraspImgRelEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_84.json"),
    },
    disable_env_checker=True,
)
gym.register(
    id="Grasp-Franka-Robotiq-IK-Rel-img",
    entry_point="grasp_env:GraspRLEnv",
    kwargs={
        "env_cfg_entry_point": FrankaRobotiqGraspImgRelEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_84.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Grasp-UR10-short-Grasp-vla",
    entry_point="grasp_env:GraspRLEnv",
    kwargs={
        "env_cfg_entry_point": UR10suctionGraspEnvRelCfg, #FrankaCubeLiftEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_84.json"),
        # "env_cfg_entry_point": FrankaCubeLiftEnvCfg,
        # "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
    },
    disable_env_checker=True,
)
# gym.register(
#     id="Grasp-Franka-Robotiq-IK-Rel-img",
#     entry_point="grasp_env:GraspRLEnv",
#     kwargs={
#         "env_cfg_entry_point": FrankaRobotiqImgRelEnvCfg, #FrankaCubeLiftEnvCfg,
#         "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_84.json"),
#         # "env_cfg_entry_point": FrankaCubeLiftEnvCfg,
#         # "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
#     },
#     disable_env_checker=True,
# )

gym.register(
    id="Grasp-UR5-ShadowHand-IK-Rel-img",
    entry_point="grasp_env:GraspRLEnv",
    kwargs={
        "env_cfg_entry_point": UR5ShadowHandGraspImgEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_84.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Grasp-Franka-InspireHand-IK-Rel-img",
    entry_point="grasp_env:GraspRLEnv",
    kwargs={
        "env_cfg_entry_point": FrankaInspireHandGraspImgRelEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_84.json"),
    },
    disable_env_checker=True,
)