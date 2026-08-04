#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Standard Library


# import omni.replicator.core as rep
import argparse
from isaaclab.app import AppLauncher
from isaacsim import SimulationApp
# import carb


# NUCLEUS_ASSET_ROOT_DIR = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
# """Path to the root directory on the Nucleus Server."""

# NVIDIA_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/NVIDIA"
# """Path to the root directory on the NVIDIA Nucleus Server."""

# ISAAC_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac"
# """Path to the ``Isaac`` directory on the NVIDIA Nucleus Server."""

# ISAACLAB_NUCLEUS_DIR = f"{ISAAC_NUCLEUS_DIR}/IsaacLab"
# """Path to the ``Isaac/IsaacLab`` directory on the NVIDIA Nucleus Server."""

import time

# Third Party
import torch
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
# CuRobo
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
    # # Table
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        
    #     spawn=UsdFileCfg(usd_path=, scale=(1.0, 1.0, 1.0),),
    #     # spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd", scale=(0.5, 1.0, 1.0),),
    # )
device = torch.device("cuda:0")
world_cfg = {
    "mesh": {
        "table": {
            "pose": [0.5, 0, 0, 0.707, 0, 0, 0.707],
            "file_path": '/home/hanyi/code/Grasp-with-trajectory/models/table.obj',
        },
        "objects": {
            "pose": [0.0, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0],
            "file_path": "/home/hanyi/code/Grasp-with-trajectory/models/models_ifl/011/textured.obj",
        }
    },
}
grasp_goal = torch.tensor([0.4898790717124939, 0.03387565165758133, 0.21, 0.08706195128902613, -0.3150224823953772, 0.9445566583529758, -0.03152413970691987], device=device)

def demo_full_config_collision_free_ik(world_file = "collision_cage.yml", robot_file="franka.yml", grasp_goals=None):
    tensor_args = TensorDeviceType()

    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )
    # world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=True,
        # use_fixed_samples=True,
    )
    ik_solver = IKSolver(ik_config)

    # print(kin_state)
    print("Running Single IK")
    # for _ in range(10):
    #     q_sample = ik_solver.sample_configs(1)
    #     kin_state = ik_solver.fk(q_sample)
    #     goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)

    #     st_time = time.time()
    #     result = ik_solver.solve_batch(goal)
    #     torch.cuda.synchronize()
    #     total_time = (time.time() - st_time) / q_sample.shape[0]
    #     print(
    #         # "Success, Solve Time(s), Total Time(s)",
    #         torch.count_nonzero(result.success).item(),
    #         result.success,
    #         # result.solve_time,
    #         # total_time,
    #         # 1.0 / total_time,
    #         # torch.mean(result.position_error) * 100.0,
    #         # torch.mean(result.rotation_error) * 100.0,
    #     )



    st_time = time.time()
    result = ik_solver.solve_batch(Pose(grasp_goal[:3], grasp_goal[3:]))
    torch.cuda.synchronize()
    total_time = (time.time() - st_time) / 1
    print(
        # "Success, Solve Time(s), Total Time(s)",
        torch.count_nonzero(result.success).item(),
        result.success,
        # result.solve_time,
        # total_time,
        # 1.0 / total_time,
        torch.mean(result.position_error) * 100.0,
        torch.mean(result.rotation_error) * 100.0,
    )
    exit()




def demo_full_config_batch_env_collision_free_ik(world_file = "collision_cage.yml", robot_file="franka.yml"):
    tensor_args = TensorDeviceType()
    world_file = ["collision_test.yml", "collision_cubby.yml"]

    robot_file = "franka.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )
    world_cfg = [
        WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), x))) for x in world_file
    ]
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=100,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=False,
        # use_fixed_samples=True,
    )
    ik_solver = IKSolver(ik_config)
    q_sample = ik_solver.sample_configs(len(world_file))
    kin_state = ik_solver.fk(q_sample)
    goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)

    print("Running Batch Env IK")
    for _ in range(3):
        st_time = time.time()
        result = ik_solver.solve_batch_env(goal)
        print(result.success)
        torch.cuda.synchronize()
        print(
            "Success, Solve Time(s), Total Time(s)",
            torch.count_nonzero(result.success).item() / len(q_sample),
            result.solve_time,
            time.time() - st_time,
        )

    q_sample = ik_solver.sample_configs(10 * len(world_file))
    kin_state = ik_solver.fk(q_sample)
    goal = Pose(
        kin_state.ee_position.view(len(world_file), 10, 3),
        kin_state.ee_quaternion.view(len(world_file), 10, 4),
    )

    print("Running Batch Env Goalset IK")
    for _ in range(3):
        st_time = time.time()
        result = ik_solver.solve_batch_env_goalset(grasp_goal)
        torch.cuda.synchronize()
        print(
            "Success, Solve Time(s), Total Time(s)",
            torch.count_nonzero(result.success).item() / len(result.success.view(-1)),
            result.solve_time,
            time.time() - st_time,
        )


if __name__ == "__main__":
    # demo_basic_ik()
    demo_full_config_collision_free_ik()
    # demo_full_config_batch_env_collision_free_ik()
