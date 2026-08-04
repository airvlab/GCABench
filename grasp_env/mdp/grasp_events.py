# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.assets import RigidObjectCollection
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

import math
import random
import torch

def reset_root_state_uni(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_object_state.clone()
    env_num, obj_num = root_states.size(0), root_states.size(1)

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], 
        ranges[:, 1], 
        (env_num, obj_num, 6), 
        device=asset.device)

    positions = (
        root_states[..., 0:3] 
        + env.scene.env_origins[:, None].repeat(1, root_states.size(1), 1) 
        + rand_samples[..., 0:3]
        )

    # random object dropped on the ground
    # if env.random_drop_obj:
    #     z_mask = torch.bernoulli(torch.full((env_num, obj_num), 0.2, device=asset.device)).bool()
    #     z_mask = z_mask[..., None].repeat(1, 1, 3)
    #     positions[z_mask] = 0.1

    orientations_delta = math_utils.quat_from_euler_xyz(
        rand_samples[..., 3], rand_samples[..., 4], rand_samples[..., 5])
    orientations = math_utils.quat_mul(root_states[..., 3:7], orientations_delta)
    
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], 
        ranges[:, 1], 
        (env_num, obj_num, 6),  
        device=asset.device)

    velocities = root_states[..., 7:13] + rand_samples

    # set into the physics simulation
    asset.write_object_pose_to_sim(torch.cat([positions, orientations], dim=-1))
    asset.write_object_velocity_to_sim(velocities)



def reset_root_state_set(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_object_state.clone()
    env_num, obj_num = root_states.size(0), root_states.size(1)

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_set = math_utils.sample_uniform(
        ranges[:, 0], 
        ranges[:, 1], 
        (env_num, 6), 
        device=asset.device)
    
    rand_samples = rand_set[:, None, :].repeat(1, obj_num, 1)
    positions = (
        root_states[..., 0:3] 
        + env.scene.env_origins[:, None].repeat(1, root_states.size(1), 1) 
        + rand_samples[..., 0:3]
        )

    # random object dropped on the ground
    # if env.random_drop_obj:
    #     z_mask = torch.bernoulli(torch.full((env_num, obj_num), 0.2, device=asset.device)).bool()
    #     z_mask = z_mask[..., None].repeat(1, 1, 3)
    #     positions[z_mask] = 0.1

    orientations_delta = math_utils.quat_from_euler_xyz(
        rand_samples[..., 3], rand_samples[..., 4], rand_samples[..., 5])
    orientations = math_utils.quat_mul(root_states[..., 3:7], orientations_delta)
    orientations = orientations / (torch.norm(orientations, dim=-1, keepdim=True) + 1e-8)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], 
        ranges[:, 1], 
        (env_num, obj_num, 6),  
        device=asset.device)

    velocities = root_states[..., 7:13] + rand_samples

    # set into the physics simulation
    asset.write_object_pose_to_sim(torch.cat([positions, orientations], dim=-1))
    asset.write_object_velocity_to_sim(velocities)


def reset_robot_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Reset the scene to the default state specified in the scene configuration."""

    for articulation_asset in env.scene.articulations.values():
        # obtain default and deal with the offset for env origins
        default_root_state = articulation_asset.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        articulation_asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        articulation_asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
        # obtain default joint positions
        default_joint_pos = articulation_asset.data.default_joint_pos[env_ids].clone()
        default_joint_vel = articulation_asset.data.default_joint_vel[env_ids].clone()
        gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
        for joint_name in gripper_joint_names:
            if joint_name in articulation_asset.joint_names:
                idx = articulation_asset.joint_names.index(joint_name)
                default_joint_pos[:, idx] = 0.04
        # set into the physics simulation
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)


def sample_object_poses(
    num_objects: int,
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_list = []

    for i in range(num_objects):
        for j in range(max_sample_tries):
            sample = [random.uniform(range[0], range[1]) for range in range_list]

            # Accept pose if it is the first one, or if reached max num tries
            if len(pose_list) == 0 or j == max_sample_tries - 1:
                pose_list.append(sample)
                break

            # Check if pose of object is sufficiently far away from all other objects
            separation_check = [math.dist(sample[:3], pose[:3]) > min_separation for pose in pose_list]
            if False not in separation_check:
                pose_list.append(sample)
                break

    return pose_list


def randomize_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    if env_ids is None:
        return

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        pose_list = sample_object_poses(
            num_objects=len(asset_cfgs),
            min_separation=min_separation,
            pose_range=pose_range,
            max_sample_tries=max_sample_tries,
        )

        # Randomize pose for each object
        for i in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[i]
            asset = env.scene[asset_cfg.name]

            # Write pose to simulation
            pose_tensor = torch.tensor([pose_list[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])
            
            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
            )