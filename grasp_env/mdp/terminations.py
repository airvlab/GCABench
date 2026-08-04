# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # rewarded if the object is lifted above the threshold
    return distance < threshold

def root_height_below_minimum_objcollection(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    dropped = asset.data.object_link_pos_w[:, :, 2] < minimum_height
    return dropped.any(dim=1)


def any_object_goal_success(
    env: ManagerBasedRLEnv,
    # threshold: float,
    minimal_height: float,
    # command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    objects_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    """Terminate when ANY object is lifted AND reaches goal position."""
    robot: RigidObject = env.scene[robot_cfg.name]
    objects = env.scene[objects_cfg.name]  # This is your RigidObjectCollection
    # command = env.command_manager.get_command(command_name)
    
    # # compute the desired position in the world frame
    # des_pos_b = command[:, :3]
    # des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    
    # Get positions of all objects: (num_envs, num_objects, 3)
    object_positions = objects.data.object_link_pos_w[:, :, :3]
    object_heights = objects.data.object_link_pos_w[:, :, 2]- objects.data.default_object_state[:, :, 2]  # Height of each object above the ground
    
    # Calculate distances for all objects: (num_envs, num_objects)
    # distances = torch.norm(des_pos_w.unsqueeze(1) - object_positions, dim=2)
    
    # Check conditions for all objects
    lifted = object_heights > minimal_height  # (num_envs, num_objects)
    # close_to_goal = distances < threshold     # (num_envs, num_objects)
    
    # Success if ANY object satisfies both conditions
    success_per_object = lifted # close_to_goal  # (num_envs, num_objects)
    return success_per_object.any(dim=1)  # (num_envs,) - True if any object succeeds
