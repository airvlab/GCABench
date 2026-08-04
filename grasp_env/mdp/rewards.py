# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from sympy import degree
import torch
from typing import TYPE_CHECKING
from math import pi

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.assets import RigidObjectCollection
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("objects")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    objects: RigidObjectCollection = env.scene[object_cfg.name]
    obj_1_height = objects.data.object_link_pos_w[:, 0, 2]  
    return torch.where(obj_1_height > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    objects: RigidObjectCollection = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    obj_1_pos_w = objects.data.object_link_pos_w[:, 0, :] 
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(obj_1_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    objects: RigidObjectCollection = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    obj_1_pos = objects.data.object_link_pos_w[:, 0, :]  
    distance = torch.norm(des_pos_w - obj_1_pos[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (obj_1_pos[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))



def keep_balance(
    env: ManagerBasedRLEnv,
    max_angle: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    """Reward the agent for keeping the object balanced."""
    objects: RigidObjectCollection = env.scene[object_cfg.name]
    object_orientations = objects.data.object_link_quat_w[:, :, :3]
    initial_orientation = objects.data.default_object_state[:, :, 3:7]
    # calculate the angle difference between the current and initial orientation in quaternion
    object_orientations_diff = torch.abs(object_orientations - initial_orientation)
    # INSERT_YOUR_CODE
    # Calculate the relative quaternion: q_rel = q_current * q_initial_conj
    # object_orientations: (num_envs, num_objects, 4)
    # initial_orientation: (num_envs, num_objects, 4)
    # Ensure both are normalized
    def normalize_quat(q):
        return q / torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-8)
    q_current = normalize_quat(objects.data.object_link_quat_w[:, :, :4])
    q_initial = normalize_quat(objects.data.default_object_state[:, :, 3:7])

    # Quaternion conjugate: (w, x, y, z) -> (w, -x, -y, -z)
    q_initial_conj = q_initial.clone()
    q_initial_conj[..., 1:] *= -1

    # Quaternion multiplication: (a, b) -> a * b
    def quat_mul(q, r):
        # q, r: (..., 4)
        w1, x1, y1, z1 = q.unbind(-1)
        w2, x2, y2, z2 = r.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)

    q_rel = quat_mul(q_current, q_initial_conj)  # (..., 4)
    # Clamp w to [-1, 1] for safe acos
    q_rel_w = q_rel[..., 0].clamp(-1.0, 1.0)
    angle = 2 * torch.acos(q_rel_w)  # (...,) in radians
    # angle in degree
    angle_degree = angle * 180 / pi
    return torch.where(angle_degree < max_angle, 1.0, 0.0)



def balance_score(
    env: ManagerBasedRLEnv,
    object_id: int = 0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    """Reward the agent for keeping the object balanced."""
    objects: RigidObjectCollection = env.scene[object_cfg.name]
    current_quat = objects.data.object_link_quat_w[:, object_id, :]
    default_quat = objects.data.default_object_state[:, object_id, 3:7]
    lin_vel_w = objects.data.object_link_vel_w[:, object_id, :3]
    ang_vel_w = objects.data.object_link_ang_vel_w[:, object_id, :3]
    # 1. TILT STABILITY - Compare current orientation with default orientation
    # Calculate relative rotation between current and default pose
    dot_product = torch.sum(current_quat * default_quat, dim=-1)  # Quaternion dot product
    angle_diff = 2 * torch.acos(torch.clamp(torch.abs(dot_product), 0, 1))  # Rotation angle
    max_tilt = torch.pi / 3  # Allow up to 60 degrees tilt
    tilt_score = torch.clamp(1 - angle_diff / max_tilt, 0, 1)
    
    # 2. HORIZONTAL DRIFT - Sideways motion during lifting
    horizontal_speed = torch.norm(lin_vel_w[..., :2], dim=-1)  # x,y velocity magnitude
    drift_score = torch.clamp(1 - horizontal_speed / 0.3, 0, 1)  # Penalize >30cm/s drift
    
    # 3. ANGULAR STABILITY - Spinning/rotating motion
    angular_speed = torch.norm(ang_vel_w, dim=-1)  # Total rotation rate
    rotation_score = torch.clamp(1 - angular_speed / 1.0, 0, 1)  # Penalize >1 rad/s
    
    # Weighted combination (tilt is most important)
    balance_score = 0.5 * tilt_score + 0.3 * drift_score + 0.2 * rotation_score
    return balance_score

def avoid_collision_score(
    env: ManagerBasedRLEnv,
    aim_object_id: int = 0,
    disturbance_threshold: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    """
    Simple collision avoidance score by measuring unwanted movement in other objects.
    
    Args:
        object_data: IsaacLab object data container
        aim_object_id: Index of the target object being manipulated (int)
        disturbance_threshold: Maximum allowed displacement for other objects (meters, default: 0.1)
    
    Returns:
        torch.Tensor: Collision avoidance score (0-1, higher is better)
                     Shape: (num_instances, 1) or (num_instances,)
    """
    objects: RigidObjectCollection = env.scene[object_cfg.name]
    # Get object positions and velocities
    current_pos = objects.data.object_com_pos_w      # (num_instances, num_objects, 3)
    default_pos = objects.data.default_object_state[..., :3]  # Default positions
    velocities = objects.data.object_com_lin_vel_w   # (num_instances, num_objects, 3)
    
    # Calculate displacement from default positions
    displacement = torch.norm(current_pos - default_pos, dim=-1)  # (num_instances, num_objects)
    
    # Calculate current velocity magnitudes  
    velocity_magnitude = torch.norm(velocities, dim=-1)  # (num_instances, num_objects)
    
    # Exclude the target object from evaluation
    num_objects = current_pos.shape[1]
    other_objects_mask = torch.ones(num_objects, dtype=torch.bool, device=current_pos.device)
    other_objects_mask[aim_object_id] = False
    
    # Get max disturbance among other objects
    other_displacement = displacement[:, other_objects_mask]  # Exclude target object
    other_velocity = velocity_magnitude[:, other_objects_mask]  # Exclude target object
    
    if other_displacement.numel() == 0:  # No other objects
        return torch.ones(current_pos.shape[0], device=current_pos.device)
    
    max_displacement = other_displacement.max(dim=-1)[0]  # Max displacement among others
    max_velocity = other_velocity.max(dim=-1)[0]          # Max velocity among others
    
    # Score based on displacement (70% weight) and velocity (30% weight)
    displacement_score = torch.clamp(1 - max_displacement / disturbance_threshold, 0, 1)
    velocity_score = torch.clamp(1 - max_velocity / 0.05, 0, 1)  # 5cm/s threshold
    
    # Combined score
    collision_avoidance_score = 0.7 * displacement_score + 0.3 * velocity_score
    
    return collision_avoidance_score




# def avoid_collision(
#     env: ManagerBasedRLEnv,
#     min_distance: float,
#     aim_object_id: int = 0,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
# ) -> torch.Tensor:
#     """Reward the agent for avoiding collision with the object."""
#     objects: RigidObjectCollection = env.scene[object_cfg.name]
#     object_positions = objects.data.object_link_pos_w[:, :, :3]
#     initial_object_positions = objects.data.default_object_state[:, :, :3]
#     # except the aim object, check the movement of the other objects is less than the min_distance
#     num_envs, num_obt add grasp_env/mdp/rewards.pyjects, _ = object_positions.shape
#     mask = torch.ones(num_objects, dtype=torch.bool, device=object_positions.device)
#     mask[aim_object_id] = Falset add grasp_env/mdp/rewards.py
#     # Compute movement for all objects except the aim object
#     movement = torch.norm(object_positions - initial_object_positions, dim=-1)  # (num_envs, num_objects)
#     movement_except_aim = movement[:, mask]  # (num_envs, num_objects-1)
#     # If all other objects moved less than min_distance, reward 1.0, else 0.0
#     reward = (movement_except_aim < min_distance).all(dim=-1).float()  # (num_envs,)
#     return reward
    # except the aim object, check the movement of the other objects is less than the min_distance


def get_current_smoothness(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Get current joint acceleration smoothness (instantaneous).
    
    Returns:
        Current smoothness score [num_envs]
    """
    robot = env.scene[robot_cfg.name]
    current_joint_acc = robot.data.joint_acc
    
    # RMS of current joint accelerations
    joint_acc_rms = torch.sqrt(torch.mean(current_joint_acc**2, dim=1))
    
    # Convert to smoothness score
    smoothness_score = 1.0 / (1.0 + joint_acc_rms)
    
    return smoothness_score