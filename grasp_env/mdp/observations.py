# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
import torch
from typing import TYPE_CHECKING
from isaaclab.sensors import FrameTransformer
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.assets import RigidObjectCollection

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.utils.math import transform_points, unproject_depth
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
N_MULTIPLE_CAM = 3
CAM_WIDTH = 640
CAM_HEIGHT = 480
DEPTH_MAX = 50.0  
def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    # object: RigidObject = env.scene[object_cfg.name]
    objects: RigidObjectCollection = env.scene[object_cfg.name]
    # object: RigidObject = env.scene[objects.object_names[0]]  
    # object_state = objects.data.default_object_state.clone()
    # object_state[..., :3] += env.scene.env_origins.unsqueeze(1)
    # object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_w = objects.data.object_link_pos_w[:, 0, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b

def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos

def ee_pose_with_gripper(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"), robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]
    # binary gripper pos
    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
    open_threshold = 0.039
    ee_frame_gripper_pos = torch.where(
        abs(finger_joint_1) > open_threshold,
        torch.tensor(1.0, device=ee_frame_pos.device),  # Open
        torch.tensor(-1.0, device=ee_frame_pos.device)  # Closed
    )
    ee_frame_pose = torch.cat([ee_frame_pos, ee_frame_quat, ee_frame_gripper_pos], dim=1)

    return ee_frame_pose

def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat


def gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Obtain the versatile gripper position of both Gripper and Suction Cup.
    """
    robot: Articulation = env.scene[robot_cfg.name]

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        # Handle multiple surface grippers by concatenating their states
        gripper_states = []
        for gripper_name, surface_gripper in env.scene.surface_grippers.items():
            gripper_states.append(surface_gripper.state.view(-1, 1))

        if len(gripper_states) == 1:
            return gripper_states[0]
        else:
            return torch.cat(gripper_states, dim=1)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Observation gripper_pos only support parallel gripper for now"
            finger_joint_1 = robot.data.joint_pos[:, gripper_joint_ids[0]].clone().unsqueeze(1)
            finger_joint_2 = -1 * robot.data.joint_pos[:, gripper_joint_ids[1]].clone().unsqueeze(1)
            return torch.cat((finger_joint_1, finger_joint_2), dim=1)
        else:
            raise NotImplementedError("[Error] Cannot find gripper_joint_names in the environment config")


def cam_pos(env: ManagerBasedEnv, camera_cfg: SceneEntityCfg = SceneEntityCfg("camera_0")) -> torch.Tensor:
    """The position of the camera in the world frame."""
    # extract the used quantities (to enable type-hinting)
    camera: RigidObject = env.scene[camera_cfg.name]
    position=camera.data.pos_w
    orientation=camera.data.quat_w_ros
    cam_pose = torch.cat([position, orientation], dim=-1)
    return cam_pose




def joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]








def rgb_capture(env: ManagerBasedEnv):
    """Height scan from the given sensor w.r.t. the sensor's frame."""
    # extract the used quantities (to enable type-hinting)
    rgb = [env.scene[f"camera_{i}"].data.output["rgb"][..., :3].float() for i in range(N_MULTIPLE_CAM)]
    rgbs = torch.stack(rgb, 0).transpose(0, 1)
    return rgbs


def table_rgb_capture(env: ManagerBasedEnv):
    """Height scan from the given sensor w.r.t. the sensor's frame."""
    # extract the used quantities (to enable type-hinting)
    rgb = env.scene[f"camera_{1}"].data.output["rgb"][..., :3].float()
    rgbs = torch.stack(rgb, 0).transpose(0, 1)
    return rgbs

def wrist_rgb_capture(env: ManagerBasedEnv):
    """Height scan from the given sensor w.r.t. the sensor's frame."""
    # extract the used quantities (to enable type-hinting)
    rgb = env.scene[f"camera_{0}"].data.output["rgb"][..., :3].float()
    rgbs = torch.stack(rgb, 0).transpose(0, 1)
    return rgbs


def image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type]

    # depth image conversion
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    # rgb/depth image normalization
    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0

    return images.clone()

def normal_capture(env: ManagerBasedEnv):
    """Height scan from the given sensor w.r.t. the sensor's frame."""
    # extract the used quantities (to enable type-hinting)
    # data=env.scene['camera_0'].data.output["normals"][..., :3]
    data = [env.scene[f"camera_{i}"].data.output["normals"][..., :3] for i in range(N_MULTIPLE_CAM)]
    data_norms = torch.stack(data, 0).transpose(0, 1)
    return data_norms


def inst_capture(env:ManagerBasedEnv):
    """Height scan from the given sensor w.r.t. the sensor's frame."""
    # extract the used quantities (to enable type-hinting)
    data = [env.scene[f"camera_{i}"].data.output["instance_segmentation_fast"].unsqueeze(-1) for i in range(N_MULTIPLE_CAM)]
    # data = env.scene['camera_0'].data.output["instance_segmentation_fast"].unsqueeze(-1)
    data_insts = torch.stack(data, 0).transpose(0, 1)
    return data_insts


def depth_capture(env:ManagerBasedEnv):
    """Height scan from the given sensor w.r.t. the sensor's frame."""
    # extract the used quantities (to enable type-hinting)
    data = [env.scene[f"camera_{i}"].data.output["distance_to_image_plane"] for i in range(N_MULTIPLE_CAM)]
    # Maximum representable value for the tensor's dtype
    # Clip positive infinity to maximum value
    # depth_data = env.scene['camera_0'].data.output["distance_to_image_plane"]
    # data = torch.clip(depth_data, 0., DEPTH_MAX)
    depth_data = torch.stack(data, 0).transpose(0, 1)
    # depth_data = torch.clip(torch.stack(data, 0).transpose(0, 1), 0.,DEPTH_MAX)
    return depth_data 


def pcd_capture(env:ManagerBasedEnv):
    # pcds = []
    # depths = env.scene["camera_0"].data.output["distance_to_image_plane"]#
    # depths = torch.clip(depths, 0., DEPTH_MAX)

    # pointcloud = create_pointcloud_from_depth(
    #     intrinsic_matrix=env.scene["camera_0"].data.intrinsic_matrices[0],
    #     depth=depths[0],
    #     position=env.scene["camera_0"].data.pos_w[0],
    #     orientation=env.scene["camera_0"].data.quat_w_ros[0],
    #     device=env.device
    # )
    # pcds=[]
    # reshaped_pcd = pointcloud.view(CAM_WIDTH, CAM_HEIGHT, 3)

    pcds = []
    for i in range(N_MULTIPLE_CAM):
        depths = env.scene[f"camera_{i}"].data.output["distance_to_image_plane"]#
        depths = torch.clip(depths, 0., DEPTH_MAX)

        pointcloud = create_pointcloud_from_depth(
            intrinsic_matrix=env.scene[f"camera_{i}"].data.intrinsic_matrices[0],
            depth=depths[0],
            position=env.scene[f"camera_{i}"].data.pos_w[0],
            orientation=env.scene[f"camera_{i}"].data.quat_w_ros[0],
            device=env.device
        )
        reshaped_pcd = pointcloud.view(CAM_WIDTH, CAM_HEIGHT, 3).permute(1, 0, 2)
        pcds.append(reshaped_pcd.unsqueeze(0))
    pcds = torch.stack(pcds, 0).transpose(0, 1)
    return pcds





