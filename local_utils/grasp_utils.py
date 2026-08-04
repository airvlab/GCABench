import numpy as np
import torch
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix



def transformation_matrix_to_pose_vector(transformation_matrix):
    rot_matrix = transformation_matrix[:3, :3]
    pose_vec = torch.cat(
        (
            transformation_matrix[:3, 3],
            quat_from_matrix(rot_matrix).squeeze(),
        )
    )
    return pose_vec

def from_rectangle_to_6D(camera_pose, grasp_config, pcd):
    x, y, w, h, angle = grasp_config
    gripper_pos= pcd[x,y]
        # Convert angle to radians and adjust (same as your reference)
    angle = (angle - 90) * np.pi / 180
    
    # Define closing direction in camera frame
    closing_cam = np.array([np.cos(angle), np.sin(angle), 0])
    
    # Define approaching direction in camera frame (downward)
    approaching_cam = np.array([0, 0, -1])
    R_cam2world = camera_pose[:3, :3]
    # Convert directions to world frame
    closing_world = R_cam2world @ closing_cam
    approaching_world = R_cam2world @ approaching_cam
    ortho_world = np.cross(closing_world, approaching_world)
    # Calculate gripper pose in world frame

    T_gripper = np.eye(4)
    T_gripper[:3, :3] = np.stack([ortho_world, closing_world, approaching_world], axis=1)
    T_gripper[:3, 3] = gripper_pos
    ee_pose = transformation_matrix_to_pose_vector(T_gripper)

    return ee_pose


def from_contact_to_6D(self, grasp_config, obj_to_world_transform=None):
    """
    Convert from hdf5 contact representation to 4x4 Matrix in World COS.
    """
    vec_a = np.array([grasp_config[0],
                            grasp_config[1],
                            grasp_config[2]])

    vec_b = np.array([grasp_config[3],
                            grasp_config[4],
                            grasp_config[5]])

    contact_pt = np.array([grasp_config[6],
                                    grasp_config[7],
                                    grasp_config[8]])

    width = grasp_config[9]
    print("parallel width", width)

    c_ = np.cross(vec_a, vec_b)
    # # rotation matrix
    R_ = [[vec_b[0], vec_a[0], c_[0]],
        [vec_b[1], vec_a[1], c_[1]],
        [vec_b[2], vec_a[2], c_[2]]]

    # translation t
    t_ = contact_pt + width/2 * vec_b + self.dis_gripper* vec_a * (-1)
    # create 4x4 transform matrix of grasp
    pregrasp_transform_ = [[R_[0][0], R_[0][1], R_[0][2], t_[0]],
                        [R_[1][0], R_[1][1], R_[1][2], t_[1]],
                        [R_[2][0], R_[2][1], R_[2][2], t_[2]],
                        [0.0, 0.0, 0.0, 1.0]]
    
    return np.array(pregrasp_transform_)


