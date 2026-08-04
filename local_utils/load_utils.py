# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import yaml
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
import isaaclab.sim as sim_utils
import os
import json
HOME_PATH = Path(os.getcwd())

class ObjectConfigLoader:
    """Loads object configurations from simplified YAML and creates Isaac Lab RigidObjectCfg."""
    
    def __init__(self, config_path: str = str(HOME_PATH)+"/tasks/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load the YAML configuration file."""
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def create_object_cfg(self, obj_config: Dict[str, Any], base_pos: List[float] = [0.5, 0, 0.2], 
                         prim_path: str = "{ENV_REGEX_NS}/Object") -> RigidObjectCfg:
        """
        Create a single RigidObjectCfg from task configuration.
        
        Args:
            task_config: Task configuration dictionary
            base_pos: Base position [x, y, z]
            prim_path: Prim path for the object
            
        Returns:
            Configured RigidObjectCfg
        """
        # Calculate final position (base_pos + object's pos offset)
        final_pos = [
            base_pos[0] + obj_config['pos'][0],
            base_pos[1] + obj_config['pos'][1], 
            base_pos[2] + obj_config['pos'][2]
        ]
        
        obj_path = os.path.join(HOME_PATH, obj_config['object_path'])
        
        OBJ_CFG = RigidObjectCfg(
            prim_path=prim_path,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=final_pos,
                rot=obj_config['rot']
            ),
            spawn=UsdFileCfg(
                usd_path=obj_path,
                scale=(1, 1, 1),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        return OBJ_CFG

class TaskBuilder:
    def __init__(self, config_path: str = str(HOME_PATH)+"/tasks/config.yaml", grasp_pose_dir: str = str(HOME_PATH)+"/models/transfered_grasps/franka_panda", task_id: str = "a01"):
        self.config_path = config_path
        self.grasp_pose_dir = grasp_pose_dir
        self.config = self._load_config()
        self.base_pos = self.config['scene_setup']['base_pose']
        self.task_id = task_id

        if self.task_id not in self.config['tasks']:
            raise ValueError(f"Task ID '{self.task_id}' not found in config")
        else:
            self.task_config = self.config['tasks'][self.task_id]
            self.task_class = self.task_config['task_class']
            self.objects = self.task_config['objects']
            self.task_instruction = self.task_config['instruction']
            self.objects_grasp_pose = []


    def _load_config(self) -> Dict[str, Any]:
        """Load the YAML configuration file."""
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def load_object_grasp_pose(self) -> List[float]:
        """
        Get the grasp pose for a specific object.
        """
        # models/graspit_grasps/franka_panda/franka_panda-002_master_chef_can.json
        # the .json file path is grasp_pose_dir +franka_panda- + object_name + .json
        
        for obj in self.objects:
            object_name = obj['name']
            grasp_pose_file = os.path.join(self.grasp_pose_dir, f"franka_panda-{object_name}.json")
            with open(grasp_pose_file, 'r') as file:
                obj_data = json.load(file)
                poses = obj_data['pose']
        # Save all objects' grasp poses to self.objects_grasp_pose with their names
        # Each entry will be a dict: {'name': object_name, 'poses': poses}
        self.objects_grasp_pose.append({'name': object_name, 'poses': poses})
        return self.objects_grasp_pose
    
    def get_object_grasp_pose(self, obj_cfg: Dict[str, Any]) -> List[float]:
        object_name = obj_cfg['name']
        grasp_pose_file = os.path.join(self.grasp_pose_dir, f"franka_panda-{object_name}.json")
        with open(grasp_pose_file, 'r') as file:
            obj_data = json.load(file)
            poses = obj_data['pose']
        return poses 
    
    def get_ee_pose_for_grasping(self, obj_cfg: Dict[str, Any]) -> List[float]:
        task_builder = TaskBuilder(task_id="z03")
        object_cfg = task_builder.task_config['objects'][0]
        # object_pose = pos,rot
        base_pos = task_builder.base_pos
        base_pos = torch.tensor(base_pos, device=device)
        object_pose = object_cfg['pos'] + object_cfg['rot']
        object_pose = torch.tensor(object_pose, device=device)
        # object_rot(w,x,y,z)
        # grasp_pose(x,y,z,w,x,y,z)
        grasp_poses = task_builder.get_object_grasp_pose(object_cfg)
        ee_poses = []
        for grasp_pose in grasp_poses:
            grasp_pose = torch.tensor(grasp_pose, device=device)
            ee_pos = grasp_pose[:3] + object_pose[:3]+base_pos
            obj_matrix = matrix_from_quat(object_pose[3:])
            grasp_matrix = matrix_from_quat(grasp_pose[3:])
            # comput the ee_pose in the world_frame#
            ee_matrix = obj_matrix @ grasp_matrix
            ee_quat = quat_from_matrix(ee_matrix)
            ee_pose = torch.cat([ee_pos, ee_quat], dim=0)
            ee_poses.append(ee_pose)
    
    def create_object_cfg(self, obj_config: Dict[str, Any], 
                         prim_path: str = "{ENV_REGEX_NS}/Object") -> RigidObjectCfg:
        """
        Create a single RigidObjectCfg from task configuration.
        
        Args:
            task_config: Task configuration dictionary
            base_pos: Base position [x, y, z]
            prim_path: Prim path for the object
            
        Returns:
            Configured RigidObjectCfg
        """
        # Calculate final position (base_pos + object's pos offset)
        final_pos = [
            self.base_pos[0] + obj_config['pos'][0],
            self.base_pos[1] + obj_config['pos'][1], 
            self.base_pos[2] + obj_config['pos'][2]
        ]
        
        obj_path = os.path.join(HOME_PATH, obj_config['object_path'])
        
        OBJ_CFG = RigidObjectCfg(
            prim_path=prim_path,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=final_pos,
                rot=obj_config['rot']
            ),
            spawn=UsdFileCfg(
                usd_path=obj_path,
                scale=(1, 1, 1),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        return OBJ_CFG


    def create_scene_objects(self, num_objs: int = 1) -> RigidObjectCollectionCfg:
        """
        Create a collection of RigidObjectCfg for the specified task_id.
        
        Args:
            task_id: Task ID like 'a01', 'b02', etc.
            num_objs: Number of objects to create (for stack tasks)
            base_pos: Base position [x, y, z]
            
        Returns:
            Configured RigidObjectCollectionCfg
        """
        loader = ObjectConfigLoader()

        task_class = self.task_config['task_class']
        rigid_objects = {}
        
        if task_class == 'stack':
            # Stack task - create multiple objects with offsets
            obj_config = self.task_config['objects'][0]
            object_cfg = self.create_object_cfg(obj_config)
            
            # For stack tasks, we need to add vertical offset between objects
            # stack_offset = task_config.get('stack_offset', [0.0, 0.0, 0.1])  # Default vertical stack
            stack_offset = self.task_config['diff']
            
            for i in range(num_objs):
                # Calculate position with stack offset
                stack_pos = [
                    self.base_pos[0] + obj_config['pos'][0] + stack_offset[0] * i,
                    self.base_pos[1] + obj_config['pos'][1] + stack_offset[1] * i,
                    self.base_pos[2] + obj_config['pos'][2] + stack_offset[2] * i
                ]
                
                stacked_obj = object_cfg.replace(
                    init_state=object_cfg.init_state.replace(pos=stack_pos),
                    prim_path=f"{{ENV_REGEX_NS}}/obj_{i}"
                )
                rigid_objects[f"obj_{i}"] = stacked_obj
            
            return RigidObjectCollectionCfg(rigid_objects=rigid_objects)

        else:
            for i, obj_config in enumerate(self.task_config['objects']):
                object_cfg = self.create_object_cfg(obj_config, f"{{ENV_REGEX_NS}}/obj_{i}")
                objs = object_cfg.replace(
                    prim_path=f"{{ENV_REGEX_NS}}/obj_{i}"
                )
                rigid_objects[f"obj_{i}"] = objs
                
            return RigidObjectCollectionCfg(rigid_objects=rigid_objects)



# Example usage functions
def get_random_task_objects(task_class: str, num_objs: int = 1) -> RigidObjectCollectionCfg:
    """Get objects for a random task from a specific class."""
    loader = ObjectConfigLoader()
    task_id = loader.sample_random_task(task_class)
    return create_scene_objects(task_id, num_objs)



