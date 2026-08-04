# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to test GraspVLA integration with Isaac Lab environment in closed-loop control."""

import argparse
import json
from collections import deque
import collections
import cv2
import gymnasium as gym
import numpy as np
import pickle
import torch
import transforms3d as t3d
import zmq

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Test GraspVLA closed-loop control.")
# parser.add_argument("--task", type=str, default="Grasp-Franka-IK-Rel-v0", help="Name of the task.")
parser.add_argument("--task", type=str, default='Grasp-UR10-short-Grasp-vla', help="Name of the task.")

parser.add_argument("--task_id", type=str, default="a01", help="Task ID in task config")
parser.add_argument("--num_objects", type=int, default=4, help="Number of objects in scene")
parser.add_argument("--num_demos", type=int, default=20, help="Number of demonstrations")
parser.add_argument("--graspvla_port", type=str, default="6666", help="GraspVLA server port")
parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode")
parser.add_argument("--replan_steps", type=int, default=8, help="Number of steps before replanning")
parser.add_argument("--seed", type=int, default=42, help="Seed to use")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.num_envs = 1

# Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Post-launch imports
import grasp_env  # noqa: F401
import isaaclab_mimic.envs  # noqa: F401
import isaaclab_tasks  # noqa: F401

from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from local_utils.load_utils import TaskBuilder


class GraspVLAClient:
    """Client for GraspVLA server communication."""

    PROPRIO_HISTORY_SIZE = 4
    GRIPPER_OPEN_THRESHOLD = 0.039

    def __init__(self, port: str = "6666"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{port}")
        self.task = args_cli.task
        print(f"[INFO]: Connected to GraspVLA server at port {port}")

    def get_actions(self, sample: dict) -> tuple[list, dict]:
        """Get actions from GraspVLA server."""
        self.socket.send(pickle.dumps(sample))
        reply = pickle.loads(self.socket.recv())

        if reply["info"] == "success":
            actions = reply["result"]
            debug_info = reply.get("debug", {})
            if "bbox" in debug_info:
                print(f"debug_info bbox: {debug_info['bbox']}")
            return actions, debug_info
        else:
            raise RuntimeError(f"GraspVLA server error: {reply}")

    def prepare_input(self, obs: dict, state_history: deque, instruction: str) -> dict:
        """Prepare input dictionary for GraspVLA server."""
        # Extract and process images
        front_rgb = obs["Image_info"]["rgb"][0, 1, :, :, :].cpu().numpy().astype(np.uint8)
        side_rgb = obs["Image_info"]["rgb"][0, 2, :, :, :].cpu().numpy().astype(np.uint8)

        front_rgb = np.expand_dims(self._crop_and_resize(front_rgb), 0)
        side_rgb = np.expand_dims(self._crop_and_resize(side_rgb), 0)

        # Build proprio array from state history
        proprio_array = [self._eef_pose_to_proprio(state) for state in state_history]

        print(f"proprio_array: {proprio_array[-1]}")
        return {
            "text": instruction,
            "front_view_image": front_rgb,
            "side_view_image": side_rgb,
            "proprio_array": proprio_array,
        }

    def _eef_pose_to_proprio(self, eef_pose: torch.Tensor) -> np.ndarray:
        """Convert EEF pose tensor to GraspVLA proprio format."""
        eef_pos = eef_pose[:, 0:3]
        eef_quat = eef_pose[:, 3:7]
        eef_gripper_pos = eef_pose[:, 7:8]

        # Convert quaternion to euler
        eef_euler = euler_xyz_from_quat(eef_quat)
        eef_euler_tensor = torch.cat(eef_euler, dim=0).unsqueeze(0)

        # Binarize gripper state in our sim, close is 1, open is -1, in graspvla, close is -1, open is 1
        gripper_binary = torch.where(
            abs(eef_gripper_pos) > self.GRIPPER_OPEN_THRESHOLD,
            torch.tensor(1.0, device=eef_gripper_pos.device),
            torch.tensor(-1.0, device=eef_gripper_pos.device),
        )
        if self.task == "Grasp-UR10-short-Grasp-vla":
            gripper_binary = -eef_gripper_pos
        proprio = torch.cat([eef_pos, eef_euler_tensor, gripper_binary], dim=1)
        # print(f"proprio: {proprio}")
        return proprio.cpu().numpy()[0]

    def _crop_and_resize(self, image: np.ndarray, target_size: int = 256) -> np.ndarray:
        """Center crop to square and resize."""
        assert image.shape[:2] == (480, 640), f"Expected (480, 640), got {image.shape[:2]}"
        
        h, w = image.shape[:2]
        start_x = (w - h) // 2
        cropped = image[:, start_x : start_x + h, :]
        return cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

    def close(self):
        """Clean up ZMQ resources."""
        self.socket.close()
        self.context.term()


def check_success(env, min_height: float = 0.05) -> torch.Tensor:
    """Check if object has been lifted above threshold."""
    object_heights = (
        env.scene["objects"].data.object_link_pos_w[:, :, 2]
        - env.scene["objects"].data.default_object_state[:, :, 2]
    )
    return (object_heights > min_height).any(dim=1)


def get_eef_state(obs: dict) -> torch.Tensor:
    """Extract end-effector state from observation."""
    return torch.cat(
        [obs["policy"]["eef_pos"], obs["policy"]["eef_quat"], obs["policy"]["gripper_pos"]],
        dim=1,
    )


def run_episode(
    env,
    client: GraspVLAClient,
    max_steps: int = 50,
    replan_steps: int = 8,
) -> dict:
    """Run closed-loop control episode with action chunking."""
    total_steps = 0
    query_count = 0

    # State history buffer (last 4 states)
    state_history: deque = deque(maxlen=4)
    
    # Action plan buffer
    action_plan: deque = deque()

    # Setup visualization marker
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))

    # Reset environment
    obs, _ = env.reset()

    # Resolve robot entity
    # robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    # robot_entity_cfg.resolve(env.scene)

    # Stabilization period - fill state history
    wait_action = torch.zeros((1, 7), device=env.device)
    wait_action[0, 6] = 1
    for _ in range(5):
        obs, _, _, _, _ = env.step(wait_action)
        state_history.append(get_eef_state(obs))
        total_steps += 1

    print(f"[INFO]: Starting closed-loop control (max_steps={max_steps})")
    last_gripper = 1
    while total_steps < max_steps:
        # Check if episode is done
        success = check_success(env)[0]
        if success:
            print(f"Episode finished at step {total_steps}. Success: {success}")
            break

        try:
            # Query GraspVLA if action plan is empty
            if not action_plan:
                graspvla_input = client.prepare_input(obs, state_history, env.instruction)
                # print(f"graspvla_input: {graspvla_input['proprio_array'][-1]}")
                actions, debug_info = client.get_actions(graspvla_input)
                query_count += 1

                assert len(actions) >= replan_steps, (
                    f"Want to replan every {replan_steps} steps, "
                    f"but policy only predicts {len(actions)} steps."
                )
                action_plan.extend(actions[:replan_steps])
                print(f"\n--- Query {query_count} (Step {total_steps}) ---")

            # Pop and execute single action
            action = action_plan.popleft()
            
            if abs(action[6]) < 1e-6: 
                gripper_action = last_gripper
            else:
                gripper_action = action[6]
                last_gripper = gripper_action
            # print(f"gripper_action: {gripper_action}", f"original: {action[6:7]}")
            print(f"last_gripper: {last_gripper}, gripper_action: {action[6]}")
            if total_steps >= max_steps:
                break

            # Convert euler to axis-angle
            axangle_ax, axangle_angle = t3d.euler.euler2axangle(*action[3:6])
            action_delta_pose = np.concatenate([
                action[0:3],
                axangle_ax * axangle_angle,
                np.array([gripper_action]),
            ])
            # last_gripper = action[6:7]
            action_tensor = torch.from_numpy(action_delta_pose).float().unsqueeze(0)

            # Step environment
            obs, _, _, _, _ = env.step(action_tensor)
            current_state = get_eef_state(obs)
            state_history.append(current_state)
            total_steps += 1

            # Update visualization
            ee_marker.visualize(current_state[:, 0:3], current_state[:, 3:7])

        except Exception as e:
            print(f"Error during query {query_count}: {e}")
            break

    final_success = check_success(env)[0].item()
    return {
        "success": final_success,
        "total_steps": total_steps,
        "queries": query_count,
    }


def main():
    """Main execution function."""
    client = GraspVLAClient(port=args_cli.graspvla_port)

    # Setup environment config
    # Setup environment
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    # Configure task
    task_builder = TaskBuilder(task_id=args_cli.task_id)
    TG_CFGs = task_builder.create_scene_objects(num_objs=args_cli.num_objects)
    instruction = task_builder.task_instruction
    env_cfg.scene.objects = TG_CFGs
    # Configure environment
    env_cfg.terminations.time_out = None
    if args_cli.task_id.startswith("a"):
        env_cfg.scene.table.spawn.scale = [0.7, 1.0, 1.0] 
    
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.seed(args_cli.seed)
    # instruction = "grab the craker box"
    instruction = task_builder.task_instruction
    env.instruction = instruction

    # Run demonstrations
    results = []
    success_count = 0

    print(f"[INFO]: Running {args_cli.num_demos} closed-loop demonstrations")
    print(f"[INFO]: Max steps: {args_cli.max_steps}, Replan steps: {args_cli.replan_steps}")

    for demo_idx in range(args_cli.num_demos):
        print(f"\n{'='*50}")
        print(f"Demo {demo_idx + 1}/{args_cli.num_demos}")
        print(f"{'='*50}")

        try:
            result = run_episode(
                env=env,
                client=client,
                max_steps=args_cli.max_steps,
                replan_steps=args_cli.replan_steps,
            )
            result["demo_idx"] = demo_idx

            if result["success"]:
                success_count += 1
                print(f"✓ Demo {demo_idx + 1} SUCCESS in {result['total_steps']} steps")
            else:
                print(f"✗ Demo {demo_idx + 1} FAILED after {result['total_steps']} steps")

            results.append(result)

        except Exception as e:
            print(f"Error in demo {demo_idx + 1}: {e}")
            results.append({
                "demo_idx": demo_idx,
                "success": False,
                "error": str(e),
                "total_steps": 0,
                "queries": 0,
            })

    # Print summary
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Success rate: {success_count}/{args_cli.num_demos} ({100*success_count/args_cli.num_demos:.1f}%)")

    successful_results = [r for r in results if r.get("success")]
    if successful_results:
        avg_steps = np.mean([r["total_steps"] for r in successful_results])
        avg_queries = np.mean([r["queries"] for r in successful_results])
        print(f"Average steps (successful): {avg_steps:.1f}")
        print(f"Average queries (successful): {avg_queries:.1f}")

    # Save results
    output_file = f"graspvla_closedloop_{args_cli.task_id}_o{args_cli.num_objects}_d{args_cli.num_demos}.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "task_id": args_cli.task_id,
                "num_objects": args_cli.num_objects,
                "num_demos": args_cli.num_demos,
                "max_steps": args_cli.max_steps,
                "replan_steps": args_cli.replan_steps,
                "success_rate": success_count / args_cli.num_demos,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"Results saved to: {output_file}")

    client.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()