import requests
import json_numpy
import collections
json_numpy.patch()
import numpy as np
import torch
import transforms3d as t3d
import argparse
import gymnasium as gym
import zmq
import pickle
import json
import time
import cv2
from rich.table import Table 
# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher
 
import dataclasses
import enum
import logging
import pathlib
 
import numpy as np
import polars as pl
import rich
import tqdm
import matplotlib.pyplot as plt
 
# Parse arguments FIRST - before any Isaac Lab imports
parser = argparse.ArgumentParser(description="Test OpenVLA closed-loop control.")
parser.add_argument("--task", type=str, default='Grasp-Franka-IK-Rel-img', help="Name of the task.")
parser.add_argument("--task_id", type=str, default='a01', help="Task ID in task config")
parser.add_argument("--num_objects", type=int, default=4, help="Number of objects in scene")
parser.add_argument("--num_demos", type=int, default=20, help="Number of demonstrations")
parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode")
parser.add_argument("--actions_per_query", type=int, default=10, help="Number of actions to execute per OpenVLA query")
parser.add_argument("--seed", type=int, default=42, help="Seed to use")
parser.add_argument("--host", type=str, default="localhost", help="Host to connect to the OpenVLA server")
parser.add_argument("--port", type=int, default=8777, help="Port to connect to the OpenVLA server")
parser.add_argument("--unnorm_key", type=str, default='gca_parallel', help="Dataset key for action denormalization")
parser.add_argument("--gripper_id", type=int, default=0, help="Gripper ID (0 for parallel-jaw, 1 for suction, etc.)")
parser.add_argument("--timing_file", type=str, default=None, help="Path to save timings (parquet)")
 
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.num_envs = 1
args_cli.device = "cuda"

'''
Example usage:
python Openvla_client_img.py \
  --host localhost \
  --port 6000 \
  --unnorm_key gca_parallel \
  --task Grasp-Franka-IK-Rel-img \
  --num_demos 5
'''
 
# Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
 
# Additional imports after simulator launch
import isaaclab_mimic.envs
import grasp_env
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from local_utils.load_utils import TaskBuilder
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from rich.console import Console
import math

logger = logging.getLogger(__name__)


class OpenVLAHttpPolicy:
    """HTTP client for OpenVLA server (deploy.py)."""

    def __init__(self, host: str, port: int, unnorm_key: str | None = None, gripper_id: int = 0):
        self.url = f"http://{host}:{port}/act"
        self.unnorm_key = unnorm_key
        self.gripper_id = gripper_id
        print(f"[OpenVLA HTTP Client] Connecting to {self.url}")
        print(f"[OpenVLA HTTP Client] unnorm_key={unnorm_key}, gripper_id={gripper_id}")

    def infer(self, obs: dict) -> dict:
        """Send observation to server and get action prediction."""
        # Add unnorm_key and gripper_id to the request
        if self.unnorm_key:
            obs["unnorm_key"] = self.unnorm_key
        obs["gripper_id"] = self.gripper_id

        response = requests.post(self.url, json=obs)
        if response.status_code != 200:
            raise RuntimeError(f"Server error: {response.status_code} - {response.text}")

        action = response.json()

        # Handle error response
        if isinstance(action, dict) and "error" in action:
            raise RuntimeError(f"Server error: {action['error']}")

        # Wrap single action as action chunk for compatibility
        # The server returns a single action, but the loop expects {"actions": [...]}
        return {"actions": [action]}

    def get_server_metadata(self) -> dict:
        """Return server metadata."""
        return {"type": "openvla", "url": self.url}


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle representation."""
    # Clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
 
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation
        return np.zeros(3)
 
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
 
 
class TimingRecorder:
    """Records timing measurements for different keys."""
 
    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}
 
    def record(self, key: str, time_ms: float) -> None:
        """Record a timing measurement for the given key."""
        if key not in self._timings:
            self._timings[key] = []
        self._timings[key].append(time_ms)
 
    def get_stats(self, key: str) -> dict[str, float]:
        """Get statistics for the given key."""
        times = self._timings[key]
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "p25": float(np.quantile(times, 0.25)),
            "p50": float(np.quantile(times, 0.50)),
            "p75": float(np.quantile(times, 0.75)),
            "p90": float(np.quantile(times, 0.90)),
            "p95": float(np.quantile(times, 0.95)),
            "p99": float(np.quantile(times, 0.99)),
        }
 
    def print_all_stats(self) -> None:
        """Print statistics for all keys."""
        table = Table(  # FIXED: Use imported Table class
            title="[bold blue]Timing Statistics[/bold blue]",
            show_header=True,
            header_style="bold white",
            border_style="blue",
            title_justify="center",
        )
 
        table.add_column("Metric", style="cyan", justify="left", no_wrap=True)
 
        stat_columns = [
            ("Mean", "yellow", "mean"),
            ("Std", "yellow", "std"),
            ("P25", "magenta", "p25"),
            ("P50", "magenta", "p50"),
            ("P75", "magenta", "p75"),
            ("P90", "magenta", "p90"),
            ("P95", "magenta", "p95"),
            ("P99", "magenta", "p99"),
        ]
 
        for name, style, _ in stat_columns:
            table.add_column(name, justify="right", style=style, no_wrap=True)
 
        for key in sorted(self._timings.keys()):
            stats = self.get_stats(key)
            values = [f"{stats[key]:.1f}" for _, _, key in stat_columns]
            table.add_row(key, *values)
 
        console = Console(width=None, highlight=True)  # FIXED: Use imported Console class
        console.print(table)
 
    def write_parquet(self, path: pathlib.Path) -> None:
        """Save the timings to a parquet file."""
        logger.info(f"Writing timings to {path}")
        frame = pl.DataFrame(self._timings)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
 
 
def _prepare_openvla_input(obs, instruction):
    """Prepare observation for OpenVLA server."""
    # Get RGB images
    front_rgb = obs["policy"]["table_cam"][0] # (H, W, 3)
    wrist_rgb = obs["policy"]["wrist_cam"][0] # (H, W, 3)
    ee_pos = obs["policy"]["eef_pos"][0].cpu().numpy() # (1,3)
    ee_quat = obs["policy"]["eef_quat"][0].cpu().numpy() # (1,4)
    ee_gripper = obs["policy"]["gripper_pos"][0].cpu().numpy() 
    ee_quat_xyzw = ee_quat[[1,2,3,0]]
    eef_angle = _quat2axisangle(ee_quat_xyzw)
    state_libero = np.concatenate([ee_pos, eef_angle, ee_gripper])
    front_rgb = obs["policy"]["table_cam"][0]  # (H, W, 3)
    wrist_rgb = obs["policy"]["wrist_cam"][0]  # (H, W, 3)
    
    # Convert to numpy and resize
    front_image_np = front_rgb.cpu().numpy()
    wrist_image_np = wrist_rgb.cpu().numpy()
    
    # Resize to 224x224
    wrist_image_np = cv2.resize(wrist_image_np, (224, 224), interpolation=cv2.INTER_CUBIC)
    front_image_np = cv2.resize(front_image_np, (224, 224), interpolation=cv2.INTER_CUBIC)
    
    # Normalize to [0, 255]
    front_image_np = (front_image_np - front_image_np.min()) / (front_image_np.max() - front_image_np.min())
    wrist_image_np = (wrist_image_np - wrist_image_np.min()) / (wrist_image_np.max() - wrist_image_np.min())
    front_image_np = (front_image_np * 255).astype(np.uint8)
    wrist_image_np = (wrist_image_np * 255).astype(np.uint8)
    
    # Prepare OpenVLA observation
    # obs_openvla = {
    #     "image": front_image_np,
    #     "image_wrist": wrist_image_np,  # Add wrist camera if using 2 images
    #     "instruction": instruction,
    # }
    # obs_openvla = {
    #     "observation": {
    #         "full_image": front_image_np,  # Convert to list for JSON serialization
    #         "wrist_image": wrist_image_np,  # Convert to list for JSON serialization
    #     },
    #     "instruction": instruction,
    # }
    obs_openvla_1 = {
                "full_image": front_image_np,  # Server looks for "full_image"
        "wrist_image": wrist_image_np,  # Server looks for "wrist_image"
        "instruction": instruction,
    }
    obs_openvla = {
                "full_image": front_image_np,  # Server looks for "full_image"
        "wrist_image": wrist_image_np,  # Server looks for "wrist_image"
        "state": state_libero,
        "instruction": instruction,
    }
    return obs_openvla
 
 
def check_success(env, min_height=0.15):
    """Check if task was successful (object lifted)."""
    object_heights = (env.scene["objects"].data.object_link_pos_w[:, :, 2] -
                     env.scene["objects"].data.default_object_state[:, :, 2])
    return (object_heights > min_height).any(dim=1)


def check_dropping(env, min_height=-0.05, max_velocity=30.0):
    """Check if object has dropped or is moving too fast (failure conditions)."""
    object_heights = (env.scene["objects"].data.object_link_pos_w[:, :, 2] -
                     env.scene["objects"].data.default_object_state[:, :, 2])
    object_velocities = env.scene["objects"].data.object_link_vel_w[:, :, :]
    
    # Check if object dropped too low
    dropped = (object_heights < min_height).any(dim=1)
    
    # Check if object velocity magnitude is too high
    velocity_magnitude = torch.norm(object_velocities, dim=2)  # [batch, objects]
    too_fast = (velocity_magnitude > max_velocity).any(dim=1)
    if too_fast[0]:
        print(f"Object is moving too fast: {velocity_magnitude[0]}")
    
    return dropped | too_fast


def check_episode_done(env):
    """Check if episode should terminate (success or failure conditions)."""
    success = check_success(env)
    failure = check_dropping(env)
    
    if failure[0]:
        success[0] = False
    
    return success[0] or failure[0], success[0]
 
def run_episode(env, policy, max_steps=500, actions_per_query=8):  # Use 8 actions per query!
    """Run closed-loop control: obs -> OpenVLA -> actions -> execute -> obs -> ..."""
    total_steps = 0
    query_count = 0
    action_plan = collections.deque()  # Store action chunk
    
    # Get initial observation
    obs, _ = env.reset()
    
    # Wait for environment to settle
    wait_action = torch.zeros((1, 7), device=env.device)
    for _ in range(5):
        obs, _, _, _, _ = env.step(wait_action)
        total_steps += 1
    
    print(f"[INFO]: Starting closed-loop control (max_steps={max_steps})")
    timing_recorder = TimingRecorder()
    
    while total_steps < max_steps:
        # Check if episode is done
        episode_done, success = check_episode_done(env)
        if episode_done:
            print(f"Episode finished at step {total_steps}. Success: {success}")
            break
        
        try:
            # Query policy when action plan is empty
            if not action_plan:
                obs_fn = _prepare_openvla_input(obs, env.instruction)
                
                inference_start = time.time()
                action_pi = policy.infer(obs_fn)
                
                # FIXED: Handle nested list structure
                action_chunk = action_pi["actions"][0]  # Get the inner list of actions
                # print(f"[INFO] Received {len(action_chunk)} actions from policy")
                
                query_count += 1
                
                # Add all actions to the plan
                action_plan.extend(action_chunk[:actions_per_query])
                
                # Record timing
                timing_recorder.record("client_infer_ms", 1000 * (time.time() - inference_start))
                for key, value in action_pi.get("server_timing", {}).items():
                    timing_recorder.record(f"server_{key}", value)
            
            # Get next action from plan
            action = action_plan.popleft()
            
            # Validate action is numpy array
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            
            if action.shape != (7,):
                raise ValueError(f"Expected action shape (7,), got {action.shape}")
            
            if total_steps >= max_steps:
                break
            
            # Unpack euler angles properly
            ai, aj, ak = float(action[3]), float(action[4]), float(action[5])
            axangle_ax, axangle_angle = t3d.euler.euler2axangle(ai, aj, ak)
            action_delta_pose = np.concatenate([
                action[0:3], 
                axangle_ax * axangle_angle, 
                -action[6:7]
            ])
            action_tensor = torch.from_numpy(action_delta_pose).float().unsqueeze(0)
            
            # Step environment
            obs, _, _, _, _ = env.step(action_tensor)
            total_steps += 1
                
        except Exception as e:
            print(f"[ERROR] Error during query {query_count}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Final check
    final_success = check_success(env)[0].item()
    
    # Print timing statistics
    if timing_recorder._timings:
        timing_recorder.print_all_stats()
    
    return {
        'success': final_success,
        'total_steps': total_steps,
        'queries': query_count,
    }
 
def main() -> None:
    """Main execution function."""
    # Initialize OpenVLA HTTP client
    print(f"[INFO] Initializing OpenVLA HTTP client")
    print(f"[INFO] Server: {args_cli.host}:{args_cli.port}")
    print(f"[INFO] Unnorm key: {args_cli.unnorm_key}")
    
    policy = OpenVLAHttpPolicy(
        host=args_cli.host,
        port=args_cli.port,
        unnorm_key=args_cli.unnorm_key,
        gripper_id=args_cli.gripper_id,
    )
    
    logger.info(f"Server metadata: {policy.get_server_metadata()}")
 
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
    env.instruction = instruction
    
    print(f"[INFO] Task instruction: {instruction}")
    
    # Run closed-loop demonstrations
    results = []
    success_count = 0
    
    print(f"\n[INFO]: Running {args_cli.num_demos} closed-loop demonstrations")
    print(f"[INFO]: Max steps per episode: {args_cli.max_steps}")
    print(f"[INFO]: Actions per query: {args_cli.actions_per_query}")
    
    for demo_idx in range(args_cli.num_demos):
        print(f"\n{'='*50}")
        print(f"Demo {demo_idx + 1}/{args_cli.num_demos}")
        print(f"{'='*50}")
        
        try:
            # Run closed-loop control
            result = run_episode(
                env=env,
                policy=policy,
                max_steps=args_cli.max_steps,
                actions_per_query=args_cli.actions_per_query
            )
            
            result['demo_idx'] = demo_idx
            
            if result['success']:
                success_count += 1
                print(f"✓ Demo {demo_idx + 1} SUCCESS in {result['total_steps']} steps with {result['queries']} queries")
            else:
                print(f"✗ Demo {demo_idx + 1} FAILED after {result['total_steps']} steps with {result['queries']} queries")
            
            results.append(result)
        
        except Exception as e:
            print(f"Error in demo {demo_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'demo_idx': demo_idx,
                'success': False,
                'error': str(e),
                'total_steps': 0,
                'queries': 0,
            })
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Success rate: {success_count}/{args_cli.num_demos} ({100*success_count/args_cli.num_demos:.1f}%)")
    
    if results:
        successful_results = [r for r in results if r['success']]
        if successful_results:
            avg_steps = np.mean([r['total_steps'] for r in successful_results])
            avg_queries = np.mean([r['queries'] for r in successful_results])
            print(f"Average steps (successful): {avg_steps:.1f}")
            print(f"Average queries (successful): {avg_queries:.1f}")
    
    # Save results
    output_file = f"openvlaoft_{args_cli.task_id}_o{args_cli.num_objects}_d{args_cli.num_demos}_{args_cli.unnorm_key}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'task_id': args_cli.task_id,
            'num_objects': args_cli.num_objects,
            'num_demos': args_cli.num_demos,
            'max_steps': args_cli.max_steps,
            'actions_per_query': args_cli.actions_per_query,
            'unnorm_key': args_cli.unnorm_key,
            'gripper_id': args_cli.gripper_id,
            'success_rate': success_count / args_cli.num_demos,
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Cleanup
    env.close()
 
 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
    simulation_app.close()