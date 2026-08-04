import requests
import json_numpy
import collections
json_numpy.patch()
import numpy as np
import torch
import transforms3d as t3d
import argparse
import gymnasium as gym
import numpy as np
import torch
import zmq
import pickle
import json
import time
import torch
import transforms3d as t3d
import cv2
# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

import dataclasses
import enum
import logging
import pathlib
import time

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import polars as pl
import rich
import tqdm
import matplotlib.pyplot as plt

# Parse arguments FIRST - before any Isaac Lab imports
parser = argparse.ArgumentParser(description="Test GraspVLA closed-loop control.")
parser.add_argument("--task", type=str, default='Grasp-Franka-IK-Rel-img', help="Name of the task.")
# parser.add_argument("--task", type=str, default= 'Grasp-Franka-Vacuum-IK-Rel-img', help="Name of the task.")
# parser.add_argument("--task", type=str, default='Grasp-Franka-Robotiq-IK-Rel-img', help="Name of the task.")
# parser.add_argument("--task", type=str, default='Grasp-UR10-Short-Suction-IK-Rel-img', help="Name of the task.")
# parser.add_argument("--task", type=str, default='Grasp-UR10-Long-Suction-IK-Rel-img', help="Name of the task.")
parser.add_argument("--task_id", type=str, default='a01', help="Task ID in task config")
parser.add_argument("--num_objects", type=int, default=4, help="Number of objects in scene")
parser.add_argument("--num_demos", type=int, default=50, help="Number of demonstrations")
parser.add_argument("--graspvla_port", type=str, default="6666", help="GraspVLA server port")
parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode")
# for a03 task, we need to set max_steps to 700
# parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
parser.add_argument("--actions_per_query", type=int, default=10, help="Number of actions to execute per GraspVLA query")
parser.add_argument("--openvla_port", type=str, default="8000", help="OpenVLA server port")
parser.add_argument("--policy", type=str, default="pi0soft", help="Policy to use")
parser.add_argument("--seed", type=int, default=42, help="Seed to use")
# parser.add_argument("--seed", type=int, default=42, help="Seed to use")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to connect to the server")
parser.add_argument("--port", type=int, default=5000, help="Port to connect to the server")
parser.add_argument("--api_key", type=str, default=None, help="API key for the server")
parser.add_argument("--timing_file", type=str, default=None, help="Path to save timings (parquet)")
parser.add_argument("--gripper_id", type=int, default=0, help="Gripper ID for HGP-MoA (0-4)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# args_cli.headless = True
args_cli.num_envs = 1
if "Suction" in args_cli.task or "Vacuum" in args_cli.task:
    args_cli.device = "cpu"
else:
    args_cli.device = "cuda"



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
import math
logger = logging.getLogger(__name__)


class EnvMode(enum.Enum):
    """Supported environments."""
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
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
        """Print statistics for all keys in a concise format."""

        table = rich.table.Table(
            title="[bold blue]Timing Statistics[/bold blue]",
            show_header=True,
            header_style="bold white",
            border_style="blue",
            title_justify="center",
        )

        # Add metric column with custom styling
        table.add_column("Metric", style="cyan", justify="left", no_wrap=True)

        # Add statistical columns with consistent styling
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

        # Add rows for each metric with formatted values
        for key in sorted(self._timings.keys()):
            stats = self.get_stats(key)
            values = [f"{stats[key]:.1f}" for _, _, key in stat_columns]
            table.add_row(key, *values)

        # Print with custom console settings
        console = rich.console.Console(width=None, highlight=True)
        console.print(table)

    def write_parquet(self, path: pathlib.Path) -> None:
        """Save the timings to a parquet file."""
        logger.info(f"Writing timings to {path}")
        frame = pl.DataFrame(self._timings)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


def _prepare_pi_input(obs, instruction):
    # Match your GraspVLA packaging style: use third_view as front, wrist/side as side
    front_rgb = obs["policy"]["table_cam"][0] # (H, W, 3)
    wrist_rgb = obs["policy"]["wrist_cam"][0] # (H, W, 3)
    ee_pos = obs["policy"]["eef_pos"][0].cpu().numpy() # (1,3)
    ee_quat = obs["policy"]["eef_quat"][0].cpu().numpy() # (1,4)
    ee_gripper = obs["policy"]["gripper_pos"][0].cpu().numpy() 
    ee_quat_xyzw = ee_quat[[1,2,3,0]]
    eef_angle = _quat2axisangle(ee_quat_xyzw)
    state_libero = np.concatenate([ee_pos, eef_angle, ee_gripper])

    front_image_np = front_rgb.cpu().numpy()
    wrist_image_np = wrist_rgb.cpu().numpy()
    # resize the image into 256x256
    wrist_image_np = cv2.resize(wrist_image_np, (224, 224), interpolation=cv2.INTER_CUBIC)
    front_image_np = cv2.resize(front_image_np, (224, 224), interpolation=cv2.INTER_CUBIC)
    front_image_np = (front_image_np - front_image_np.min()) / (front_image_np.max() - front_image_np.min())
    wrist_image_np = (wrist_image_np - wrist_image_np.min()) / (wrist_image_np.max() - wrist_image_np.min())
    front_image_np = front_image_np * 255
    wrist_image_np = wrist_image_np * 255
    front_image_np = front_image_np.astype(np.uint8)
    wrist_image_np = wrist_image_np.astype(np.uint8)
    
    gripper_id = 0
    if args_cli.policy == "pi0":
        obs_pi = {
            "observation/image": front_image_np,
            "observation/wrist_image": wrist_image_np,
            "observation/state": state_libero,
            "prompt": instruction,
        }
    elif args_cli.policy == "openvla":
        obs_pi = {
            "image": front_image_np,
            # "observation/wrist_image": wrist_image_np,
            # "observation/state": state_libero,
            "instruction": instruction,
            # "gripper_id": gripper_id,
        }
    else:
        obs_pi = {
            "observation/image": front_image_np,
            "observation/wrist_image": wrist_image_np,
            "observation/state": state_libero,
            "prompt": instruction,
            "gripper_id": gripper_id,
        }
    return obs_pi


@staticmethod
def action_to_env_delta(action7):
    # Convert [dx, dy, dz, r, p, y, g] to [dx, dy, dz, axis_angle(3), g]
    ax, ang = t3d.euler.euler2axangle(*action7[3:6])
    delta_pose = np.concatenate([action7[0:3], np.array(ax) * ang, -action7[6:7]])
    return torch.from_numpy(delta_pose).float().unsqueeze(0)


# def check_success(env, min_height=0.15):
# for suction a01 task, we need to set min_height to 0.10
def check_success(env, min_height=0.15): 
    """Check if task was successful (object lifted)."""
    object_heights = (env.scene["objects"].data.object_link_pos_w[:, :, 2] - 
                     env.scene["objects"].data.default_object_state[:, :, 2])
    return (object_heights > min_height).any(dim=1)

def check_dropping(env, min_height=-0.05, max_velocity=50.0):
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
    
    # Return True if either condition is met (failure)
    return dropped | too_fast

def check_episode_done(env):
    """Check if episode should terminate (success or failure conditions)."""
    # Check success
    success = check_success(env)
    
    # Check failure conditions (optional)
    failure = check_dropping(env) # Add your failure conditions here
    if failure[0]:
        success[0] = False
    return success[0] or failure[0], success[0]


def run_episode(env, policy, max_steps=500, actions_per_query=8):
    replan_steps = actions_per_query
    """
    Run closed-loop control: obs -> GraspVLA -> actions -> execute -> obs -> ...
    """
    total_steps = 0
    query_count = 0
    action_plan = collections.deque()
    # Get initial observation
    obs, _ = env.reset() 
    
    wait_action = torch.zeros((1, 7), device=env.device)
    trajectory = []
    for _ in range(5):
        obs, _, _, _,_ = env.step(wait_action)
        total_steps += 1
    
    print(f"[INFO]: Starting closed-loop control (max_steps={max_steps})")
    timing_recorder = TimingRecorder()
    gripper_close = False
    close_cnt = 0
    inference_time_ms = 0.0  # initialise so it's always defined

    while total_steps < max_steps:
        # Check if episode is done
        episode_done, success = check_episode_done(env)
        if episode_done:
            print(f"Episode finished at step {total_steps}. Success: {success}")
            break
        try:
            if not action_plan:
                obs_fn = _prepare_pi_input(obs, env.instruction)

                # ── inference timing ──────────────────────────────────────────
                inference_start = time.time()
                action_pi = policy.infer(obs_fn)
                inference_time_ms = 1000 * (time.time() - inference_start)
                query_count += 1  # FIX: was never incremented before
                print(
                    f"[Query {query_count:>4d} | Step {total_steps:>5d}] "
                    f"Inference time: {inference_time_ms:.1f} ms"
                )
                # ─────────────────────────────────────────────────────────────

                timing_recorder.record("client_infer_ms", inference_time_ms)
                for key, value in action_pi.get("server_timing", {}).items():
                    timing_recorder.record(f"server_{key}", value)
                for key, value in action_pi.get("policy_timing", {}).items():
                    timing_recorder.record(f"policy_{key}", value)

                action_chunk = action_pi["actions"]
                assert (
                    len(action_chunk) >= replan_steps
                ), f"We want to replan every {replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                action_plan.extend(action_chunk[:replan_steps])

            action = action_plan.popleft()

            if total_steps >= max_steps:
                break

            axangle_ax, axangle_angle = t3d.euler.euler2axangle(*action[3:6])
            action_delta_pose = np.concatenate([action[0:3], axangle_ax * axangle_angle, -action[6:7]])
            action = torch.from_numpy(action_delta_pose).float().unsqueeze(0)

            obs, _, _, _, _ = env.step(action)
            eef_pose_gripper = torch.cat(
                [obs["policy"]["eef_pos"], obs["policy"]["eef_quat"], obs["policy"]["gripper_pos"]], dim=1
            )
            trajectory.append(eef_pose_gripper)
            total_steps += 1

        except Exception as e:
            print(f"Error during query {query_count}: {e}")
            break

    # Print per-episode timing summary
    if "client_infer_ms" in timing_recorder._timings:
        timing_recorder.print_all_stats()

    # Final check
    final_success = check_success(env)[0].item()
    
    return {
        'success': final_success,
        'total_steps': total_steps,
        'queries': query_count,
    }


def main() -> None:
    """Main execution function."""
    # Initialize GraspVLA client
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args_cli.host,
        port=args_cli.port,
        api_key=args_cli.api_key,
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
    instruction = task_builder.task_instruction
    env.instruction = instruction
    
    # Run closed-loop demonstrations
    results = []
    success_count = 0
    
    print(f"[INFO]: Running {args_cli.num_demos} closed-loop demonstrations")
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
        avg_steps = np.mean([r['total_steps'] for r in results if r['success']])
        avg_queries = np.mean([r['queries'] for r in results if r['success']])
        print(f"Average steps (successful): {avg_steps:.1f}")
        print(f"Average queries (successful): {avg_queries:.1f}")
    
    # Save results
    output_file = f"json_result/adaptation/{args_cli.policy}_{args_cli.gripper_id}_{args_cli.task_id}_d{args_cli.num_demos}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'task_id': args_cli.task_id,
            'num_objects': args_cli.num_objects,
            'num_demos': args_cli.num_demos,
            'max_steps': args_cli.max_steps,
            'actions_per_query': args_cli.actions_per_query,
            'success_rate': success_count / args_cli.num_demos,
            'results': results
        }, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
    simulation_app.close()