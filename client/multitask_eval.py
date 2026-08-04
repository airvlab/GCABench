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
import rich
import rich.table
import rich.console

# Parse arguments FIRST - before any Isaac Lab imports
parser = argparse.ArgumentParser(description="Test GraspVLA closed-loop control.")
# parser.add_argument("--task", type=str, default='Grasp-Franka-IK-Rel-img', help="Name of the task.")
parser.add_argument("--task", type=str, default='Grasp-UR10-Short-Suction-IK-Rel-img', help="Name of the task.")
# parser.add_argument("--task_ids", type=str, nargs='+', default=['a01', 'b01', 'h02', 'z02', 'z03'], 
#                      help="Task IDs to evaluate (e.g., a01 b01 c01 d03)")
# parser.add_argument("--task_ids", type=str, nargs='+', default=['a07', 'a08', 'b08','b09','h07','h09','z08','z09'] )
# parser.add_argument("--task_ids", type=str, nargs='+', default=['a07', 'a08','h07','h09'] )
parser.add_argument("--task_ids", type=str, nargs='+', default=['b08','b09','z08','z09'] )
# parser.add_argument("--task_ids", type=str, nargs='+', default=['h07','h09','z08','z09'] )
parser.add_argument("--num_objects", type=int, default=4, help="Number of objects in scene")
parser.add_argument("--num_demos", type=int, default=20, help="Number of demonstrations per task")
parser.add_argument("--graspvla_port", type=str, default="6666", help="GraspVLA server port")
parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode")
parser.add_argument("--actions_per_query", type=int, default=10, help="Number of actions to execute per GraspVLA query")
parser.add_argument("--openvla_port", type=str, default="8000", help="OpenVLA server port")
parser.add_argument("--policy", type=str, default="pi0soft", help="Policy to use")
parser.add_argument("--seed", type=int, default=42, help="Seed to use")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to connect to the server")
parser.add_argument("--port", type=int, default=5000, help="Port to connect to the server")
parser.add_argument("--api_key", type=str, default=None, help="API key for the server")
parser.add_argument("--timing_file", type=str, default=None, help="Path to save timings (parquet)")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.num_envs = 1
if "Suction" in args_cli.task:
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
from isaacsim.core.utils.stage import create_new_stage
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
    Copied from robosuite.
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class TimingRecorder:
    """Records timing measurements for different keys."""

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}

    def record(self, key: str, time_ms: float) -> None:
        if key not in self._timings:
            self._timings[key] = []
        self._timings[key].append(time_ms)

    def get_stats(self, key: str) -> dict[str, float]:
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
        table = rich.table.Table(
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
        console = rich.console.Console(width=None, highlight=True)
        console.print(table)

    def write_parquet(self, path: pathlib.Path) -> None:
        logger.info(f"Writing timings to {path}")
        frame = pl.DataFrame(self._timings)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


def _prepare_pi_input(obs, instruction, policy_type="pi0"):
    front_rgb = obs["policy"]["table_cam"][0]
    wrist_rgb = obs["policy"]["wrist_cam"][0]
    ee_pos = obs["policy"]["eef_pos"][0].cpu().numpy()
    ee_quat = obs["policy"]["eef_quat"][0].cpu().numpy()
    ee_gripper = obs["policy"]["gripper_pos"][0].cpu().numpy()
    ee_quat_xyzw = ee_quat[[1, 2, 3, 0]]
    eef_angle = _quat2axisangle(ee_quat_xyzw)
    state_libero = np.concatenate([ee_pos, eef_angle, ee_gripper])

    front_image_np = front_rgb.cpu().numpy()
    wrist_image_np = wrist_rgb.cpu().numpy()
    wrist_image_np = cv2.resize(wrist_image_np, (224, 224), interpolation=cv2.INTER_CUBIC)
    front_image_np = cv2.resize(front_image_np, (224, 224), interpolation=cv2.INTER_CUBIC)
    front_image_np = (front_image_np - front_image_np.min()) / (front_image_np.max() - front_image_np.min())
    wrist_image_np = (wrist_image_np - wrist_image_np.min()) / (wrist_image_np.max() - wrist_image_np.min())
    front_image_np = (front_image_np * 255).astype(np.uint8)
    wrist_image_np = (wrist_image_np * 255).astype(np.uint8)

    gripper_id = 1
    if policy_type == "pi0":
        obs_pi = {
            "observation/image": front_image_np,
            "observation/wrist_image": wrist_image_np,
            "observation/state": state_libero,
            "prompt": instruction,
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


def check_success(env, min_height=0.15):
    object_heights = (env.scene["objects"].data.object_link_pos_w[:, :, 2] -
                      env.scene["objects"].data.default_object_state[:, :, 2])
    return (object_heights > min_height).any(dim=1)


def check_dropping(env, min_height=-0.06):
    object_heights = (env.scene["objects"].data.object_link_pos_w[:, :, 2] -
                      env.scene["objects"].data.default_object_state[:, :, 2])
    return (object_heights < min_height).any(dim=1)


def check_episode_done(env):
    success = check_success(env)
    failure = check_dropping(env)
    return success[0] or failure[0], success[0]


def run_episode(env, policy, max_steps=500, actions_per_query=8, policy_type="pi0"):
    replan_steps = actions_per_query
    total_steps = 0
    query_count = 0
    action_plan = collections.deque()
    obs, _ = env.reset()

    wait_action = torch.zeros((1, 7), device=env.device)
    trajectory = []
    for _ in range(5):
        obs, _, _, _, _ = env.step(wait_action)
        total_steps += 1

    timing_recorder = TimingRecorder()

    while total_steps < max_steps:
        episode_done, success = check_episode_done(env)
        if episode_done:
            break
        try:
            if not action_plan:
                obs_fn = _prepare_pi_input(obs, env.instruction, policy_type)
                inference_start = time.time()
                action_pi = policy.infer(obs_fn)
                action_chunk = action_pi["actions"]
                query_count += 1

                assert len(action_chunk) >= replan_steps, \
                    f"We want to replan every {replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                action_plan.extend(action_chunk[:replan_steps])

            action = action_plan.popleft()
            timing_recorder.record("client_infer_ms", 1000 * (time.time() - inference_start))
            for key, value in action_pi.get("server_timing", {}).items():
                timing_recorder.record(f"server_{key}", value)
            for key, value in action_pi.get("policy_timing", {}).items():
                timing_recorder.record(f"policy_{key}", value)

            if total_steps >= max_steps:
                break
            axangle_ax, axangle_angle = t3d.euler.euler2axangle(*action[3:6])
            action_delta_pose = np.concatenate([action[0:3], axangle_ax * axangle_angle, -action[6:7]])
            action = torch.from_numpy(action_delta_pose).float().unsqueeze(0)

            obs, _, _, _, _ = env.step(action)
            eef_pose_gripper = torch.cat([obs["policy"]["eef_pos"], obs["policy"]["eef_quat"],
                                          obs["policy"]["gripper_pos"]], dim=1)
            trajectory.append(eef_pose_gripper)
            total_steps += 1

        except Exception as e:
            print(f"Error during query {query_count}: {e}")
            break

    final_success = check_success(env)[0].item()

    return {
        'success': final_success,
        'total_steps': total_steps,
        'queries': query_count,
    }


def create_env_for_task(task_id, task_name, device, num_objects):
    """Create environment configured for a specific task."""
    env_cfg = parse_env_cfg(task_name, device=device, num_envs=1)
    task_builder = TaskBuilder(task_id=task_id)
    TG_CFGs = task_builder.create_scene_objects(num_objs=num_objects)
    instruction = task_builder.task_instruction
    env_cfg.scene.objects = TG_CFGs
    env_cfg.terminations.time_out = None
    if task_id.startswith("a"):
        env_cfg.scene.table.spawn.scale = [0.7, 1.0, 1.0]

    env = gym.make(task_name, cfg=env_cfg).unwrapped
    env.instruction = instruction
    return env, instruction


def main() -> None:
    """Main execution function."""
    # Initialize policy client
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args_cli.host,
        port=args_cli.port,
        api_key=args_cli.api_key,
    )
    logger.info(f"Server metadata: {policy.get_server_metadata()}")

    # Store results for all tasks
    all_results = {}
    overall_success = 0
    overall_total = 0

    print(f"\n{'#'*60}")
    print(f"# Multi-Task Evaluation")
    print(f"# Tasks: {args_cli.task_ids}")
    print(f"# Demos per task: {args_cli.num_demos}")
    print(f"{'#'*60}\n")

    for task_id in args_cli.task_ids:
        print(f"\n{'='*60}")
        print(f"TASK: {task_id}")
        print(f"{'='*60}")

        # Create environment for this task
        env, instruction = create_env_for_task(
            task_id=task_id,
            task_name=args_cli.task,
            device=args_cli.device,
            num_objects=args_cli.num_objects
        )
        env.seed(args_cli.seed)
        print(f"Instruction: {instruction}")

        # Run demos for this task
        task_results = []
        success_count = 0

        for demo_idx in range(args_cli.num_demos):
            print(f"\n--- Demo {demo_idx + 1}/{args_cli.num_demos} (Task: {task_id}) ---")
            try:
                result = run_episode(
                    env=env,
                    policy=policy,
                    max_steps=args_cli.max_steps,
                    actions_per_query=args_cli.actions_per_query,
                    policy_type=args_cli.policy
                )
                result['demo_idx'] = demo_idx
                result['task_id'] = task_id

                if result['success']:
                    success_count += 1
                    print(f"✓ SUCCESS in {result['total_steps']} steps ({result['queries']} queries)")
                else:
                    print(f"✗ FAILED after {result['total_steps']} steps ({result['queries']} queries)")

                task_results.append(result)

            except Exception as e:
                print(f"Error in demo {demo_idx + 1}: {e}")
                task_results.append({
                    'demo_idx': demo_idx,
                    'task_id': task_id,
                    'success': False,
                    'error': str(e),
                    'total_steps': 0,
                    'queries': 0,
                })

        # Task summary
        task_success_rate = success_count / args_cli.num_demos
        print(f"\n[Task {task_id}] Success: {success_count}/{args_cli.num_demos} ({100*task_success_rate:.1f}%)")

        all_results[task_id] = {
            'instruction': instruction,
            'success_count': success_count,
            'total_demos': args_cli.num_demos,
            'success_rate': task_success_rate,
            'results': task_results
        }

        overall_success += success_count
        overall_total += args_cli.num_demos

        # Close environment for this task
        # del env

    
        try:
            env.close()
        except (AttributeError, RuntimeError) as e:
            if "SurfaceGripper" in str(e) or "cfg" in str(e):
                print(f"Expected error: {e}")
                pass  # Expected error, ignore it
            else:
                raise

        # env.close()
        create_new_stage()
    # Final summary
    print(f"\n{'#'*60}")
    print(f"# FINAL SUMMARY")
    print(f"{'#'*60}")

    # Per-task results table
    table = rich.table.Table(title="Per-Task Results", show_header=True, header_style="bold")
    table.add_column("Task ID", style="cyan")
    table.add_column("Instruction", style="white", max_width=40)
    table.add_column("Success", justify="right", style="green")
    table.add_column("Rate", justify="right", style="yellow")

    for task_id in args_cli.task_ids:
        r = all_results[task_id]
        table.add_row(
            task_id,
            r['instruction'][:40] + "..." if len(r['instruction']) > 40 else r['instruction'],
            f"{r['success_count']}/{r['total_demos']}",
            f"{100*r['success_rate']:.1f}%"
        )

    console = rich.console.Console()
    console.print(table)

    overall_rate = overall_success / overall_total if overall_total > 0 else 0
    print(f"\nOverall Success Rate: {overall_success}/{overall_total} ({100*overall_rate:.1f}%)")

    # Save results
    output_file = f"json_result/{args_cli.policy}_{'_'.join(args_cli.task_ids)}_o{args_cli.num_objects}_d{args_cli.num_demos}_{args_cli.task}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'task_ids': args_cli.task_ids,
            'num_objects': args_cli.num_objects,
            'num_demos': args_cli.num_demos,
            'max_steps': args_cli.max_steps,
            'actions_per_query': args_cli.actions_per_query,
            'overall_success_rate': overall_rate,
            'per_task_results': all_results
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
    simulation_app.close()