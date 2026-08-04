import collections
import json
import logging
import math
import pathlib
import time

import cv2
import json_numpy
json_numpy.patch()
import numpy as np
import torch
import transforms3d as t3d
import argparse
import gymnasium as gym
import polars as pl
import rich
import rich.table
import rich.console

# Isaac Lab AppLauncher  — must be imported before any Isaac Lab modules
from isaaclab.app import AppLauncher

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing  (must happen before AppLauncher)
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GR00T closed-loop evaluation in Isaac Lab.")

# Task / scene
parser.add_argument("--task",        type=str, default="Grasp-Franka-IK-Rel-img",
                    help="Isaac Lab task name.")
# parser.add_argument("--task", type=str, default="Grasp-Franka-Vacuum-IK-Rel-img")
# parser.add_argument("--task", type=str, default="Grasp-Franka-Robotiq-IK-Rel-img")
# parser.add_argument("--task", type=str, default="Grasp-UR10-Short-Suction-IK-Rel-img")
# parser.add_argument("--task", type=str, default="Grasp-UR10-Long-Suction-IK-Rel-img")
parser.add_argument("--task_id",     type=str, default="a01",
                    help="Task ID in task config.")
parser.add_argument("--num_objects", type=int, default=4,
                    help="Number of objects in the scene.")
parser.add_argument("--num_demos",   type=int, default=50,
                    help="Number of evaluation episodes.")
parser.add_argument("--max_steps",   type=int, default=500,
                    help="Maximum simulator steps per episode.")
# For a03 use --max_steps 700

# Action chunking
parser.add_argument("--actions_per_query", type=int, default=10,
                    help="How many actions from each chunk to execute before re-querying.")

# GR00T server
parser.add_argument("--host",    type=str, default="localhost",
                    help="GR00T server host.")
parser.add_argument("--port",    type=int, default=4000,
                    help="GR00T server port.")
parser.add_argument("--api_key", type=str, default=None,
                    help="Optional API key for the GR00T server.")

# Misc
parser.add_argument("--policy",     type=str, default="groot",
                    help="Policy label used in output filenames.")
parser.add_argument("--seed",       type=int, default=42)
parser.add_argument("--gripper_id", type=int, default=0,
                    help="Gripper ID (used in output filename only).")
parser.add_argument("--timing_file", type=str, default=None,
                    help="Path to save per-episode timing parquet (optional).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.num_envs = 1
# Suction/vacuum gripper sims run on CPU
if "Suction" in args_cli.task or "Vacuum" in args_cli.task:
    args_cli.device = "cpu"
else:
    args_cli.device = "cuda"


# ─────────────────────────────────────────────────────────────────────────────
# Launch Isaac Lab simulator  (must happen before any Isaac Lab imports)
# ─────────────────────────────────────────────────────────────────────────────
app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ── Isaac Lab imports (after simulator launch) ────────────────────────────────
import sys

_repo_root = pathlib.Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import isaaclab_mimic.envs  # noqa: F401 — registers extra envs
import grasp_env             # noqa: F401
import isaaclab_tasks        # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from local_utils.minimal_gr00t_client import PolicyClient
from local_utils.load_utils  import TaskBuilder
from isaaclab.utils.math     import euler_xyz_from_quat, quat_from_euler_xyz  # noqa: F401
from isaaclab.managers       import SceneEntityCfg                             # noqa: F401
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg  # noqa: F401
from isaaclab.controllers    import DifferentialIKControllerCfg                # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (x,y,z,w) to axis-angle.
    Adapted from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490
    """
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class TimingRecorder:
    """Accumulates timing samples and prints/saves statistics."""

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}

    def record(self, key: str, time_ms: float) -> None:
        self._timings.setdefault(key, []).append(time_ms)

    def get_stats(self, key: str) -> dict[str, float]:
        times = self._timings[key]
        return {
            "mean": float(np.mean(times)),
            "std":  float(np.std(times)),
            "p25":  float(np.quantile(times, 0.25)),
            "p50":  float(np.quantile(times, 0.50)),
            "p75":  float(np.quantile(times, 0.75)),
            "p90":  float(np.quantile(times, 0.90)),
            "p95":  float(np.quantile(times, 0.95)),
            "p99":  float(np.quantile(times, 0.99)),
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
        stat_cols = [
            ("Mean", "yellow", "mean"), ("Std",  "yellow", "std"),
            ("P25",  "magenta","p25"),  ("P50",  "magenta","p50"),
            ("P75",  "magenta","p75"),  ("P90",  "magenta","p90"),
            ("P95",  "magenta","p95"),  ("P99",  "magenta","p99"),
        ]
        for name, style, _ in stat_cols:
            table.add_column(name, justify="right", style=style, no_wrap=True)
        for key in sorted(self._timings):
            stats  = self.get_stats(key)
            values = [f"{stats[k]:.1f}" for _, _, k in stat_cols]
            table.add_row(key, *values)
        rich.console.Console(width=None, highlight=True).print(table)

    def write_parquet(self, path: pathlib.Path) -> None:
        logger.info(f"Writing timings to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(self._timings).write_parquet(path)


# ─────────────────────────────────────────────────────────────────────────────
# Observation preparation  (Isaac Lab  →  GR00T schema)
# ─────────────────────────────────────────────────────────────────────────────
def _prepare_groot_input(obs: dict, instruction: str) -> dict:
    """
    Pack Isaac Lab observations into GR00T's nested-dict observation format.

    GR00T's check_observation expects TOP-LEVEL keys: "video", "state", "annotation"
    with sub-keys nested underneath — NOT flat dotted keys like "video.image".

    The flat dotted keys (e.g. "video.image") are used in the LeRobot dataset schema
    and observation_space definitions, but the live inference server expects nested dicts.
    """
    # ── images ───────────────────────────────────────────────────────────────
    front_rgb = obs["policy"]["table_cam"][0].cpu().numpy()
    wrist_rgb = obs["policy"]["wrist_cam"][0].cpu().numpy()

    front_rgb = cv2.resize(front_rgb, (256, 256), interpolation=cv2.INTER_CUBIC).astype(np.uint8)
    wrist_rgb = cv2.resize(wrist_rgb, (256, 256), interpolation=cv2.INTER_CUBIC).astype(np.uint8)

    # GR00T inference server expects video tensors as (B, T, H, W, C).
    front_rgb = np.expand_dims(np.expand_dims(front_rgb, axis=0), axis=0)
    wrist_rgb = np.expand_dims(np.expand_dims(wrist_rgb, axis=0), axis=0)

    # ── robot state ───────────────────────────────────────────────────────────
    ee_pos  = obs["policy"]["eef_pos"][0].cpu().numpy()       # (3,)  xyz
    ee_quat = obs["policy"]["eef_quat"][0].cpu().numpy()      # (4,)  wxyz
    gripper = obs["policy"]["gripper_pos"][0].cpu().numpy()   # (2,)

    ee_quat_xyzw = ee_quat[[1, 2, 3, 0]]
    rpy = _quat2axisangle(ee_quat_xyzw)                       # (3,) axis-angle

    gripper = np.broadcast_to(gripper.astype(np.float32), (2,)).copy()

    # GR00T inference server expects state tensors as (B, T, D).
    def _bt(arr: np.ndarray) -> np.ndarray:
        return np.expand_dims(np.expand_dims(arr, axis=0), axis=0)

    # ── nested dict format expected by GR00T server ───────────────────────────
    return {
        "video": {
            "image":       front_rgb,   # (1, 1, 256, 256, 3) uint8
            "wrist_image": wrist_rgb,   # (1, 1, 256, 256, 3) uint8
        },
        "state": {
            "x":       _bt(np.array([ee_pos[0]], dtype=np.float32)),
            "y":       _bt(np.array([ee_pos[1]], dtype=np.float32)),
            "z":       _bt(np.array([ee_pos[2]], dtype=np.float32)),
            "rx":    _bt(np.array([rpy[0]], dtype=np.float32)),
            "ry":   _bt(np.array([rpy[1]], dtype=np.float32)),
            "rz":     _bt(np.array([rpy[2]], dtype=np.float32)),
            "gripper_left": _bt([gripper[0]]),
            "gripper_right": _bt([gripper[1]]),
        },
        # GR00T language modality expects list[list[str]] with shape (B, T).
        # Keep multiple common keys for compatibility across embodiment configs.
        "language": {
            "instruction": [[instruction]],
            "task": [[instruction]],
            "annotation.human.task_description": [[instruction]],
        },
        "annotation.human.task_description": instruction,
        "annotation": {
            "human": {
                "task_description": instruction,
            }
        },
    }

    # def _prepare_groot_input(obs: dict, instruction: str) -> dict:
    # # ── images ───────────────────────────────────────────────────────────────
    # front_rgb = obs["policy"]["table_cam"][0].cpu().numpy()
    # wrist_rgb = obs["policy"]["wrist_cam"][0].cpu().numpy()

    # front_rgb = cv2.resize(front_rgb, (256, 256), interpolation=cv2.INTER_CUBIC).astype(np.uint8)
    # wrist_rgb = cv2.resize(wrist_rgb, (256, 256), interpolation=cv2.INTER_CUBIC).astype(np.uint8)
    # # Shape: (256, 256, 3)  — server adds B/T dims internally

    # # ── robot state ───────────────────────────────────────────────────────────
    # ee_pos  = obs["policy"]["eef_pos"][0].cpu().numpy()
    # ee_quat = obs["policy"]["eef_quat"][0].cpu().numpy()
    # gripper = obs["policy"]["gripper_pos"][0].cpu().numpy()

    # ee_quat_xyzw = ee_quat[[1, 2, 3, 0]]
    # rpy = _quat2axisangle(ee_quat_xyzw)
    # gripper = np.broadcast_to(gripper.astype(np.float32), (2,)).copy()

    # # ── nested dict — plain arrays, no manual B/T expansion ──────────────────
    # return {
    #     "video": {
    #         "image":       front_rgb,                                    # (256,256,3)
    #         "wrist_image": wrist_rgb,                                    # (256,256,3)
    #     },
    #     "state": {
    #         "x":       np.array([ee_pos[0]], dtype=np.float32),          # (1,)
    #         "y":       np.array([ee_pos[1]], dtype=np.float32),
    #         "z":       np.array([ee_pos[2]], dtype=np.float32),
    #         "rx":      np.array([rpy[0]],    dtype=np.float32),
    #         "ry":      np.array([rpy[1]],    dtype=np.float32),
    #         "rz":      np.array([rpy[2]],    dtype=np.float32),
    #         "gripper_left":  np.array([gripper[0]], dtype=np.float32),
    #         "gripper_right": np.array([gripper[1]], dtype=np.float32),
    #     },
    #     "annotation": {
    #         "human": {
    #             "task_description": instruction,
    #         }
    #     },
    # }

# def _prepare_groot_input(obs: dict, instruction: str) -> dict:
#     """
#     Convert Isaac Lab observation dict into the flat key/value schema that
#     GR00T's PolicyClient expects, mirroring LiberoEnv._process_observation().

#     GR00T observation schema
#     ────────────────────────
#     video.image          (H, W, 3)  uint8  – primary (table) camera
#     video.wrist_image    (H, W, 3)  uint8  – wrist camera
#     state.x / y / z      (1,)      float32 – EEF position
#     state.roll/pitch/yaw (1,)      float32 – EEF orientation as axis-angle
#     state.gripper        (2,)      float32 – gripper joint positions
#     annotation.human.action.task_description  str

#     NOTE: GR00T normalises images internally; do NOT scale pixels to [0,1].
#     """
#     # ── images ───────────────────────────────────────────────────────────────
#     front_rgb = obs["policy"]["table_cam"][0].cpu().numpy()   # (H, W, 3)
#     wrist_rgb = obs["policy"]["wrist_cam"][0].cpu().numpy()   # (H, W, 3)

#     # GR00T / LIBERO reference uses 256×256
#     front_rgb = cv2.resize(front_rgb, (256, 256), interpolation=cv2.INTER_CUBIC).astype(np.uint8)
#     wrist_rgb = cv2.resize(wrist_rgb, (256, 256), interpolation=cv2.INTER_CUBIC).astype(np.uint8)

#     # ── robot state ───────────────────────────────────────────────────────────
#     ee_pos  = obs["policy"]["eef_pos"][0].cpu().numpy()       # (3,)  xyz
#     ee_quat = obs["policy"]["eef_quat"][0].cpu().numpy()      # (4,)  wxyz  ← Isaac convention
#     gripper = obs["policy"]["gripper_pos"][0].cpu().numpy()   # (2,) gripper qpos

#     # Isaac Lab: wxyz  →  convert to xyzw for _quat2axisangle
#     ee_quat_xyzw = ee_quat[[1, 2, 3, 0]]
#     rpy = _quat2axisangle(ee_quat_xyzw)                       # (3,) axis-angle

#     # Ensure gripper is always (2,) float32 (some configs return (1,))
#     gripper = np.broadcast_to(gripper.astype(np.float32), (2,)).copy()

#     return {
#         "video.image":       front_rgb,
#         "video.wrist_image": wrist_rgb,
#         "state.x":           np.array([ee_pos[0]], dtype=np.float32),
#         "state.y":           np.array([ee_pos[1]], dtype=np.float32),
#         "state.z":           np.array([ee_pos[2]], dtype=np.float32),
#         "state.roll":        np.array([rpy[0]], dtype=np.float32),
#         "state.pitch":       np.array([rpy[1]], dtype=np.float32),
#         "state.yaw":         np.array([rpy[2]], dtype=np.float32),
#         "state.gripper":     gripper,
#         "annotation.human.action.task_description": instruction,
#     }


# ─────────────────────────────────────────────────────────────────────────────
# Action unpacking  (GR00T schema  →  Isaac Lab 7-DoF delta)
# ─────────────────────────────────────────────────────────────────────────────

def _unpack_groot_actions(action_dict: dict) -> np.ndarray:
    """
    Unpack GR00T's per-DoF action dict into a (chunk_len, 7) float32 array.

    GR00T returns tensors shaped (B, T, chunk_len, 1) from the server.
    We squeeze the B and T leading dims to get (chunk_len, 1) before concat.
    """
    def _pick(*names: str) -> np.ndarray:
        for name in names:
            if name in action_dict:
                arr = action_dict[name]
                # Squeeze all leading dims until shape is (chunk_len, dof)
                while arr.ndim > 2:
                    arr = arr.squeeze(0)
                return arr
        raise KeyError(names[0])

    try:
        actions = np.concatenate([
            _pick("action.x",     "x"),
            _pick("action.y",     "y"),
            _pick("action.z",     "z"),
            _pick("action.roll",  "roll",  "action.rx", "rx"),
            _pick("action.pitch", "pitch", "action.ry", "ry"),
            _pick("action.yaw",   "yaw",   "action.rz", "rz"),
            _pick("action.gripper", "gripper"),
        ], axis=-1)                             # (chunk_len, 7)

    except KeyError as exc:
        raise KeyError(
            f"Missing GR00T action key: {exc}. "
            f"Available keys: {list(action_dict.keys())}"
        ) from exc

    return actions.astype(np.float32)          # (chunk_len, 7)
# ─────────────────────────────────────────────────────────────────────────────
# Success / failure checks
# ─────────────────────────────────────────────────────────────────────────────

def check_success(env, min_height: float = 0.15) -> torch.Tensor:
    """Object lifted above its spawn height by at least min_height."""
    delta_z = (
        env.scene["objects"].data.object_link_pos_w[:, :, 2]
        - env.scene["objects"].data.default_object_state[:, :, 2]
    )
    return (delta_z > min_height).any(dim=1)


def check_dropping(env, min_height: float = -0.05, max_velocity: float = 50.0) -> torch.Tensor:
    """Object fell below spawn or is moving unrealistically fast (physics blow-up)."""
    delta_z = (
        env.scene["objects"].data.object_link_pos_w[:, :, 2]
        - env.scene["objects"].data.default_object_state[:, :, 2]
    )
    vel_mag = torch.norm(env.scene["objects"].data.object_link_vel_w, dim=2)

    dropped   = (delta_z < min_height).any(dim=1)
    too_fast  = (vel_mag > max_velocity).any(dim=1)
    if too_fast[0]:
        print(f"  [WARN] Object moving too fast: {vel_mag[0]}")
    return dropped | too_fast


def check_episode_done(env):
    """Returns (done: bool, success: bool)."""
    success = check_success(env)
    failure = check_dropping(env)
    if failure[0]:
        success[0] = False
    return (success[0] or failure[0]).item(), success[0].item()


# ─────────────────────────────────────────────────────────────────────────────
# Episode loop
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(
    env,
    policy: PolicyClient,
    max_steps: int = 500,
    actions_per_query: int = 10,
) -> dict:
    """
    Run one closed-loop episode:
        obs  →  GR00T (policy.get_action)  →  action chunk  →  env.step  →  obs  →  …

    Returns a result dict with keys: success, total_steps, queries.
    """
    replan_steps = actions_per_query
    total_steps  = 0
    query_count  = 0
    action_plan  = collections.deque()

    obs, _ = env.reset()

    # Let the scene settle before sending the first observation
    wait_action = torch.zeros((1, 7), device=env.device)
    for _ in range(5):
        obs, _, _, _, _ = env.step(wait_action)
        total_steps += 1

    print(f"[INFO] Starting episode (max_steps={max_steps}, "
          f"actions_per_query={replan_steps})")

    timing_recorder  = TimingRecorder()
    inference_time_ms = 0.0

    while total_steps < max_steps:
        episode_done, success = check_episode_done(env)
        if episode_done:
            print(f"  Episode ended at step {total_steps} — success={success}")
            break

        try:
            # ── Query GR00T when the action buffer is empty ───────────────────
            if not action_plan:
                groot_obs = _prepare_groot_input(obs, env.instruction)

                t0 = time.time()
                # ── GR00T inference ───────────────────────────────────────────
                action_dict, infer_info = policy.get_action(groot_obs)
                print(f"action_dict: {action_dict}")
                # ─────────────────────────────────────────────────────────────
                inference_time_ms = 1000.0 * (time.time() - t0)
                query_count += 1

                print(
                    f"  [Query {query_count:>4d} | Step {total_steps:>5d}] "
                    f"Inference: {inference_time_ms:.1f} ms"
                )
                timing_recorder.record("client_infer_ms", inference_time_ms)

                # Optional server-side timings forwarded by GR00T
                for key, value in infer_info.get("server_timing", {}).items():
                    timing_recorder.record(f"server_{key}", value)

                # ── Unpack action chunk  (chunk_len, 7) ───────────────────────
                action_chunk = _unpack_groot_actions(action_dict)   # (chunk_len, 7)
                # print(f"action_chunk: {action_chunk}")

                assert len(action_chunk) >= replan_steps, (
                    f"Expected ≥{replan_steps} actions from GR00T, "
                    f"got {len(action_chunk)}."
                )
                action_plan.extend(action_chunk[:replan_steps])

            # ── Execute one action ────────────────────────────────────────────
            action = action_plan.popleft()   # (7,) float32: [dx,dy,dz,r,p,y,g]

            if total_steps >= max_steps:
                break

            # Convert Euler → axis-angle and flip gripper sign for Isaac Lab
            axangle_ax, axangle_angle = t3d.euler.euler2axangle(*action[3:6])
            action_delta_pose = np.concatenate(
                [action[0:3], axangle_ax * axangle_angle, -action[6:7]]
            )
            action_tensor = torch.from_numpy(action_delta_pose).float().unsqueeze(0)

            obs, _, _, _, _ = env.step(action_tensor)
            print(f"action_tensor: {action_tensor}")
            total_steps += 1

        except Exception as exc:
            print(f"  [ERROR] Query {query_count}: {exc}")
            break

    # ── Episode summary ───────────────────────────────────────────────────────
    if "client_infer_ms" in timing_recorder._timings:
        timing_recorder.print_all_stats()

    if args_cli.timing_file is not None:
        timing_recorder.write_parquet(pathlib.Path(args_cli.timing_file))

    final_success = check_success(env)[0].item()
    return {
        "success":     final_success,
        "total_steps": total_steps,
        "queries":     query_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── GR00T PolicyClient ────────────────────────────────────────────────────
    policy = PolicyClient(host=args_cli.host, port=args_cli.port)
    try:
        modality_cfg = policy.get_modality_config()
        print("[GR00T] Server modality config:")
        for modality, cfg in modality_cfg.items():
            print(f"  {modality}: {cfg}")
    except Exception as e:
        print(f"[GR00T] Could not fetch modality config: {e}")
    logger.info("GR00T PolicyClient connected at "
                f"{args_cli.host}:{args_cli.port}")

    # ── Isaac Lab environment ─────────────────────────────────────────────────
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)

    task_builder      = TaskBuilder(task_id=args_cli.task_id)
    TG_CFGs           = task_builder.create_scene_objects(num_objs=args_cli.num_objects)
    instruction       = task_builder.task_instruction
    env_cfg.scene.objects = TG_CFGs

    env_cfg.terminations.time_out = None
    if args_cli.task_id.startswith("a"):
        env_cfg.scene.table.spawn.scale = [0.7, 1.0, 1.0]

    env             = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.seed(args_cli.seed)
    env.instruction = instruction

    # ── Evaluation loop ───────────────────────────────────────────────────────
    results       = []
    success_count = 0

    print(f"\n{'='*55}")
    print(f"  Task:              {args_cli.task}  [{args_cli.task_id}]")
    print(f"  Instruction:       {instruction}")
    print(f"  Episodes:          {args_cli.num_demos}")
    print(f"  Max steps:         {args_cli.max_steps}")
    print(f"  Actions per query: {args_cli.actions_per_query}")
    print(f"  GR00T server:      {args_cli.host}:{args_cli.port}")
    print(f"{'='*55}\n")

    for demo_idx in range(args_cli.num_demos):
        print(f"\n{'─'*55}")
        print(f"  Demo {demo_idx + 1:>3d} / {args_cli.num_demos}")
        print(f"{'─'*55}")

        try:
            result = run_episode(
                env=env,
                policy=policy,
                max_steps=args_cli.max_steps,
                actions_per_query=args_cli.actions_per_query,
            )
            result["demo_idx"] = demo_idx

            if result["success"]:
                success_count += 1
                print(f"  ✓ SUCCESS — {result['total_steps']} steps, "
                      f"{result['queries']} queries")
            else:
                print(f"  ✗ FAILED  — {result['total_steps']} steps, "
                      f"{result['queries']} queries")

            results.append(result)

        except Exception as exc:
            print(f"  [ERROR] Demo {demo_idx + 1}: {exc}")
            results.append({
                "demo_idx":    demo_idx,
                "success":     False,
                "error":       str(exc),
                "total_steps": 0,
                "queries":     0,
            })

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  FINAL RESULTS")
    print(f"{'='*55}")
    success_rate = success_count / args_cli.num_demos
    print(f"  Success rate: {success_count}/{args_cli.num_demos} "
          f"({100 * success_rate:.1f} %)")

    successful = [r for r in results if r.get("success")]
    if successful:
        print(f"  Avg steps   (success): "
              f"{np.mean([r['total_steps'] for r in successful]):.1f}")
        print(f"  Avg queries (success): "
              f"{np.mean([r['queries'] for r in successful]):.1f}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    import os
    os.makedirs("json_result/adaptation", exist_ok=True)
    output_file = (
        f"json_result/adaptation/"
        f"{args_cli.policy}_{args_cli.gripper_id}_"
        f"{args_cli.task_id}_d{args_cli.num_demos}.json"
    )
    with open(output_file, "w") as fh:
        json.dump(
            {
                "task":              args_cli.task,
                "task_id":           args_cli.task_id,
                "instruction":       instruction,
                "num_objects":       args_cli.num_objects,
                "num_demos":         args_cli.num_demos,
                "max_steps":         args_cli.max_steps,
                "actions_per_query": args_cli.actions_per_query,
                "host":              args_cli.host,
                "port":              args_cli.port,
                "success_rate":      success_rate,
                "results":           results,
            },
            fh,
            indent=2,
        )
    print(f"\n  Results saved → {output_file}")

    env.close()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
    simulation_app.close()