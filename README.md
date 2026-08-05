# Grasp with Complex Trajectory Benchmark

This repository provides a benchmark and toolkit for collecting and evaluating robotic grasping demonstrations with complex trajectories using **Isaac Lab**.

---

## 🐳 Docker Quick Start (Recommended)

We provide a pre-configured Docker environment to simplify setup and ensure reproducibility. This is the **recommended** way to run the benchmark.

### 1. Prerequisites
Ensure you have the following installed on your host machine:
* **NVIDIA Driver**
* [**Docker** & **NVIDIA Container Toolkit**](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
* Pull the official Isaac Lab base image:
  ```bash
  docker pull nvcr.io/nvidia/isaac-lab:2.3.0
  ```

### 2. Clone Repository
Clone the repository to your local machine:

```bash
git clone https://github.com/airvlab/GCABench.git
cd GCABench
```

### 3. Download Assets
Download asset usd files [objects_usd_fixed_density.zip](https://utdallas.app.box.com/v/multi-gripper-grasp-data) from [MultiGripperGrasp Toolkit](https://irvlutd.github.io/MultiGripperGrasp/).
Extract the contents into the `/models/objects_MG` folder inside this project.

### 4. Run
We provide a helper script `run.sh` to automatically build the environment and mount your workspace.

```bash
# Grant execution permission
chmod +x run.sh

# Launch the container
./run.sh
```

> **Note:** If you encounter a `permission denied` error (e.g., related to Docker daemon socket), please try running with sudo:
> ```bash
> sudo ./run.sh
> ```

Once inside the container terminal, you can run the tasks immediately:

```bash
# Example: Record demonstrations using keyboard
python record_demos.py --teleop_device keyboard --num_demos 10 --task_id a01
```

---

## 📦 Local Installation (Alternative)

If you prefer to run the environment locally without Docker, follow these steps.

### 1. Install Isaac Lab
Install the Isaac Lab main version by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) with **Isaac Sim 5.1.0**.

### 2. Clone the Repository
Clone this repository outside the IsaacLab directory:

```bash
git clone https://github.com/airvlab/GCABench.git
```

### 3. Activate Environment
Use a Python interpreter that has Isaac Lab installed:

```bash
conda activate env_isaaclab
```

### 4. Download Assets
(If you haven't done this in the Docker step above)
Download asset usd files [objects_usd_fixed_density.zip](https://utdallas.app.box.com/v/multi-gripper-grasp-data) from [MultiGripperGrasp Toolkit](https://irvlutd.github.io/MultiGripperGrasp/).
Extract the contents into the `/models/objects_MG` folder.

---

## 🤖 Robot Status & Availability

Robot environments are registered in `grasp_env/__init__.py`. Select one via `--task` in `record_demos.py` (default: `Grasp-Franka-Short-Suction-IK-Rel-img`).

| Robot / Gripper | Env ID (`--task`) |
| :--- | :--- |
| **Franka + Panda hand** | `Grasp-Franka-IK-Rel-img` |
| **Franka + short suction** *(default)* | `Grasp-Franka-Short-Suction-IK-Rel-img` |
| **Franka + vacuum** | `Grasp-Franka-Vacuum-IK-Rel-img` |
| **Franka + Robotiq 2F-85** | `Grasp-Franka-Robotiq-IK-Rel-img` |
| **UR10 short suction** | `Grasp-UR10-Short-Suction-IK-Rel-img` |
| **UR10 long suction** | `Grasp-UR10-Long-Suction-IK-Rel-img` |
| **UR10 parallel gripper** | `Grasp-UR10-Parallel-IK-Rel-img` |
| **Kinova Gen3 (J2N7S300)** | `Grasp-Kinova-J2N75300-IK-Rel-img` |
| **UR5 + Shadow Hand** | `Grasp-UR5-ShadowHand-IK-Rel-img` |
| **Franka + Inspire hand** | `Grasp-Franka-InspireHand-IK-Rel-img` |
Suction / vacuum envs automatically force `--device cpu` in `record_demos.py` for physics stability.

---

## 📋 Tasks

Tasks are defined in [`tasks/config.yaml`](tasks/config.yaml). Select a scene with `--task_id` (e.g. `--task_id a01`). There are currently **63** tasks.

### Task types (`task_class`)

| Type | Count | Description |
| :--- | ---: | :--- |
| `single_obj` | 36 | Pick a single target object (possibly among others) |
| `balance` | 17 | Grasp while keeping support / contents stable (plate, tray, basket, etc.) |
| `stack` | 9 | Pick an object from a vertical stack |
| `clutter` | 1 | Pick a target from a cluttered arrangement |

### Scene groups (by task ID prefix)

| Prefix | Theme | Example IDs |
| :--- | :--- | :--- |
| `a*` | Flat objects | `a01`–`a10` |
| `b*` | Stacked objects | `b01`–`b10` |
| `c*` | Scatter / clutter | `c01`–`c06` |
| `d*` | Constrained in bin | `d01`–`d05` |
| `e*` | Constrained in small basket | `e01`–`e05` |
| `f*` | Semantic | `f010`, `f011`, … |
| `g*` | Complex shapes | `g01`–`g05` |
| `z*` | Single-object baselines | `z01`–`z11` |


## 🛠️ Task Design and Data Collection

### 1. Creating Custom Tasks
Add a new entry under `tasks:` in `tasks/config.yaml` with a unique ID, `task_class`, `instruction`, and object list (`object_path`, `pos`, `rot`). Object poses are relative to the table center (`scene_setup.base_pose`).

### 2. To collect the data, run:

**Option A: Using SpaceMouse (Default)**
```bash
python record_demos.py --teleop_device spacemouse --num_demos 10 --task_id a01
```

**Option B: Using Keyboard**
```bash
python record_demos.py --teleop_device keyboard --num_demos 10 --task_id a01
```

> **How to change the teleoperation device:**
> simply modify the `--teleop_device` argument in the command line.
> * **Available options:** `spacemouse`, `keyboard`, `gamepad`, `handtracking`.

### 3. To replay the data, run:
```bash
python replay_demos.py --dataset_file ./datasets/<your_demo>.hdf5
```

---

## ⚙️ Configuration Guide

### 1. Switching Robots (Mechanical Arm)
To switch robots, change the default `--task` in `record_demos.py`, or pass it on the command line:

```bash
python record_demos.py --task Grasp-Franka-Robotiq-IK-Rel-img --task_id a01 --teleop_device keyboard
```

| Robot Type | Task ID (Config Name) | Preview |
| :--- | :--- | :---: |
| **Franka Emika** | `Grasp-Franka-IK-Rel-img` | <img src="./docs/images/robot_franka.png" width="200"> |
| **Kinova Gen3** | `Grasp-Kinova-J2N75300-IK-Rel-img` | <img src="./docs/images/robot_kinova.png" width="200"> |
| **UR10** <br> *(Long Suction)* | `Grasp-UR10-Long-Suction-IK-Rel-img` | <img src="./docs/images/robot_ur10_long.png" width="200"> |
| **UR10** <br> *(Short Suction)* | `Grasp-UR10-Short-Suction-IK-Rel-img` | <img src="./docs/images/robot_ur10_short.png" width="200"> |

**Example Modification in `record_demos.py`:**
```python
# parser.add_argument("--task", type=str, default= 'Grasp-Franka-IK-Rel-img', help="Name of the task.")
parser.add_argument("--task", type=str, default= 'Grasp-Franka-Short-Suction-IK-Rel-img', help="Name of the task.")
```

> [!WARNING]
> **Important Note for Suction Grippers:**
> If you are using **Suction / Vacuum** tasks, the physics simulation requires the CPU pipeline. `record_demos.py` sets `device=cpu` automatically when the env name contains `Suction` or `Vacuum`.

### 2. Simulation Device (CPU vs GPU)
* **Suction / Vacuum tasks:** use CPU (`device=cpu`).
* **Parallel / multi-finger grippers:** default to CUDA in `record_demos.py`.

---

### 3. 🎮 Supported Input Devices

- **Keyboard**: WASD and arrow keys for control
- **Spacemouse**: 6DOF 3D mouse for precise manipulation
- **Gamepad**: Xbox controller or similar with dual joysticks
- **OpenXR**: Hand tracking using index/thumb tip averaging with pinch-to-grip

For detailed device configuration, see the [IsaacLab device documentation](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.devices.html#device-base) and [teleoperation guide](https://isaac-sim.github.io/IsaacLab/main/source/overview/teleop_imitation.html#teleoperation).

---

<!-- ## 💡 Configuration Tips

- **Positioning**: `pos` and `rot` values are relative to the base pose (table center).
- **Stacking**: Use `diff` parameter to offset objects for tight stacking based on object dimensions.
- **Balance Tasks**: Place the primary grasp target object **first** in the configuration list. -->

---

## 📄 License

[License Information Here]
