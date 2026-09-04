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

### Scene types

Scenes are grouped by workspace structure and the kind of reasoning they require, not by gripper. Object pose and placement are varied within each family (for example, a bowl upright vs. upside down).

| Scene type | Task IDs | Description |
| :--- | :--- | :--- |
| **Singulated** | `a*`, `g*`, `z*` | Grasp and lift a single object on a flat table. Covers common / YCB objects (`z*`), thin or flat objects that are hard to grasp (`a*`), and irregular shapes such as toys (`g*`). |
| **Cluttered** | `b*`, `c*` | Grasp a target among overlapping or stacked objects. Subtypes below increase occlusion and collision risk. |
| **Constrained** | `d*`, `e*` | Grasp inside a limited workspace (bin or basket). Requires precise approach, collision avoidance, and reaching into corners or against walls. |
| **Semantic** | `f*` | Task-level context beyond geometry: keep a cup, plate, tray, or pan balanced (no spillage), prefer a handle or edge, or retrieve an object without disturbing its contents. |

**Cluttered subtypes**

| Subtype | IDs | Description |
| :--- | :--- | :--- |
| Scattered | `c01`–`c06` | Objects are spread across the workspace with some spacing. |
| Stacked | `b01`–`b10` | Objects are piled tightly, with occlusion and a higher chance of knocking neighbors. |
| Packed | e.g. `b06`–`b09` | Objects are packed along an axis with little clearance for gripper insertion. |

### Scene groups (by task ID prefix)

| Prefix | Scene type | Theme | IDs |
| :--- | :--- | :--- | :--- |
| `a*` | Singulated | Flat objects | `a01`–`a10` |
| `b*` | Cluttered | Stacked / packed | `b01`–`b10` |
| `c*` | Cluttered | Scattered | `c01`–`c06` |
| `d*` | Constrained | Bin | `d01`–`d05` |
| `e*` | Constrained | Small basket | `e01`–`e05` |
| `f*` | Semantic | Balance / instruction | `f010`, `f011`, `f012`, `f020`, `f021`, `f030`, `f040`, `f041`, `f050`, `f051`, `f060` |
| `g*` | Singulated | Complex shapes | `g01`–`g05` |
| `z*` | Singulated | Single-object / YCB baselines | `z01`–`z11` |


### Creating Custom Tasks
Add a new entry under `tasks:` in `tasks/config.yaml` with a unique ID, `task_class`, `instruction`, and object list (`object_path`, `pos`, `rot`). Object poses are relative to the table center (`scene_setup.base_pose`).

## 🛠️ Data Collection



### To collect the data, run:

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

### To replay the data, run:
```bash
python replay_demos.py --dataset_file ./datasets/<your_demo>.hdf5
```

---

## 📊 Evaluation

Closed-loop evaluation queries a remote policy server, executes the predicted actions in Isaac Lab, and scores each episode by whether the object is lifted. Start the policy server first, then run the matching client from `client/`.

`--task` selects the robot / gripper (see [Robot Status](#-robot-status--availability)). `--task_id` selects the scene (see [Tasks](#-tasks)). Suction / vacuum tasks automatically use CPU.

### Policy clients

| Policy | Client | Default server |
| :--- | :--- | :--- |
| OpenPI / π0 | `client/Openpi_client_img.py` | `--host 0.0.0.0 --port 5000` |
| OpenPI (multi-task) | `client/multitask_eval.py` | `--host 0.0.0.0 --port 5000` |
| GR00T | `client/gr00t_client.py` | `--host localhost --port 4000` |
| OpenVLA | `client/Openvla_client_img.py` | `--host localhost --port 6001` |
| OpenVLA-OFT | `client/Openvla-oft_client_img.py` | `--host localhost --port 8777` |
| GraspVLA | `client/GraspVLA_Client.py` | `--graspvla_port 6666` |

### Run evaluation

Example: 

**OpenPI / π0 (single task)**
```bash
python client/Openpi_client_img.py \
  --task Grasp-Franka-IK-Rel-img \
  --task_id a01 \
  --num_demos 20 \
  --host 0.0.0.0 \
  --port 5000
```

**OpenPI (multiple tasks)**
```bash
python client/multitask_eval.py \
  --task Grasp-Franka-IK-Rel-img \
  --task_ids a01 b01 z02 \
  --num_demos 20 \
  --host 0.0.0.0 \
  --port 5000
```


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
