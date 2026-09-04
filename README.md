# Grasp with Complex Trajectory Benchmark

This repository provides a benchmark and toolkit for collecting and evaluating robotic grasping demonstrations with complex trajectories using **Isaac Lab**.

**Paper:** [Beyond Visual Grasping: Benchmarking Complex Grasping from Detection to Execution](https://arxiv.org/abs/2607.14341) (IROS 2026)  
**Project page:** [https://airvlab.github.io/GCA-Bench/](https://airvlab.github.io/GCA-Bench/)

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

## 📦 Local Installation

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

## 🛠️ Data Collection

`--task` selects the robot / gripper (see [Robot Status](#-robot-status--availability)). `--task_id` selects the scene (see [Tasks](#-tasks)).

### Record demonstrations

```bash
python record_demos.py --teleop_device spacemouse --num_demos 10 --task_id a01
```

### Teleoperation devices

Change the device with `--teleop_device`. Options: `spacemouse`, `keyboard`, `gamepad`, `handtracking`.

- **Keyboard**: WASD and arrow keys
- **Spacemouse**: 6DOF 3D mouse
- **Gamepad**: Xbox-style controller with dual joysticks
- **OpenXR**: Hand tracking (index/thumb tip averaging, pinch-to-grip)

For device setup, see the [Isaac Lab device documentation](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.devices.html#device-base) and [teleoperation guide](https://isaac-sim.github.io/IsaacLab/main/source/overview/teleop_imitation.html#teleoperation).

### Switching robots

Pass `--task` on the command line, or change the default in `record_demos.py`:

```bash
python record_demos.py --task Grasp-Franka-Robotiq-IK-Rel-img --task_id a01 --teleop_device keyboard
```

> [!WARNING]
> **Suction / vacuum grippers:** physics requires the CPU pipeline. `record_demos.py` sets `device=cpu` automatically when the env name contains `Suction` or `Vacuum`.

### Replay demonstrations

```bash
python replay_demos.py --dataset_file ./datasets/<your_demo>.hdf5
```


## 📊 Evaluation

Closed-loop evaluation queries a remote policy server, executes the predicted actions in Isaac Lab, and scores each episode by whether the object is lifted. Start the policy server first, then run the matching client from `client/`.

`--task` selects the robot / gripper (see [Robot Status](#-robot-status--availability)). `--task_id` selects the scene (see [Tasks](#-tasks)). Suction / vacuum tasks automatically use CPU.

### Policy clients

| Policy | Client | 
| :--- | :--- | :--- |
| OpenPI / π0 | `client/Openpi_client_img.py` |
| OpenPI (multi-task) | `client/multitask_eval.py` |
| GR00T | `client/gr00t_client.py` |
| OpenVLA | `client/Openvla_client_img.py` | 
| OpenVLA-OFT | `client/Openvla-oft_client_img.py` | 
| GraspVLA | `client/GraspVLA_Client.py` |

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


## 🤖 Robot Status & Availability

Robot environments are registered in `grasp_env/__init__.py`. Select one via `--task` in `record_demos.py` (default: `Grasp-Franka-Short-Suction-IK-Rel-img`).

| Robot / Gripper | Env ID (`--task`) |
| :--- | :--- |
| **Franka + Panda hand** | `Grasp-Franka-IK-Rel-img` |
| **Franka + short suction** | `Grasp-Franka-Short-Suction-IK-Rel-img` |
| **Franka + vacuum** | `Grasp-Franka-Vacuum-IK-Rel-img` |
| **Franka + Robotiq 2F-85** | `Grasp-Franka-Robotiq-IK-Rel-img` |
| **UR10 short suction** | `Grasp-UR10-Short-Suction-IK-Rel-img` |
| **UR10 long suction** | `Grasp-UR10-Long-Suction-IK-Rel-img` |
| **UR10 parallel gripper** | `Grasp-UR10-Parallel-IK-Rel-img` |
| **Kinova Gen3 (J2N7S300)** | `Grasp-Kinova-J2N75300-IK-Rel-img` |
---

## 📋 Tasks

Tasks are defined in [`tasks/config.yaml`](tasks/config.yaml). Select a scene with `--task_id` (e.g. `--task_id a01`).

### Scene types

We include 4 different type of complex grasping tasks as shown below:

| Scene type | Task IDs | Description |
| :--- | :--- | :--- |
| **Singulated** | `a*`, `g*`, `z*` | Grasp and lift a single object on a flat table. Covers common / YCB objects (`z*`), thin or flat objects that are hard to grasp (`a*`), and irregular shapes such as toys (`g*`). |
| **Cluttered** | `b*`, `c*` | Grasp a target among overlapping or stacked objects. Subtypes below increase occlusion and collision risk. |
| **Constrained** | `d*`, `e*` | Grasp inside a limited workspace (bin or basket). Requires precise approach, collision avoidance, and reaching into corners or against walls. |
| **Semantic** | `f*` | Task-level context beyond geometry: keep a cup, plate, tray, or pan balanced (no spillage), prefer a handle or edge, or retrieve an object without disturbing its contents. |


### Creating Custom Tasks
Add a new entry under `tasks:` in `tasks/config.yaml` with a unique ID, `task_class`, `instruction`, and object list (`object_path`, `pos`, `rot`). Object poses are relative to the table center (`scene_setup.base_pose`).


---

<!-- ## 💡 Configuration Tips

- **Positioning**: `pos` and `rot` values are relative to the base pose (table center).
- **Stacking**: Use `diff` parameter to offset objects for tight stacking based on object dimensions.
- **Balance Tasks**: Place the primary grasp target object **first** in the configuration list. -->


## 📄 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhang2026gcabench,
  title={Beyond Visual Grasping: Benchmarking Complex Grasping from Detection to Execution},
  author={Zhang, Hanyi and Nguyen, Khang and Munasinghe, Charith and Hela, Basu and Li, Tianyu and Luo, Zihong and Nguyen, Hoan and van de Venn, Hans Wernher and Zheng, Yalin and Prakash, Ravi and Ta, Tung D. and Nguyen, Anh and Huang, Baoru},
  year={2026},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  eprint={2607.14341},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2607.14341}
}
```

---
