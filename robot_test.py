"""Spawn FRANKA_ROBOTIQ_GRIPPER_CFG in Isaac Sim."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--physics",
    action="store_true",
    default=False,
    help="Step physics (gripper joints may drift). Default: paused for inspection.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from robot_assets.franka_cfg import FRANKA_ROBOTIQ_GRIPPER_CFG

def main():
    sim = SimulationContext(sim_utils.SimulationCfg(device=args.device))
    sim.set_camera_view([1.8, 1.8, 1.4], [0.0, 0.0, 0.6])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    dome_light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
    dome_light_cfg.func("/World/DomeLight", dome_light_cfg)

    distant_light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
    distant_light_cfg.func("/World/DistantLight", distant_light_cfg, translation=(1.0, 1.0, 2.0))

    robot_cfg = FRANKA_ROBOTIQ_GRIPPER_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(cfg=robot_cfg)

    sim.reset()
    robot.reset()

    if args.physics:
        # --physics: auto-play and step (Isaac Lab hides the Play button when stopped).
        sim._timeline.play()
        print("[INFO]: Physics running — joints held at default targets. Close window to exit.")
    else:
        sim._timeline.stop()
        print("[INFO]: Simulation paused for inspection. Use --physics to test drift under simulation.")

    while simulation_app.is_running():
        if args.physics:
            robot.set_joint_position_target(robot.data.default_joint_pos)
            robot.write_data_to_sim()
            sim.step(render=True)
            robot.update(sim.get_physics_dt())
        else:
            sim.render()

    simulation_app.close()

if __name__ == "__main__":
    main()