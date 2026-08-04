import rospy
import numpy as np
import time
import cv2
import collections
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from openpi_client import websocket_client_policy

# Franka control imports
from franky import Robot, Gripper, CartesianWaypointMotion, CartesianWaypoint, CartesianState, RobotPose, Affine

# --- CONFIGURATION ---
SERVER_IP = "10.72.18.159"
SERVER_PORT = 5000         
ROBOT_IP = "192.168.1.12"  

AI_FREQUENCY = 10.0  
REPLAN_STEPS = 3     

GROUND_CAM_TOPIC = "/ground_cam/color/image_raw"
HAND_CAM_TOPIC = "/hand_cam/color/image_raw"

class FrankyOpenPiROS:
    def __init__(self):
        rospy.init_node("openpi_franky_vision", anonymous=True)
        self.bridge = CvBridge()
        
        self.latest_ground_img = None
        self.latest_hand_img = None
        
        rospy.Subscriber(GROUND_CAM_TOPIC, Image, self._ground_cam_callback)
        rospy.Subscriber(HAND_CAM_TOPIC, Image, self._hand_cam_callback)

        print(f"[Init] Connecting to Franka at {ROBOT_IP}...")
        self.robot = Robot(ROBOT_IP)
        self.robot.recover_from_errors()
        
        # 1. HARDWARE SPEED CUT IN HALF: Changed from 0.1 to 0.05
        self.robot.relative_dynamics_factor = 0.05 
        
        self.gripper_is_open = True
        self.gripper = Gripper(ROBOT_IP)
        self.gripper.move(0.08, 0.1)  
        
        print("[Init] Robot Hardware Ready.")
        self.client = websocket_client_policy.WebsocketClientPolicy(host=SERVER_IP, port=SERVER_PORT)

    def _ground_cam_callback(self, msg):
        try: self.latest_ground_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except: pass

    def _hand_cam_callback(self, msg):
        try: self.latest_hand_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except: pass

    def get_images(self):
        ground = self.latest_ground_img if self.latest_ground_img is not None else np.zeros((480,640,3), dtype=np.uint8)
        hand = self.latest_hand_img if self.latest_hand_img is not None else np.zeros((480,640,3), dtype=np.uint8)
        
        g_res = cv2.resize(ground, (224, 224), interpolation=cv2.INTER_CUBIC)
        h_res = cv2.resize(hand, (224, 224), interpolation=cv2.INTER_CUBIC)
        return g_res, h_res

    def ai_inference_loop(self, task_instruction):
        print(f"Starting VLA Loop for: '{task_instruction}'")
        rate = rospy.Rate(AI_FREQUENCY)
        action_plan = collections.deque() 

        while not rospy.is_shutdown():
            t_start = time.time()
            
            # --- 1. CAPTURE HARDWARE STATE ---
            try:
                raw_data = np.array(self.robot.state.O_T_EE).flatten()
                if raw_data.size != 16:
                    raw_data = np.array(self.robot.state.O_T_EE.matrix).flatten()
                
                current_O_T_EE = raw_data.reshape(4, 4).T
                eef_pos = current_O_T_EE[:3, 3]
                
                rot_matrix = current_O_T_EE[:3, :3]
                angle = np.arccos(np.clip((np.trace(rot_matrix) - 1) / 2, -1.0, 1.0))
                if angle < 1e-6:
                    eef_axangle = np.zeros(3)
                else:
                    eef_axangle = (angle / (2 * np.sin(angle))) * np.array([
                        rot_matrix[2, 1] - rot_matrix[1, 2],
                        rot_matrix[0, 2] - rot_matrix[2, 0],
                        rot_matrix[1, 0] - rot_matrix[0, 1]
                    ])
            except Exception as e:
                print(f"State Error: {e}")
                rate.sleep()
                continue

            # --- 2. REPLAN IF ACTION PLAN IS LOW ---
            if not action_plan:
                ext_img, wrist_img = self.get_images()
                try:
                    width = self.robot.state.gripper_width if self.robot.state.gripper_width else 0.0
                except:
                    width = 0.08 if self.gripper_is_open else 0.0
                
                pi0_state = np.concatenate([eef_pos, eef_axangle, [width]]).astype(np.float32)

                observation = {
                    "observation/image": ext_img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": pi0_state,
                    "prompt": task_instruction,
                }

                try:
                    out = self.client.infer(observation)
                    action_plan.extend(out["actions"][:REPLAN_STEPS]) 
                except Exception as e:
                    print(f"Inference Error: {e}")
                    rate.sleep()
                    continue
            
            # --- 3. EXECUTE SINGLE ACTION ---
            if action_plan:
                action = action_plan.popleft()
                
                # 2. SOFTWARE SPEED CUT IN HALF: Changed from 0.3 to 0.15
                target_pos = eef_pos + (action[:3] * 0.15)
                
                target_O_T_EE = current_O_T_EE.copy()
                target_O_T_EE[:3, 3] = target_pos
                
                try:
                    affine_obj = Affine(target_O_T_EE) 
                    pose_obj = RobotPose(affine_obj)
                    state_obj = CartesianState(pose_obj)
                    motion = CartesianWaypointMotion([CartesianWaypoint(state_obj)])
                    
                    self.robot.move(motion, asynchronous=True)
                except Exception as e:
                    print(f"Motion Error: {e}")
                    self.robot.recover_from_errors()
                
                # --- 4. GRIPPER COMMANDS ---
                gripper_cmd = action[6]
                want_close = gripper_cmd > 0.0 
                if want_close and self.gripper_is_open:
                    self.gripper.move(0.0, 0.1) 
                    self.gripper_is_open = False
                elif not want_close and not self.gripper_is_open:
                    self.gripper.move(0.08, 0.1) 
                    self.gripper_is_open = True

            loop_time = time.time() - t_start
            print(f"[Loop] Remaining Actions: {len(action_plan)} | DT: {loop_time:.3f}s")
            rate.sleep()

if __name__ == "__main__":
    try:
        agent = FrankyOpenPiROS()
        task_instruction = input("Enter the task: ")
        agent.ai_inference_loop(task_instruction=task_instruction)
    except rospy.ROSInterruptException: pass
    except KeyboardInterrupt:
        print("\nStopping robot...")
        agent.robot.stop()
