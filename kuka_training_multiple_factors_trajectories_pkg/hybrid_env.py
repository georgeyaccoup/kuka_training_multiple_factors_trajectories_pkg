import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
import time
from kuka_training_multiple_factors_trajectories_pkg.lr_baseline import EnergyTeacher

class KukaHybridEnv(gym.Env):
    def __init__(self):
        super(KukaHybridEnv, self).__init__()
        
        # Action: [6 Speeds (0.3-2.0), 6 Offsets (-0.05-0.05)]
        low_bounds  = np.array([0.3]*6 + [-0.05]*6, dtype=np.float32)
        high_bounds = np.array([2.0]*6 + [ 0.05]*6, dtype=np.float32)
        
        self.action_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

        if not rclpy.ok(): rclpy.init()
        self.node = rclpy.create_node('rl_gym_manager')
        
        self.traj_pub = self.node.create_publisher(Float64MultiArray, '/rl_trial_config', 10)
        self.box_reset_pub = self.node.create_publisher(String, '/reset_box', 10)
        self.result_sub = self.node.create_subscription(Float64MultiArray, '/energy_result', self.result_cb, 10)
        self.sensor_sub = self.node.create_subscription(Float64MultiArray, '/robot_sensors', self.sensor_cb, 10)

        self.teacher = EnergyTeacher()
        self.latest_energy = None
        self.current_joints = np.zeros(6)
        self.current_vels = np.zeros(6)
        self.rng = np.random.default_rng()
        self.forced_path = None

    def set_test_path(self, px, py, dx, dy):
        self.forced_path = [px, py, dx, dy]

    def step(self, action):
        joint_speeds = action[0:6]
        joint_offsets = action[6:12]
        
        # --- 1. COORDINATE GENERATION ---
        if self.forced_path is not None:
            pick_x, pick_y, drop_x, drop_y = self.forced_path
            self.forced_path = None 
        else:
            # [CRITICAL FIX] FIXED PICK LOCATION
            pick_x = 0.95
            pick_y = 0.0
            
            # RANDOM DROP LOCATION (Anywhere in 360 degrees)
            # Radius 0.65 to 0.95 (Safe Zone)
            drop_angle = self.rng.uniform(-np.pi, np.pi)
            drop_radius = self.rng.uniform(0.65, 0.95)
            drop_x = drop_radius * np.cos(drop_angle)
            drop_y = drop_radius * np.sin(drop_angle)

        packet = [pick_x, pick_y, drop_x, drop_y] + joint_speeds.tolist() + joint_offsets.tolist()
        
        msg = Float64MultiArray()
        msg.data = packet
        self.traj_pub.publish(msg)
        
        # --- 2. WAIT FOR RESULT ---
        self.latest_energy = None
        start_wait = time.time()
        
        # Timeout 25s (Generous buffer for the 0.5s pause)
        TIMEOUT_LIMIT = 25.0 
        
        while self.latest_energy is None:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.time() - start_wait > TIMEOUT_LIMIT:
                self.node.get_logger().warn("Timeout! Robot stuck.")
                self.latest_energy = 150000.0 
                break

        actual_energy = self.latest_energy
        
        # --- 3. REWARD ---
        avg_speed = np.mean(joint_speeds)
        time_penalty = 0.0
        
        # Penalize if slower than standard (1.0)
        # Tolerance: 0.9 is acceptable.
        if avg_speed < 0.9:
             time_penalty = (0.9 - avg_speed) * 20000.0
        
        payload = 5.0 
        pred_energy = self.teacher.predict(payload, self.current_joints, self.current_vels)
        reward = ((pred_energy - actual_energy) * 0.1) - time_penalty
        
        obs = np.concatenate([self.current_joints, self.current_vels, [payload], [pred_energy]])
        self.trigger_reset()
        return obs, reward, True, False, {}

    def trigger_reset(self):
        rst_msg = String()
        rst_msg.data = "reset_now"
        self.box_reset_pub.publish(rst_msg)
        time.sleep(0.2) 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.trigger_reset()
        payload = 5.0
        pred_energy = self.teacher.predict(payload, self.current_joints, self.current_vels)
        obs = np.concatenate([self.current_joints, self.current_vels, [payload], [pred_energy]])
        return obs, {}

    def result_cb(self, msg):
        self.latest_energy = msg.data[0]

    def sensor_cb(self, msg):
        data = np.array(msg.data)
        if len(data) >= 12:
            self.current_joints = data[0:6] 
            self.current_vels = data[6:12]