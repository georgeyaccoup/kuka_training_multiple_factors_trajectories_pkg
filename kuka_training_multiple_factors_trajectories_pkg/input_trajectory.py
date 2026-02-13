import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray, String
import numpy as np

class EndlessTrajectoryGenerator(Node):
    def __init__(self):
        super().__init__('endless_trajectory_generator')
        
        self.coord_pub = self.create_publisher(Pose, 'coordinates', 10)
        self.state_pub = self.create_publisher(String, '/experiment_state', 10)
        self.create_subscription(Float64MultiArray, '/rl_trial_config', self.rl_config_callback, 10)

        self.BASE_DURATION = 0.8  # Fast Speed
        self.FREQUENCY = 100.0    
        
        self.SAFE_HEIGHT = 0.95
        self.HOME_POSITION = [0.5, 0.0, 0.90]
        self.PICK_Z = 0.78 # Tested Pick Height
        self.DROP_Z = 0.3 

        self.waypoints = []
        self.current_segment = 0
        self.current_step = 0
        self.total_steps = 0
        self.is_moving = False
        self.last_pose = None
        self.wait_steps = 0
        self.MAX_WAIT_STEPS = 100 
        
        # Pause Counter
        self.pick_pause_counter = 0 
        
        self.timer = self.create_timer(1.0 / self.FREQUENCY, self.timer_callback)

    def rl_config_callback(self, msg):
        if self.is_moving: return
        
        data = msg.data
        pick_x, pick_y = data[0], data[1]
        drop_x, drop_y = data[2], data[3]
        joint_speeds = data[4:10]
        
        avg_speed_factor = sum(joint_speeds) / 6.0
        if avg_speed_factor < 0.1: avg_speed_factor = 0.1
            
        duration = self.BASE_DURATION / avg_speed_factor
        self.total_steps = int(duration * self.FREQUENCY)
        
        self.generate_waypoints(pick_x, pick_y, drop_x, drop_y)
        
        self.current_segment = 0
        self.current_step = 0
        self.wait_steps = 0
        self.pick_pause_counter = 0 
        self.is_moving = True
        
        state_msg = String()
        state_msg.data = "start_optimized"
        self.state_pub.publish(state_msg)

    def generate_waypoints(self, px, py, dx, dy):
        home = np.array([self.HOME_POSITION[0], self.HOME_POSITION[1], self.HOME_POSITION[2], 0.0]) 
        
        wp1 = np.array([px, py, self.SAFE_HEIGHT, 0.0])
        wp2 = np.array([px, py, self.PICK_Z, 1.0])       # Down (Vac ON)
        wp3 = np.array([px, py, self.SAFE_HEIGHT, 1.0])  # Lift
        wp4 = np.array([dx, dy, self.SAFE_HEIGHT, 1.0])
        wp5 = np.array([dx, dy, self.DROP_Z, 1.0])
        wp6 = np.array([dx, dy, self.DROP_Z, 0.0])
        wp7 = np.array([dx, dy, self.SAFE_HEIGHT, 0.0])
        wp8 = home

        self.waypoints = [home, wp1, wp2, wp3, wp4, wp5, wp6, wp7, wp8]

    def timer_callback(self):
        if not self.is_moving:
            if self.last_pose: self.coord_pub.publish(self.last_pose)
            return

        if self.current_segment >= len(self.waypoints) - 1:
            if self.wait_steps < self.MAX_WAIT_STEPS:
                self.wait_steps += 1
                final_wp = self.waypoints[-1]
                msg = Pose()
                msg.position.x = final_wp[0]
                msg.position.y = final_wp[1]
                msg.position.z = final_wp[2]
                msg.orientation.w = float(final_wp[3]) 
                self.coord_pub.publish(msg)
                return
            else:
                self.is_moving = False
                stop_msg = String()
                stop_msg.data = "stop"
                self.state_pub.publish(stop_msg)
                return

        # --- PAUSE LOGIC (0.5s at the bottom) ---
        if self.current_segment == 2 and self.current_step == 0:
            if self.pick_pause_counter < 50: # 50 ticks = 0.5s
                self.pick_pause_counter += 1
                hold_wp = self.waypoints[2]
                msg = Pose()
                msg.position.x = hold_wp[0]
                msg.position.y = hold_wp[1]
                msg.position.z = hold_wp[2]
                msg.orientation.w = float(hold_wp[3])
                self.coord_pub.publish(msg)
                return 

        start_wp = self.waypoints[self.current_segment]
        end_wp   = self.waypoints[self.current_segment + 1]
        
        alpha = self.current_step / self.total_steps
        current_pos_xyz = start_wp[:3] + (end_wp[:3] - start_wp[:3]) * alpha
        
        if end_wp[3] > 0.5:
            current_suction = 1.0
        else:
            current_suction = 0.0 if alpha > 0.5 else start_wp[3]

        msg = Pose()
        msg.position.x = current_pos_xyz[0]
        msg.position.y = current_pos_xyz[1]
        msg.position.z = current_pos_xyz[2]
        msg.orientation.w = float(current_suction)
        
        self.coord_pub.publish(msg)
        self.last_pose = msg 

        self.current_step += 1
        if self.current_step > self.total_steps:
            self.current_step = 0
            self.current_segment += 1

def main(args=None):
    rclpy.init(args=args)
    node = EndlessTrajectoryGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()