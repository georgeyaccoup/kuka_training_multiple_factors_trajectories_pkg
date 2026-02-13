import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
import numpy as np

class EnergyCalculator(Node):
    def __init__(self):
        super().__init__('energy_calculator')
        
        # Subscriptions
        self.create_subscription(Float64MultiArray, '/robot_sensors', self.sensor_callback, 10)
        self.create_subscription(String, '/experiment_state', self.state_callback, 10)
        
        # Publisher (Sends result to RL Agent)
        self.result_pub = self.create_publisher(Float64MultiArray, '/energy_result', 10)

        self.recording = False
        self.total_energy = 0.0
        self.prev_time = None

    def state_callback(self, msg):
        command = msg.data
        
        if command == "start_optimized":
            # START RECORDING
            self.total_energy = 0.0
            self.recording = True
            self.prev_time = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info("Started Recording Energy...")

        elif command == "stop":
            # STOP RECORDING & PUBLISH RESULT
            if self.recording:
                self.recording = False
                self.publish_result()
                self.get_logger().info(f"Stopped. Total Energy: {self.total_energy:.2f} J")

    def sensor_callback(self, msg):
        if not self.recording:
            return

        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.prev_time is None:
            self.prev_time = current_time
            return

        dt = current_time - self.prev_time
        self.prev_time = current_time
        
        # Parse Data (Torques [0-5], Velocities [6-11])
        data = np.array(msg.data)
        if len(data) < 12: return
        
        torques = data[0:6]
        velocities = data[6:12]

        # Power = Torque * Velocity
        mechanical_power = np.sum(np.abs(torques * velocities))
        
        # Electrical Loss Model (Heat loss in windings)
        # P_loss = R * (Torque/K_t)^2  (Simplified as 0.1 * Torque^2)
        electrical_loss = np.sum(0.1 * (torques ** 2))
        
        total_power = mechanical_power + electrical_loss
        
        # Energy = Integral of Power
        self.total_energy += total_power * dt

    def publish_result(self):
        msg = Float64MultiArray()
        # [Total Energy, Duration (Placeholder)]
        msg.data = [self.total_energy, 0.0] 
        self.result_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = EnergyCalculator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()