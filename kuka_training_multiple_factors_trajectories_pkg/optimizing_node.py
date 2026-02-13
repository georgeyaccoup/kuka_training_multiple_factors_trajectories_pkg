import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import time
from sklearn.linear_model import LinearRegression

# ==========================================
# PART 1: The Machine Learning Logic (Brain)
# ==========================================
class EnergyPredictor:
    def __init__(self):
        # In a real scenario, you would load your trained model here:
        # self.model = joblib.load('energy_model.pkl')
        
        # For now, we initialize a Linear Regression model manually
        # This mimics physics: Energy increases with Velocity and Acceleration
        self.model = LinearRegression()
        
        # We manually set weights to simulate a trained state
        # Formula: Energy = 0.5 * Vel + 0.1 * Accel
        self.model.coef_ = np.array([0.5, 0.1]) 
        self.model.intercept_ = 0.0

    def predict_energy(self, velocity, acceleration):
        """
        Inputs: Average Velocity (scalar), Average Acceleration (scalar)
        Output: Predicted Energy (Joules/Step)
        """
        features = np.array([[abs(velocity), abs(acceleration)]])
        return self.model.predict(features)[0]

    def optimize_velocity(self, current_vel, current_accel, target_energy_reduction=0.2):
        """
        Logic: Find a new velocity that lowers energy by X%
        """
        current_energy = self.predict_energy(current_vel, current_accel)
        
        # Goal: Reduce energy by 20%
        desired_energy = current_energy * (1.0 - target_energy_reduction)
        
        if current_energy <= 0.001: return current_vel # Already 0
        
        # Calculate ratio: New_Vel = Old_Vel * (Desired_E / Old_E)
        optimization_ratio = desired_energy / current_energy
        
        # Safety Clamp: Don't slow down more than 50%
        optimization_ratio = max(0.5, min(1.0, optimization_ratio))
        
        new_velocity = current_vel * optimization_ratio
        return new_velocity

# ==========================================
# PART 2: The ROS Node (Communication)
# ==========================================
class SmartEnergyOptimizer(Node):
    def __init__(self):
        super().__init__('energy_optimizer_node')

        # 1. SUBSCRIBE to Inverse Kinematics Output
        # The IK node must publish to '/ik_targets' for this to work
        self.sub_ik = self.create_subscription(
            Float64MultiArray, 
            '/ik_targets', 
            self.ik_callback, 
            10
        )

        # 2. PUBLISH to Robot
        # The robot listens to '/joint_angles'
        self.pub_robot = self.create_publisher(
            Float64MultiArray, 
            '/joint_angles', 
            10
        )

        # Initialize the Brain
        self.brain = EnergyPredictor()

        # State Variables
        self.prev_angles = np.zeros(7)
        self.prev_time = time.time()
        self.initialized = False

        # Configuration
        self.optimization_enabled = True
        
        self.get_logger().info("Optimizer Node Running: Listening to /ik_targets -> Publishing to /joint_angles")

    def ik_callback(self, msg):
        current_time = time.time()
        dt = current_time - self.prev_time
        
        # Safety check for first run or zero time diff
        if dt <= 0.0001: 
            dt = 0.02 # Assume standard 50Hz if dt is bad

        target_angles = np.array(msg.data) # [J1, J2, J3, J4, J5, J6, Suction]

        # 1. Handle Initialization
        if not self.initialized:
            self.prev_angles = target_angles
            self.prev_time = current_time
            self.initialized = True
            self.pub_robot.publish(msg) # Pass through first command
            return

        # 2. Calculate Derivatives (Physics Inputs)
        # We only care about joints 0-6 (ignore suction at index 6)
        joints_target = target_angles[0:6]
        joints_prev   = self.prev_angles[0:6]

        # Velocity = (Change in Position) / Time
        velocities = (joints_target - joints_prev) / dt
        avg_velocity = np.mean(np.abs(velocities))

        # Acceleration = (Velocity) / Time (Approximate)
        avg_acceleration = avg_velocity / dt 

        # 3. PREDICT ENERGY
        predicted_energy = self.brain.predict_energy(avg_velocity, avg_acceleration)

        # 4. OPTIMIZE (The "Parameter Change" Step)
        # If predicted energy is too high, we slow the robot down.
        
        final_angles = target_angles.copy()
        
        # Threshold: 0.5 Joules per step (You can tune this)
        if self.optimization_enabled and predicted_energy > 0.5: 
            
            # Ask ML model for the optimal velocity to save energy
            opt_velocity = self.brain.optimize_velocity(avg_velocity, avg_acceleration)
            
            # Calculate Scaling Factor
            scale = opt_velocity / avg_velocity if avg_velocity > 1e-6 else 1.0
            
            # Apply Scaling: INTERPOLATION
            # Instead of going all the way to 'target', we go part of the way.
            # This makes the robot take smaller steps per loop = Slower Speed = Less Energy.
            joints_optimized = joints_prev + (joints_target - joints_prev) * scale
            
            # Update the message to send
            final_angles[0:6] = joints_optimized
            
            # Debug Print (Optional)
            # print(f"Optimizing: Energy {predicted_energy:.2f} too high. Scaling speed by {scale:.2f}")

        # 5. Send to Robot
        out_msg = Float64MultiArray()
        out_msg.data = final_angles.tolist()
        self.pub_robot.publish(out_msg)

        # 6. Update History
        # We update using 'final_angles' so the next velocity calculation is accurate to what really happened
        self.prev_angles = final_angles 
        self.prev_time = current_time

def main(args=None):
    rclpy.init(args=args)
    node = SmartEnergyOptimizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()