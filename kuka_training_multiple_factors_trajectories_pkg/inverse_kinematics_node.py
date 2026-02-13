import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray
import numpy as np
import math

class KR16IKNode(Node):
    def __init__(self):
        super().__init__('kr16_ik_node')
        self.subscription = self.create_subscription(Pose, 'coordinates', self.cb, 10)
        self.publisher_ = self.create_publisher(Float64MultiArray, 'joint_angles', 10)
        
        # --- Robot Dimensions (Calibrated to XML) ---
        self.d1 = 0.675 
        self.a1 = 0.26  
        
        # [FIX 1] Forearm Update: XML is 0.674, Code was 0.68
        self.a2 = 0.674 
        
        # [FIX 2] Arm Length Update: XML is 0.702, Code was 0.67
        # This 3cm difference was causing the X-axis overshoot!
        self.d4 = 0.702 
        
        # [FIX 3] Gripper Length: 0.158 (Wrist) + 0.076 (Gripper Tip) = 0.234
        # Code was 0.258 (2.4cm gap)
        self.d6 = 0.234

    def cb(self, msg):
        try:
            angles = self.calc_ik(msg.position.x, msg.position.y, msg.position.z)
            
            # --- START FIX ---
            # Get suction value (0.0 or 1.0) passed from input_testing.py
            suction_val = msg.orientation.w 
            
            # Append it to the angles list. Now list is [th1, th2, th3, th4, th5, th6, suction]
            angles.append(suction_val) 
            # --- END FIX ---

            out = Float64MultiArray()
            out.data = angles
            self.publisher_.publish(out)
        # ... (Inside the cb function) ...
        except ValueError as e:
            # [FIX] Don't just pass. Log it!
            self.get_logger().warn(f"Target Unreachable! IK Failed: {e}")
            # Optional: Publish the last valid angles so robot doesn't go limp
            if len(self.last_angles) == 6:
                out = Float64MultiArray()
                out.data = self.last_angles + [msg.orientation.w] # Append suction
                self.publisher_.publish(out)

    def calc_ik(self, tx, ty, tz):
        """
        Calculates joint angles for [theta1, ..., theta6]
        Goal: Reach (tx, ty, tz) with gripper pointing VERTICALLY DOWN.
        """
        
        # --- 1. Wrist Center Calculation ---
        # To point down, the Wrist Center (W) is simply d6 distance *above* the target.
        # W = Target + (0, 0, d6)
        wx = tx
        wy = ty
        wz = tz + self.d6 

        # --- 2. Base Rotation (Theta 1) ---
        theta1 = math.atan2(wy, wx)

        # --- 3. Geometric Parameters for Arm ---
        # r = horizontal distance from Axis 2 to Wrist Center
        r = math.sqrt(wx**2 + wy**2) - self.a1
        # z = vertical distance from Axis 2 to Wrist Center
        z = wz - self.d1
        # S = Distance from Axis 2 to Wrist Center (hypotenuse)
        S = math.sqrt(r**2 + z**2)

        # Safety: Check if target is out of reach
        max_reach = self.a2 + self.d4
        if S > max_reach:
            # Clamp to max reach to prevent math errors, or raise error
            ratio = max_reach / S
            S = max_reach
            r *= ratio
            z *= ratio
            # raise ValueError("Target out of reach")

        # --- 4. Elbow Angle (Theta 3) ---
        # Law of Cosines for the elbow triangle
        cos_el = (self.a2**2 + self.d4**2 - S**2) / (2 * self.a2 * self.d4)
        
        # Clamp value to [-1, 1] to handle float inaccuracies
        cos_el = max(min(cos_el, 1.0), -1.0)
        
        # KUKA Internal angle logic: 
        # Typically, theta3 = pi/2 - internal_angle for "Elbow Up" configuration
        theta3 = (math.pi / 2) - math.acos(cos_el)

        # --- 5. Shoulder Angle (Theta 2) ---
        # beta = angle of the imaginary line S relative to horizontal
        beta = math.atan2(z, r)
        
        # psi = angle between line S and link a2
        cos_psi = (self.a2**2 + S**2 - self.d4**2) / (2 * self.a2 * S)
        cos_psi = max(min(cos_psi, 1.0), -1.0)
        psi = math.acos(cos_psi)

        # Theta 2 determines the lift
        theta2 = (math.pi / 2) - (beta + psi)

        # --- 6. Wrist Orientation (THE FIX) ---
        # We want the tool to point DOWN (-Z World).
        # The sum of angles (Global Pitch) determines the tool inclination.
        # Global Pitch = theta2 + theta3 + theta5
        # We want Global Pitch = -180 degrees (-pi radians) to point down.
        
        theta4 = 0.0 # Keep wrist rotation neutral
        theta6 = 0.0 # Keep tool rotation neutral

        # Calculate Theta 5 to satisfy: theta2 + theta3 + theta5 = -pi
        theta5 = -math.pi - (theta2 + theta3)

        # --- 7. Joint Limit Handling (Optional but Recommended) ---
        # If theta5 is too negative (e.g., -200 degrees), we can add 360 (2*pi) 
        # to flip it to the positive range, or adjust the "Down" definition to +pi.
        # KUKA wrist limits are typically +/- 120 approx.
        if theta5 < -math.pi:
             theta5 += 2 * math.pi
        if theta5 > math.pi:
             theta5 -= 2 * math.pi
        theta5-=math.pi/2

        return [theta1, theta2, theta3, theta4, theta5, theta6]

def main(args=None):
    rclpy.init(args=args)
    node = KR16IKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()