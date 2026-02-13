import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
import mujoco
import mujoco.viewer
import numpy as np
import threading
import time

class MuJoCoRosConnector(Node):
    def __init__(self):
        super().__init__('mujoco_runner')
        self.create_subscription(Float64MultiArray, 'joint_angles', self.control_callback, 10)
        self.create_subscription(String, '/reset_box', self.reset_callback, 10)
        self.create_subscription(Float64MultiArray, '/rl_trial_config', self.config_callback, 10)
        self.sensor_publisher = self.create_publisher(Float64MultiArray, '/robot_sensors', 10)
        
        self.target_angles = np.zeros(7) 
        self.new_command_received = False
        self.reset_requested = False
        self.current_box_pos = [0.95, 0.0, 0.75]

        self.ai_joint_speeds = np.ones(6)  
        self.ai_joint_offsets = np.zeros(6)
        self.smoothed_command = np.zeros(6)

    def control_callback(self, msg):
        if len(msg.data) >= 6:
            self.target_angles = np.array(msg.data) 
            self.new_command_received = True

    def reset_callback(self, msg):
        self.reset_requested = True

    def config_callback(self, msg):
        data = np.array(msg.data)
        if len(data) >= 16:
            # [CRITICAL] Update box position from env (which is now FIXED to 0.95, 0.0)
            self.current_box_pos = [data[0], data[1], 0.75]
            self.ai_joint_speeds = data[4:10]
            self.ai_joint_offsets = data[10:16]
            self.smoothed_command = np.zeros(6)

    def publish_sensors(self, torques, velocities):
        msg = Float64MultiArray()
        msg.data = torques.tolist() + velocities.tolist()
        self.sensor_publisher.publish(msg)

def run_mujoco(ros_node, xml_path):
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    actuator_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f'actuator_{i+1}') for i in range(6)]
    suction_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, 'suction_actuator')
    
    torque_sensor_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, f"torque_{i+1}") for i in range(6)]
    vel_sensor_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, f"vel_{i+1}") for i in range(6)]
    
    ee_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "suction_point")
    box_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pickable_box")
    box_joint_id = m.body_jntadr[box_body_id] if box_body_id != -1 else -1

    step_counter = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            if ros_node.new_command_received:
                for i in range(6):
                    raw_target = ros_node.target_angles[i]
                    optimized_target = raw_target + ros_node.ai_joint_offsets[i]
                    factor = ros_node.ai_joint_speeds[i]
                    
                    if factor > 1.0: factor = 1.0
                    if factor < 0.3: factor = 0.3 

                    ros_node.smoothed_command[i] += factor * (optimized_target - ros_node.smoothed_command[i])
                    
                    if actuator_ids[i] != -1:
                        d.ctrl[actuator_ids[i]] = ros_node.smoothed_command[i]

                if len(ros_node.target_angles) > 6 and suction_id != -1:
                    d.ctrl[suction_id] = ros_node.target_angles[6]

            # --- SUCTION LOGIC ---
            suction_on = ros_node.target_angles[6] > 0.5 if len(ros_node.target_angles) > 6 else False
            if suction_on and ee_site_id != -1 and box_joint_id != -1:
                ee_pos = d.site_xpos[ee_site_id]
                box_pos = d.qpos[m.jnt_qposadr[box_joint_id]:m.jnt_qposadr[box_joint_id]+3]
                dist = np.linalg.norm(ee_pos - box_pos)
                
                # High tolerance for easier training grasp
                if dist < 0.45:
                    qpos_adr = m.jnt_qposadr[box_joint_id]
                    d.qpos[qpos_adr] = ee_pos[0]
                    d.qpos[qpos_adr+1] = ee_pos[1]
                    d.qpos[qpos_adr+2] = ee_pos[2] - 0.06 
                    d.qvel[m.jnt_dofadr[box_joint_id]:m.jnt_dofadr[box_joint_id]+6] = 0

            if ros_node.reset_requested and box_joint_id != -1:
                qpos_adr = m.jnt_qposadr[box_joint_id]
                new_qpos = [ros_node.current_box_pos[0], ros_node.current_box_pos[1], ros_node.current_box_pos[2], 1.0, 0.0, 0.0, 0.0]
                d.qpos[qpos_adr:qpos_adr+7] = new_qpos
                d.qvel[:] = 0 
                mujoco.mj_forward(m, d)
                ros_node.reset_requested = False

            mujoco.mj_step(m, d)
            
            # Pub Sensors
            torques = np.array([d.sensordata[m.sensor_adr[i]] for i in torque_sensor_ids])
            velocities = np.array([d.sensordata[m.sensor_adr[i]] for i in vel_sensor_ids])
            ros_node.publish_sensors(torques, velocities)
            
            # Turbo Render
            step_counter += 1
            if step_counter % 15 == 0:
                viewer.sync() 

def main():
    rclpy.init()
    ros_node = MuJoCoRosConnector()
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()
    xml_path = "/home/george/mujoco_research/kr16_l6/urdf/kuka_final.xml"  
    run_mujoco(ros_node, xml_path)
    if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()