# kuka_training_multiple_factors_trajectories_pkg

ROS 2 + MuJoCo package for **energy prediction and energy-aware trajectory optimization** on a **KUKA KR16** pick-and-place task.

It runs a full loop:
1) RL agent proposes **joint speed factors + joint offsets**  
2) A trajectory generator publishes Cartesian waypoints  
3) An IK node converts Pose → 6 joint angles (+ suction)  
4) MuJoCo sim executes and publishes **torque + velocity sensors**  
5) Energy is computed by integrating power and returned as the RL “cost”

---

## Key features
- ✅ ROS 2 nodes (Python / `rclpy`)
- ✅ MuJoCo simulation bridge + suction grasp logic
- ✅ End-to-end pipeline (trajectory → IK → sim → energy → reward)
- ✅ PPO training with auto-resume + TensorBoard logs
- ✅ Linear Regression baseline (“teacher”) for reward shaping

---

## Executables (ROS 2)
These come from `setup.py` (`console_scripts`):

- `robot_sim` — MuJoCo runner + ROS bridge  
- `trajectory_planner` — trajectory generator (home → pick → place → home)  
- `ik_solver` — KR16 inverse kinematics (Pose → joints)  
- `energy_monitor` — energy integration from torque/velocity sensors  
- `train_rl` — PPO training loop (resume mode)

Also included (if present in your repo):
- `benchmark_collector`
- `testing_model`
- `model_status`
- `model_status_graph`

---

## System diagram (data flow)
PPO (train_rl) → Gym Env (hybrid_env) → /rl_trial_config
|
v
Trajectory Generator (trajectory_planner)
publishes Pose → "coordinates"
|
v
IK Solver (ik_solver)
publishes → "joint_angles"
|
v
MuJoCo Runner (robot_sim)
publishes → /robot_sensors (torques + vels)
|
v
Energy Monitor (energy_monitor)
publishes → /energy_result
|
└── back to Gym Env reward


---

## ROS topics

| Topic | Type | Publisher → Subscriber | Notes |
|---|---|---|---|
| `/rl_trial_config` | `std_msgs/Float64MultiArray` | `hybrid_env` → `trajectory_planner`, `robot_sim` | Packet: `[pick_x, pick_y, drop_x, drop_y] + 6 speeds + 6 offsets` |
| `coordinates` | `geometry_msgs/Pose` | `trajectory_planner` → `ik_solver` | `orientation.w` is used as suction (0/1) |
| `joint_angles` | `std_msgs/Float64MultiArray` | `ik_solver` → `robot_sim` | `[th1..th6, suction]` |
| `/robot_sensors` | `std_msgs/Float64MultiArray` | `robot_sim` → `energy_monitor`, `hybrid_env` | `[torques(6), velocities(6)]` |
| `/experiment_state` | `std_msgs/String` | `trajectory_planner` → `energy_monitor` | `"start_optimized"` starts recording, `"stop"` ends + publishes |
| `/energy_result` | `std_msgs/Float64MultiArray` | `energy_monitor` → `hybrid_env` | `[total_energy, placeholder]` |
| `/reset_box` | `std_msgs/String` | `hybrid_env` → `robot_sim` | Reset pickable object position |

---

## Energy model (what is integrated)
Energy monitor integrates:
- Mechanical power: `sum(|torque * velocity|)`
- Electrical loss: `sum(0.1 * torque^2)`
- Total energy: `∫ (mechanical_power + electrical_loss) dt`

---

## RL environment details (important)
- Action space (12D):  
  `[6 speed factors (0.3 → 2.0), 6 joint offsets (-0.05 → 0.05)]`
- A trial publishes `/rl_trial_config`, then waits for `/energy_result`
- Timeout protection: if no energy result within ~25s → energy is set to a large value (penalty)

---

## Requirements
You typically need:
- ROS 2 (with Python `rclpy`)
- Python: `numpy`, `gymnasium`, `stable-baselines3`
- MuJoCo Python: `mujoco` + `mujoco.viewer`

> Install steps depend on your OS + ROS distro.

---

## Installation (ROS 2 workspace)
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/georgeyaccoup/kuka_training_multiple_factors_trajectories_pkg.git
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
