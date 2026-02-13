import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import os
import re
from kuka_training_multiple_factors_trajectories_pkg.hybrid_env import KukaHybridEnv
from stable_baselines3 import PPO

class TrainingProgressLogger(Node):
    def __init__(self):
        super().__init__('training_progress_logger')
        self.get_logger().info("--- Training Progress Logger Started ---")

def main(args=None):
    rclpy.init(args=args)
    logger_node = TrainingProgressLogger()

    # --- 1. SETUP ---
    models_dir = os.path.expanduser("~/mujoco_research/models/PPO")
    if not os.path.exists(models_dir):
        logger_node.get_logger().error(f"❌ No models found at {models_dir}")
        return

    # Find and sort all .zip files numerically (256, 512, ... 1024)
    files = [f for f in os.listdir(models_dir) if f.endswith(".zip")]
    
    # Extract step numbers safely
    model_steps = []
    for f in files:
        try:
            step = int(f.replace(".zip", ""))
            model_steps.append(step)
        except ValueError:
            pass
            
    model_steps.sort() # Ensure we process in order: 256 -> 512 -> ...
    
    if not model_steps:
        logger_node.get_logger().error("No valid numbered model files found.")
        return

    logger_node.get_logger().info(f"Found {len(model_steps)} checkpoints. Generating Learning Curve...")

    # --- 2. GENERATE FIXED TEST SET (The "Exam") ---
    # We use the same 5 scenarios for every model to ensure fair comparison.
    test_scenarios = []
    baseline_energies = []
    
    env = KukaHybridEnv()
    rng = np.random.default_rng(seed=42) # Fixed seed for consistency

    logger_node.get_logger().info("Calculating Baselines for Fixed Test Set...")

    # Create 5 standardized tests
    for i in range(5):
        px, py = 0.95, 0.0
        d_angle = rng.uniform(-np.pi, np.pi)
        d_rad = rng.uniform(0.65, 0.95)
        dx = d_rad * np.cos(d_angle)
        dy = d_rad * np.sin(d_angle)
        
        test_scenarios.append((px, py, dx, dy))
        
        # Run Baseline (Standard Speed 1.0)
        action_base = np.array([1.0]*6 + [0.0]*6, dtype=np.float32)
        env.set_test_path(px, py, dx, dy)
        _, _, _, _, _ = env.step(action_base)
        baseline_energies.append(env.latest_energy)

    total_baseline_energy = sum(baseline_energies)
    logger_node.get_logger().info(f"Baseline Total Energy for Test Set: {total_baseline_energy:.1f} J")

    # --- 3. EVALUATE EACH MODEL ---
    data_rows = []

    for step in model_steps:
        model_path = f"{models_dir}/{step}.zip"
        print(f"Testing Model: {step}...", end="\r")
        
        try:
            # Load Model
            model = PPO.load(model_path, env=env)
            
            current_model_energy_sum = 0.0
            
            # Run the 5 scenarios
            for i, (px, py, dx, dy) in enumerate(test_scenarios):
                env.set_test_path(px, py, dx, dy)
                
                obs, _ = env.reset()
                action_ai, _ = model.predict(obs, deterministic=True)
                _, _, _, _, _ = env.step(action_ai)
                
                current_model_energy_sum += env.latest_energy
            
            # Calculate Savings
            saved_joules = total_baseline_energy - current_model_energy_sum
            avg_saving_pct = (saved_joules / total_baseline_energy) * 100.0
            
            # Log
            data_rows.append({
                "Training Steps": step,
                "Average Energy Saving (%)": avg_saving_pct
            })
            
            print(f"Step {step}: {avg_saving_pct:.2f}% Saving           ")

        except Exception as e:
            logger_node.get_logger().warn(f"Failed to load/test step {step}: {e}")

    # --- 4. SAVE TO EXCEL ---
    output_file = "learning_curve_data.xlsx"
    df = pd.DataFrame(data_rows)
    df.to_excel(output_file, index=False)
    
    logger_node.get_logger().info(f"✅ SUCCESS: Data saved to {output_file}")
    
    env.close()
    logger_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()