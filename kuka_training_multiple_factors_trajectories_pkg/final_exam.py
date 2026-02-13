import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import os
import time
from kuka_training_multiple_factors_trajectories_pkg.hybrid_env import KukaHybridEnv
from stable_baselines3 import PPO

class FinalExamNode(Node):
    def __init__(self):
        super().__init__('final_exam_node')
        self.get_logger().info("Final Exam Node Initialized")

def main(args=None):
    # 1. Initialize ROS
    rclpy.init(args=args)
    
    # We create a dummy node just for logging purposes if needed, 
    # but the Env creates its own internal node manager.
    exam_node = FinalExamNode()

    # 2. Load the Latest Model
    MODEL_STEP = 25856 
    model_path = os.path.expanduser(f"~/mujoco_research/models/PPO/{MODEL_STEP}.zip")
    
    if not os.path.exists(model_path):
        exam_node.get_logger().error(f"❌ Model {MODEL_STEP} not found at {model_path}!")
        return

    exam_node.get_logger().info(f"🎓 LOADING FINAL EXAM: Model {MODEL_STEP}")
    
    # Initialize Environment
    # Note: KukaHybridEnv() internally calls rclpy.create_node, so we share the context
    env = KukaHybridEnv()
    
    # Load Model
    model = PPO.load(model_path, env=env)
    
    results = []
    num_tests = 10 

    print(f"\n--- RUNNING {num_tests} COMPARISON TESTS ---\n")

    total_savings_pct = 0.0

    for i in range(1, num_tests + 1):
        # Generate Random Path
        px, py = 0.95, 0.0
        rng = np.random.default_rng()
        d_angle = rng.uniform(-np.pi, np.pi)
        d_rad = rng.uniform(0.65, 0.95)
        dx = d_rad * np.cos(d_angle)
        dy = d_rad * np.sin(d_angle)

        # --- RUN 1: BASELINE (Standard Factory Settings) ---
        # Speed = 1.0 (Standard), Offsets = 0.0
        action_baseline = np.array([1.0]*6 + [0.0]*6, dtype=np.float32)
        
        env.set_test_path(px, py, dx, dy)
        print(f"Test {i}: Baseline Run...", end="\r")
        
        # Step the environment
        _, _, _, _, _ = env.step(action_baseline)
        energy_base = env.latest_energy
        
        # --- RUN 2: AI (Optimized) ---
        env.set_test_path(px, py, dx, dy)
        print(f"Test {i}: AI Run...      ", end="\r")
        
        obs, _ = env.reset()
        action_ai, _ = model.predict(obs, deterministic=True)
        _, _, _, _, _ = env.step(action_ai)
        energy_ai = env.latest_energy
        
        # --- CALCULATE SCORE ---
        saved_joules = energy_base - energy_ai
        saved_pct = (saved_joules / energy_base) * 100.0
        total_savings_pct += saved_pct
        
        # Get AI decision details
        avg_speed = np.mean(action_ai[0:6])
        
        print(f"Test {i}: Base={energy_base:.0f}J | AI={energy_ai:.0f}J | Saved: {saved_pct:.1f}% | AI Speed: {avg_speed:.2f}x")
        
        results.append({
            "Test": i,
            "Base Energy": energy_base,
            "AI Energy": energy_ai,
            "Saved (%)": saved_pct,
            "AI Avg Speed": avg_speed
        })

    avg_savings = total_savings_pct / num_tests
    print(f"\n\n🏆 FINAL GRADE: The AI reduces energy by average of {avg_savings:.1f}%")
    
    if avg_savings > 15.0:
        print("✅ VERDICT: EXCELLENT. Stop training.")
    elif avg_savings > 5.0:
        print("⚠️ VERDICT: GOOD. Consider training more for perfection.")
    else:
        print("❌ VERDICT: BAD. The model hasn't learned enough yet.")

    # Save results to Excel
    df = pd.DataFrame(results)
    df.to_excel("final_exam_results.xlsx", index=False)
    print("Results saved to final_exam_results.xlsx")

    # Cleanup
    env.close()
    exam_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()