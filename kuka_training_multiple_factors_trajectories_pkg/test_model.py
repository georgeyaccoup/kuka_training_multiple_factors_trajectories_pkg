import rclpy
from robot_trainging_multi_trajectories.hybrid_env import KukaHybridEnv
from stable_baselines3 import PPO
import os

def main():
    # 1. Initialize ROS
    if not rclpy.ok():
        rclpy.init()

    # 2. Define the Model Path
    # [IMPORTANT] Update the number '256' if you want to test a later version (e.g., 512, 1024)
    model_path = os.path.expanduser("~/mujoco_research/models/PPO/512.zip")

    if not os.path.exists(model_path):
        print(f"ERROR: Could not find model at: {model_path}")
        print("Check if the file exists in ~/mujoco_research/models/PPO/")
        return

    print(f"--- LOADING BRAIN: {model_path} ---")

    # 3. Load Environment and Model
    env = KukaHybridEnv()
    
    # Load the trained agent
    model = PPO.load(model_path, env=env)

    # 4. Run the Loop (Inference)
    obs, _ = env.reset()
    
    print("--- ROBOT RUNNING (Ctrl+C to stop) ---")
    
    try:
        while True:
            # predict(obs, deterministic=True) tells the AI:
            # "Do NOT experiment. Just do the best thing you know."
            action, _states = model.predict(obs, deterministic=True)
            
            # Print what the AI decided to do
            print(f"AI Decision -> Speed Factor: {action[0]:.2f} | Height Offset: {action[1]:.2f}")
            
            # Execute the action
            obs, reward, done, truncated, info = env.step(action)
            
            if done:
                print("Trial Complete. Resetting...")
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\nStpped by User.")
        env.close()

if __name__ == '__main__':
    main()