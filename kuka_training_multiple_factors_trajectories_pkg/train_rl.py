from kuka_training_multiple_factors_trajectories_pkg.hybrid_env import KukaHybridEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
import time
import sys

# Create saving directories
models_dir = "models/PPO"
log_dir = "logs"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# --- PROGRESS & ETA CALLBACK ---
class ProgressCallback(BaseCallback):
    def __init__(self, batch_size, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.batch_size = batch_size
        self.start_time = None

    def _on_step(self) -> bool:
        # Initialize start time on the very first step
        if self.start_time is None:
            self.start_time = time.time()
            return True

        # 1. Calculate Elapsed Time
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # 2. Get Steps Completed
        steps_done = self.n_calls 
        
        # 3. Calculate Speed & ETA
        if steps_done > 0:
            avg_time_per_step = elapsed_time / steps_done
            steps_remaining = self.batch_size - steps_done
            
            if steps_remaining < 0: steps_remaining = 0
            
            eta_seconds = steps_remaining * avg_time_per_step
            eta_min = int(eta_seconds // 60)
            eta_sec = int(eta_seconds % 60)

            # 4. Print EXACT Requested Format with FLUSH
            # flush=True forces the terminal to show the text immediately
            msg = f"steps : {steps_done}/{self.batch_size}, Time remining : {eta_min}m {eta_sec}s untill saving   "
            print(msg, end='\r', flush=True)

        return True

def main():
    print("--- STARTING HYBRID RL TRAINING (RESUME MODE) ---")
    
    # 1. Initialize Environment
    env = KukaHybridEnv()
    
    # 2. Check for existing models to RESUME
    # Get all .zip files in the folder
    existing_models = []
    if os.path.exists(models_dir):
        files = os.listdir(models_dir)
        for f in files:
            if f.endswith(".zip"):
                try:
                    # Extract number from "1024.zip" -> 1024
                    step_num = int(f.replace(".zip", ""))
                    existing_models.append(step_num)
                except ValueError:
                    pass
    
    # 3. Load or Create Model
    BATCH_SIZE = 256
    start_batch = 1

    if existing_models:
        # Find the latest model
        latest_step = max(existing_models)
        model_path = f"{models_dir}/{latest_step}.zip"
        
        # [ADDED] Explicit Success Message
        print(f"\n SUCCESS: Found existing model: {model_path}")
        print(f" RESUMING training from step {latest_step}...\n")
        
        # LOAD the model
        model = PPO.load(model_path, env=env, tensorboard_log=log_dir)
        
        # Calculate where to start the loop
        start_batch = (latest_step // BATCH_SIZE) + 1
    else:
        # [ADDED] Explicit New Training Message
        print("\n No saved models found.")
        print(" Starting FRESH training...\n")
        
        # CREATE a new model
        model = PPO('MlpPolicy', env, verbose=0, tensorboard_log=log_dir, n_steps=256)

    # 4. Training Loop
    for i in range(start_batch, 10000): # Increased limit
        print(f"\n\n--- STARTING BATCH {i} ---")
        
        callback = ProgressCallback(batch_size=BATCH_SIZE)
        
        model.learn(
            total_timesteps=BATCH_SIZE, 
            reset_num_timesteps=False,  # CRITICAL: Keeps the internal counter running
            tb_log_name="PPO",
            callback=callback
        )
        
        current_step = BATCH_SIZE * i
        model.save(f"{models_dir}/{current_step}")
        print(f"\n [SAVED] Model saved at step {current_step}")

if __name__ == '__main__':
    main()