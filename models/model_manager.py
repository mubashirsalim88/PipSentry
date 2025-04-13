import os
import json
from stable_baselines3 import PPO

class ModelManager:
    def __init__(self, save_dir="models/saved_models"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_model(self, agent, name, metadata=None):
        """Save agent model with metadata in its own folder."""
        model_dir = os.path.join(self.save_dir, name)
        os.makedirs(model_dir, exist_ok=True)  # Create folder if it doesn’t exist
        path = os.path.join(model_dir, f"{name}.zip")
        agent.save(path)
        if metadata:
            with open(os.path.join(model_dir, f"{name}_metadata.json"), "w") as f:
                json.dump(metadata, f)
    
    def load_model(self, agent_class, name):
        """Load model from its folder."""
        path = os.path.join(self.save_dir, name, f"{name}.zip")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model {path} not found")
        return agent_class.load(path)