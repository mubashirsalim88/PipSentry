import argparse
from agents.ppo_agent import PPOAgent
from environments.forex_env import ForexEnv
from data.preprocessor import load_forex_data
from utils.config import load_config
from models.model_manager import ModelManager
import pandas as pd
import os
import logging
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="PipSentry Forex Trading Bot")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    args = parser.parse_args()
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"ppo_{args.mode}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    
    config = load_config()
    data = load_forex_data()
    
    # Split data: Train on 2023–2024, reserve 2025 for testing
    train_data = data[data.index < '2025-01-01']
    logging.info(f"Training data: {len(train_data)} rows")
    
    env = ForexEnv(
        data=train_data,
        initial_balance=config["initial_balance"],
        daily_loss_limit=config["challenge"]["daily_loss_limit"],
        max_drawdown=config["challenge"]["max_drawdown"]
    )
    
    agent = PPOAgent(env, learning_rate=config["agent"]["learning_rate"])
    model_manager = ModelManager()
    model_name = f"ppo_{config['pair']}_2025"
    
    if args.mode == "train":
        logging.info(f"Training PPO on {config['pair']} at {pd.Timestamp.now(tz='Asia/Kolkata')}")
        agent.train(env, timesteps=config["agent"]["timesteps"])
        model_manager.save_model(agent, model_name, metadata={"pair": config["pair"], "date": "2025-04-14"})
        logging.info(f"Training complete. Model saved to models/saved_models/{model_name}/{model_name}.zip")
    elif args.mode == "test":
        logging.info(f"Testing PPO on {config['pair']} at {pd.Timestamp.now(tz='Asia/Kolkata')}")
        agent.load(f"models/saved_models/{model_name}/{model_name}.zip")
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.predict(state)
            state, reward, done, truncated, info = env.step(action)
            logging.info(f"Step: {env.current_step}, Balance: {info['balance']:.2f}, Equity: {info['equity']:.2f}, Reward: {reward}")

if __name__ == "__main__":
    main()