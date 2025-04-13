# Configs (pairs, timeframes, challenge rules)
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Default config (written to config.yaml below)
DEFAULT_CONFIG = {
    "pair": "EURUSD",
    "timeframe": "1H",
    "initial_balance": 10000,
    "challenge": {
        "profit_target": 0.08,
        "daily_loss_limit": 0.05,
        "max_drawdown": 0.10,
        "min_trading_days": 5
    },
    "agent": {
        "learning_rate": 0.0001,
        "timesteps": 100000
    }
}