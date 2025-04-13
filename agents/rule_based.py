# Rule-based strategies (e.g., MA crossover)
from agents.base_agent import BaseAgent
import numpy as np

class RuleBasedAgent(BaseAgent):
    def __init__(self, env):
        self.env = env
    
    def predict(self, state):
        # State indices after normalization: [open, high, low, close, rsi, macd, signal, atr, balance, position]
        close = state[3] * 0.2 + 1.0  # Denormalize close
        sma20 = self.env.data.iloc[self.env.current_step]["sma20"]
        sma50 = self.env.data.iloc[self.env.current_step]["sma50"]
        
        if sma20 > sma50 and self.env.position == 0:
            return 1  # Buy
        elif sma20 < sma50 and self.env.position != 0:
            return 2  # Sell
        return 0  # Hold
    
    def train(self, env, timesteps):
        pass  # No training
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass