import gymnasium as gym
import numpy as np
import pandas as pd

class ForexEnv(gym.Env):
    def __init__(self, data, initial_balance=10000, daily_loss_limit=0.05, max_drawdown=0.10, spread=1.0):
        super().__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.daily_loss_limit = daily_loss_limit
        self.max_drawdown = max_drawdown
        self.spread = spread
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.equity = initial_balance
        self.max_equity = initial_balance
        self.daily_pnl = 0
        
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
    
    def step(self, action):
        done = False
        reward = 0
        
        current_price = self.data.iloc[self.current_step]["close"]
        next_price = self.data.iloc[self.current_step + 1]["close"] if self.current_step + 1 < len(self.data) else current_price
        
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price + (self.spread * 0.0001)
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = current_price - (self.spread * 0.0001)
        elif action in [1, 2] and self.position != 0:
            if self.position == 1:
                profit = (current_price - self.entry_price) * 10000
            else:
                profit = (self.entry_price - current_price) * 10000
            self.balance += profit
            self.equity = self.balance
            self.daily_pnl += profit
            if profit > 30:
                reward = (profit / self.initial_balance) * 100
            elif profit > 0:
                reward = (profit / self.initial_balance) * 60
            else:
                reward = (profit / self.initial_balance) * 80
            if action == 2 and profit > 0:
                reward *= 1.2  # Boost sell wins
            print(f"Step {self.current_step}, Action: {action}, Profit: {profit}, Reward: {reward}")
            self.position = 0
        
        self.current_step += 1
        self.equity = self.balance + (next_price - self.entry_price) * 10000 * self.position
        self.max_equity = max(self.max_equity, self.equity)
        
        daily_loss = -self.daily_pnl / self.initial_balance
        drawdown = (self.max_equity - self.equity) / self.max_equity
        if daily_loss > self.daily_loss_limit or drawdown > self.max_drawdown:
            done = True
            reward = -20.0
        
        if self.current_step >= len(self.data) - 1:
            done = True
        
        if self.position != 0:
            unrealized = (next_price - self.entry_price) * 10000 * self.position
            reward += unrealized / self.initial_balance * 5 if unrealized > 15 else 0
            print(f"Step {self.current_step}, Action: {action}, Unrealized: {unrealized}, Reward: {reward}")
        
        state = self._get_state()
        info = {"balance": self.balance, "equity": self.equity, "drawdown": drawdown}
        return state, reward, done, False, info
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.max_equity = self.initial_balance
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.daily_pnl = 0
        return self._get_state(), {}
    
    def _get_state(self):
        row = self.data.iloc[self.current_step]
        state = np.array([
            row["open"], row["high"], row["low"], row["close"],
            row["rsi"], row["macd"], row["signal"], row["atr"],
            self.balance / self.initial_balance, self.position
        ], dtype=np.float32)
        state[0:4] = (state[0:4] - 1.0) / 0.2
        state[4] /= 100.0
        state[5:7] /= 0.01
        state[7] /= 0.001
        state[8] = state[8]
        state[9] = (state[9] + 1) / 2
        return state