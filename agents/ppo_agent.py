from stable_baselines3 import PPO
from agents.base_agent import BaseAgent
import logging
from stable_baselines3.common.callbacks import BaseCallback

class RolloutLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RolloutLoggerCallback, self).__init__(verbose)
        self.action_counts = {0: 0, 1: 0, 2: 0}
    
    def _on_step(self) -> bool:
        action = self.locals["actions"][0]
        self.action_counts[action] += 1
        return True
    
    def _on_rollout_end(self) -> None:
        ep_rew_mean = self.locals.get("rollout_buffer").rewards.mean()
        total_actions = sum(self.action_counts.values())
        action_pct = {k: v/total_actions*100 for k, v in self.action_counts.items()}
        logging.info(f"Rollout {self.n_calls}: ep_rew_mean = {ep_rew_mean:.4f}, actions = {self.action_counts}, action_pct = {action_pct}")
        self.action_counts = {0: 0, 1: 0, 2: 0}
        return True

class PPOAgent(BaseAgent):
    def __init__(self, env, learning_rate=0.0001, policy="MlpPolicy"):
        self.model = PPO(
            policy,
            env,
            learning_rate=learning_rate,
            verbose=1,
            ent_coef=0.01
        )
    
    def predict(self, state, deterministic=False):
        action, _ = self.model.predict(state, deterministic=deterministic)
        return int(action.item())
    
    def train(self, env, timesteps=500000):  # More timesteps
        self.model.set_env(env)
        callback = RolloutLoggerCallback()
        self.model.learn(total_timesteps=timesteps, log_interval=1, callback=callback)
    
    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        self.model = PPO.load(path, env=self.model.env if hasattr(self.model, 'env') else None)