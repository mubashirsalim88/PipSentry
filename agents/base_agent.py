# Abstract base class for all agents
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def predict(self, state):
        """Predict action given state."""
        pass
    
    @abstractmethod
    def train(self, env, timesteps):
        """Train agent on environment."""
        pass
    
    def save(self, path):
        """Save model to path."""
        pass
    
    def load(self, path):
        """Load model from path."""
        pass