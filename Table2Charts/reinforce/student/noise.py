import numpy as np
from threading import Lock


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = mu * np.ones(size)
        self.reset()
        self.lock = Lock()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu * np.ones(self.size)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        with self.lock:
            x = self.state
            dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
            self.state = x + dx
            return self.state
