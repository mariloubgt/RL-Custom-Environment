import gymnasium as gym
import numpy as np
import math

class OrbitalDefenderEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=np.array([-math.pi, -10, -10, -2, -2]),
            high=np.array([math.pi, 10, 10, 2, 2]),
            dtype=np.float32
        )
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.turret_angle = 0.0
        self.asteroid_pos = np.array([8.0, 0.0])
        self.asteroid_vel = np.array([-0.5, 0.0])
        return self._get_obs(), {}
    
    def _get_obs(self):
        return np.array([
            self.turret_angle,
            self.asteroid_pos[0],
            self.asteroid_pos[1],
            self.asteroid_vel[0],
            self.asteroid_vel[1]
        ], dtype=np.float32)
    
    def step(self, action):
        # Simple test environment
        self.turret_angle += 0.1 if action % 2 == 0 else -0.1
        self.asteroid_pos += self.asteroid_vel
        
        # Simple hit detection
        distance = np.linalg.norm(self.asteroid_pos)
        reward = -0.1
        done = False
        
        if distance < 3.0:
            reward = -5.0
            done = True
        elif abs(self.turret_angle) < 0.2 and distance < 5.0:
            reward = 10.0
            done = True
        
        return self._get_obs(), reward, done, False, {}

# Quick test
if __name__ == "__main__":
    env = OrbitalDefenderEnv()
    obs, _ = env.reset()
    print("Test passed! Environment created.")
    print(f"State shape: {obs.shape}")
