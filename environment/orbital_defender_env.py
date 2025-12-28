import gymnasium as gym
import numpy as np
import math

class OrbitalDefenderEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Actions: rotate left, rotate right, fire
        self.action_space = gym.spaces.Discrete(3)

        # Observation:
        # turret_angle +
        # 2 closest asteroids (angle, distance, angular_velocity)
        self.observation_space = gym.spaces.Box(
            low=np.array([-math.pi] + [-math.pi, 0.0, -1.0] * 2),
            high=np.array([ math.pi] + [ math.pi, 10.0,  1.0] * 2),
            dtype=np.float32
        )

        self.planet_radius = 2.0
        self.max_asteroids = 5
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.turret_angle = 0.0
        self.steps = 0

        # Create asteroids
        self.asteroids = []
        for _ in range(self.max_asteroids):
            self.asteroids.append({
                "angle": np.random.uniform(-math.pi, math.pi),
                "distance": np.random.uniform(6.0, 10.0),
                "angular_velocity": np.random.uniform(-0.2, 0.2)
            })

        return self._get_obs(), {}

    def _get_obs(self):
        # Sort asteroids by distance (closest first)
        asts = sorted(self.asteroids, key=lambda a: a["distance"])[:2]

        obs = [self.turret_angle]

        for a in asts:
            obs.extend([a["angle"], a["distance"], a["angular_velocity"]])

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = -0.01
        terminated = False

        # Rotate turret
        if action == 0:
            self.turret_angle -= 0.1
        elif action == 1:
            self.turret_angle += 0.1

        self.turret_angle = (self.turret_angle + math.pi) % (2 * math.pi) - math.pi

        # Move asteroids
        for a in self.asteroids:
            a["angle"] += a["angular_velocity"]
            a["distance"] -= 0.05

            # Planet impact
            if a["distance"] <= self.planet_radius:
                reward = -10.0
                terminated = True

        # Fire action
        if action == 2:
            for a in self.asteroids:
                angle_diff = abs(self.turret_angle - a["angle"])
                if angle_diff < 0.15 and a["distance"] < 6.0:
                    reward = 10.0
                    self.asteroids.remove(a)
                    break

        # Episode limit
        if self.steps >= 300:
            terminated = True

        return self._get_obs(), reward, terminated, False, {}
