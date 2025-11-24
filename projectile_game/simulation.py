from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .vector_math import (
    Vector,
    magnitude,
    normalize,
    perpendicular_component,
    rotate_about_axis,
    to_vector,
)


@dataclass(slots=True)
class SimulationConfig:
    speed: float = 60.0
    acceleration_magnitude: float = 25.0
    rotation_speed: float = 2.5  # radians per second
    trail_length: int = 2000


@dataclass(slots=True)
class SimulationState:
    position: Vector
    velocity: Vector
    acceleration: Vector

    @property
    def speed(self) -> float:
        return magnitude(self.velocity)

    @property
    def control_axis(self) -> Vector:
        return np.cross(self.velocity, self.acceleration)


class Projectile:
    """Constant-speed projectile with steerable perpendicular acceleration."""

    def __init__(
        self,
        position: Vector,
        velocity: Vector,
        config: SimulationConfig = SimulationConfig(),
        acceleration: Optional[Vector] = None,
    ) -> None:
        self.config = config
        self.state = SimulationState(
            position=to_vector(position),
            velocity=self._init_velocity(velocity, config.speed),
            acceleration=self._init_acceleration(velocity, acceleration, config.acceleration_magnitude),
        )

    def _init_velocity(self, velocity: Vector, speed: float) -> Vector:
        vel = to_vector(velocity)
        if magnitude(vel) == 0:
            raise ValueError("Velocity must be non-zero")
        return normalize(vel) * speed

    def _init_acceleration(
        self,
        velocity: Vector,
        acceleration: Optional[Vector],
        accel_mag: float,
    ) -> Vector:
        vel = to_vector(velocity)
        if acceleration is not None:
            acc = to_vector(acceleration)
        else:
            seed = np.array([0.0, 0.0, 1.0])
            if np.isclose(magnitude(perpendicular_component(seed, vel)), 0.0):
                seed = np.array([0.0, 1.0, 0.0])
            acc = perpendicular_component(seed, vel)
        perp = perpendicular_component(acc, vel)
        norm = magnitude(perp)
        if norm == 0:
            raise ValueError("Acceleration must have a component perpendicular to velocity")
        return normalize(perp) * accel_mag

    def rotate_acceleration(self, direction: float, dt: float) -> None:
        """Rotate acceleration around velocity based on player input direction (-1..1)."""
        if abs(direction) < 1e-6:
            return
        angle = direction * self.config.rotation_speed * dt
        v_hat = normalize(self.state.velocity)
        rotated = rotate_about_axis(self.state.acceleration, v_hat, angle)
        perpendicular = perpendicular_component(rotated, v_hat)
        self.state.acceleration = normalize(perpendicular) * self.config.acceleration_magnitude

    def update(self, dt: float) -> None:
        self.state.velocity = self.state.velocity + self.state.acceleration * dt
        self.state.velocity = normalize(self.state.velocity) * self.config.speed
        corrected = perpendicular_component(self.state.acceleration, self.state.velocity)
        self.state.acceleration = normalize(corrected) * self.config.acceleration_magnitude
        self.state.position = self.state.position + self.state.velocity * dt

    def snapshot(self) -> SimulationState:
        return SimulationState(
            position=self.state.position.copy(),
            velocity=self.state.velocity.copy(),
            acceleration=self.state.acceleration.copy(),
        )
