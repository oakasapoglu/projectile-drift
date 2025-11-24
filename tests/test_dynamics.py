from __future__ import annotations

import numpy as np
import pytest

from projectile_game.simulation import Projectile, SimulationConfig
from projectile_game.vector_math import magnitude


def test_speed_remains_constant():
    config = SimulationConfig(speed=40.0, acceleration_magnitude=15.0)
    projectile = Projectile(
        position=np.zeros(3),
        velocity=np.array([1.0, 0.0, 0.0]),
        config=config,
    )

    initial_speed = magnitude(projectile.state.velocity)
    for _ in range(120):
        projectile.rotate_acceleration(direction=0.5, dt=0.016)
        projectile.update(0.016)

    assert magnitude(projectile.state.velocity) == pytest.approx(initial_speed, rel=1e-4)


def test_acceleration_stays_perpendicular():
    config = SimulationConfig(speed=30.0, acceleration_magnitude=9.0)
    projectile = Projectile(
        position=np.zeros(3),
        velocity=np.array([0.0, 30.0, 0.0]),
        config=config,
    )

    for step in range(200):
        direction = (-1) ** step
        projectile.rotate_acceleration(direction=direction, dt=0.01)
        projectile.update(0.01)
        dot = float(np.dot(projectile.state.velocity, projectile.state.acceleration))
        assert abs(dot) < 1e-6
