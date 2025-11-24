# 3D Projectile Steering Demo

This mini-game demonstrates a projectile that moves through 3D space with constant speed while the player steers by rotating the acceleration vector around the velocity direction. The result is a smooth, controllable arc that keeps speed and curvature radius predictable. The latest update adds an orbit camera, floating structures, and glowy gates so you can understand the surrounding space while carving ribbons through the air.

## Features

- Constant-speed motion (|v| is fixed)
- Constant-magnitude acceleration that is always perpendicular to velocity
- Orbit camera with yaw/pitch/zoom controls for exploring the world around the projectile
- Floating gates, crystalline beacons, neon floor grid, and pillars for spatial reference
- Real-time visualization of trajectory, velocity, and acceleration vectors plus HUD metrics (speed, |a|, radius of curvature, |w|)

## Requirements

- Python 3.11+
- `pygame`, `numpy`, `pytest` (install via `pip install -r requirements.txt`)

## Running the demo

```bash
pip install -r requirements.txt
python game.py
```

Controls:

- **Left / Right arrows** – Rotate the acceleration vector around the velocity direction
- **A / D** – Orbit camera yaw
- **W / S** – Tilt camera pitch
- **Q / E** – Zoom in/out
- **R / F** – Raise/lower camera height
- **C** – Reset the camera to the default orbit
- **Space** – Clear the trail
- **Esc / Window close** – Quit

## Tests

The physics helpers are covered by a lightweight test that verifies constant speed and perpendicular acceleration.

```bash
python -m pytest
```

## How it works

- `projectile_game.vector_math` implements basic vector utilities and Rodrigues' rotation formula.
- `projectile_game.simulation.Projectile` keeps the projectile speed constant, maintains a perpendicular acceleration of constant magnitude, and exposes `rotate_acceleration` for player control.
- `game.py` runs a Pygame loop that samples input, updates the projectile, and renders its path with an interactive orbit camera, simple line-based models (rings, pillars, crystals), and perspective projection.

At any instant:

- $|v| = \text{const}$
- $|a| = \text{const}$ and $a \perp v$
- $w = v \times a$
- Radius of curvature $r = \frac{|v|^2}{|a|}$

Because $a$ only changes direction via rotation around $v$, the projectile behaves like a steerable centripetal system—perfect for futuristic homing shots or ribbon-like trails.
