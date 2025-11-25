# 3D Projectile Steering Demo

A “vibe coding” uniform circular motion simulator with changing centers. 

This mini-game demonstrates a projectile that moves through 3D space with constant speed while the player steers by rotating the acceleration vector around the velocity direction. The result is a smooth, controllable arc that keeps speed and curvature radius predictable. The latest update adds swappable camera rigs plus a pointy projectile silhouette so you can instantly read heading while carving ribbons through the air.

## Features

- Constant-speed motion (|v| is fixed)
- Constant-magnitude acceleration that is always perpendicular to velocity
- Multiple camera rigs: free-orbit (manual), chase (always behind the projectile), and a center-of-arena view you can cycle with **V**
- Floating gates, crystalline beacons, neon floor grid, and pillars for spatial reference
- Real-time visualization of trajectory, velocity, and acceleration vectors plus HUD metrics (speed, |a|, radius of curvature, |w|)
- Pointy wireframe projectile that shows the current heading angle while still behaving like the classic glowing orb
- Objective loop with glowing energy gates to chase, earn score, and set lap records
- Lock-to-target mode (press **L**) that automatically steers the projectile toward the current gate
- Cached geometry and a 120 FPS render loop for smoother flight and lower CPU overhead

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
- **C** – Reset the camera to the default orbit (also snaps back to Free view)
- **V** – Cycle camera modes (Free → Chase → Center)
- **L** – Toggle direction lock / target assist
- **Space** – Clear the trail
- **Esc / Window close** – Quit

## Objective

Hunt the luminous target gates in order. Fly through the highlighted gate to bank points, then chase the next one. Clear all five gates to finish a lap—your score climbs, the arena respawns a fresh pattern, and the HUD records your lap time (with best-time tracking) so you can perfect your ribbon line. Need a hand? Toggle **L** to lock the controls so the projectile automatically steers toward the current gate while you focus on camera framing. The render loop now targets 120 FPS and reuses cached geometry so the action stays fluid even with all the neon set dressing enabled.

## Tests

The physics helpers are covered by a lightweight test that verifies constant speed and perpendicular acceleration.

```bash
python -m pytest
```

## How it works

- `projectile_game.vector_math` implements basic vector utilities and Rodrigues' rotation formula.
- `projectile_game.simulation.Projectile` keeps the projectile speed constant, maintains a perpendicular acceleration of constant magnitude, and exposes `rotate_acceleration` for player control.
- `game.py` runs a Pygame loop that samples input, updates the projectile, renders its path with an interactive orbit camera, simple line-based models (rings, pillars, crystals), and keeps track of the floating target sequence for scoring.

At any instant:

- $|v| = \text{const}$
- $|a| = \text{const}$ and $a \perp v$
- $w = v \times a$
- Radius of curvature $r = \frac{|v|^2}{|a|}$

Because $a$ only changes direction via rotation around $v$, the projectile behaves like a steerable centripetal system—perfect for futuristic homing shots or ribbon-like trails.
