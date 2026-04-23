# Design Document: Three.js Port of Projectile Steering Demo

This document is about **3D Projectile Steering Demo** , a high-performance, single-file web application using **Three.js**.

## Core Concept
The goal is to recreate the constant-speed projectile steering simulation as a standalone HTML file. This mini-game demonstrates a projectile that moves through 3D space with constant speed while the player steers by rotating the acceleration vector around the velocity direction. The result is a smooth, controllable arc that keeps speed and curvature radius predictable.

## Technical Stack
- **Engine**: Three.js (via CDN)
- **Physics**: Custom implementation of Rodrigues' rotation formula in JavaScript.
- **Rendering**: WebGL with a focus on neon aesthetics (bloom, additive blending).
- **Format**: **One huge HTML file** containing all HTML, CSS, and JS.
- **Dependency**: None (Pure Browser/JavaScript).
## Gameplay Loop
The player hunts luminous target gates in a specific sequence. Flying through a highlighted gate banks points and spawns/highlights the next target. Clearing a full set of five gates completes a lap, recording the lap time and tracking the "Best Time" in the HUD. For players who want to focus on the visuals, the **Target Assist (L)** automatically steers the projectile toward the active gate.

## Key Features
- **Constant-speed motion**: Velocity magnitude $|v|$ is fixed normally.
- **Constant-magnitude acceleration**: $|a|$ is fixed and always perpendicular to velocity ($a \perp v$).
- **Dynamic Steering**: Player rotates the acceleration vector around the velocity direction to carve paths.
- **Neon Arena**: Floating gates, crystalline beacons, neon floor grid, and pillars for spatial reference.
- **Real-time HUD**: Visual metrics for Speed, $|a|$, Radius of Curvature, and angular velocity $|w|$.
- **Target Assist**: Lock-to-target mode (Toggle with **L**) that automatically steers toward the current gate.
- **Performance**: Targeted 120 FPS render loop with cached geometry for fluid motion.
- **Objective Loop**: Hunt luminous target gates in order. Fly through a highlighted gate to bank points and trigger the next one. Clear five gates to finish a lap and record your time.


- **Multiple Camera Rigs**:
    - **Free-Orbit**: Manual control for exploration.
    - **Chase**: Follows directly behind the projectile.
    - **Center**: Fixed at the arena center, tracking the projectile.

## Keys and User Controls
### Controls

- **Left / Right arrows** – Rotate the acceleration vector around the velocity direction
- **O / P** – Increase / Decrease the acceleration vector magnitude logarithmically by 1.1
- **J / K** – Increase / Decrease the velocity vector magnitude logarithmically by 1.1


### Camera Controls

- **A / D** – Look around the projectile left / right
  Orbits the camera horizontally so you can see what's ahead of the projectile, behind it, or off to the sides.
- **W / S** – Look at the projectile from above or from ground level
  Tilts the camera up or down — push to a high angle for a god-view of the trail, or sink low for a dramatic chase feel.
- **R / F** – Move the camera up / down in the world
  Shifts the camera's absolute height independently of the orbit angle, useful for clearing terrain or matching the projectile's altitude.
- **Q / E** – Get closer to or further from the projectile
  Zooms the orbit radius in or out, letting you inspect the cone's heading up close or pull back to see the full ribbon of trail.


- **Mouse Controls**: 


### Camera Controls

- **C** – Reset the camera to the default orbit: 
Instantly snaps the camera back to its default orbital vantage point, clearing all manual yaw, pitch, and zoom adjustments.
- **V** – Cycle camera modes (Free → Chase → Center): 
Toggles between three specialized camera rigs—**Free** (manual framing), **Chase** (trailing the projectile), and **Center** (fixed arena vantage tracking).
- **L** – Toggle direction lock / target assist: 
Activates/deactivates the **Autopilot Mode**, which automatically steers the projectile toward the current active gate.

### Micellaneous

- **Space** – Clear the trail
Make sure trail is seen from every angle.
- **Esc** – Close / Reset
- **T** Toggle background theme between 5 variants, 
dark, tokyo night, monokai, gruvbox, gruvbox soft light

## Implementation Details

### 1. Physics Engine
- **State**: Position, Velocity, Acceleration vectors (using `THREE.Vector3`).
- **Constraint**: $|v|$ and $|a|$ remain constant.
- **Steering**: Apply Rodrigues' rotation formula to $a$ using $v$ as the axis.

### 2. Mathematical Foundation
At any instant, the simulation enforces:
- $|v| = \text{const}$ (Constant Speed)
- $|a| = \text{const}$ and $a \perp v$ (Constant Centripetal Acceleration)
- Angular Velocity: $w = v \times a$
- Radius of Curvature: $r = \frac{|v|^2}{|a|}$

Because $a$ only changes direction via rotation around $v$ (using the Rodrigues formula), the projectile behaves like a steerable centripetal system—perfect for futuristic homing shots or ribbon-like trails.

### 3. Scene Components
- **Projectile**: A pointy wireframe geometry (Conic or custom mesh) to indicate heading.
- **Trail**: A `THREE.BufferGeometry` with a sliding window of points to show the path history.
- **Arena**: 
    - Infinite neon grid floor using `THREE.GridHelper`.
    - Beacons/Pillars: Simple geometries with emissive materials.
    - Gates: Torus geometries with glowing materials and collision detection.
- **Lighting**: Emissive materials + UnrealBloomPass for the neon glow effect.

### 3. Single-File Architecture
To achieve the "one huge HTML file" requirement:
- All Three.js libraries and post-processing modules will be included via CDN links.
- CSS will be in `<style>` tags.
- All game logic will reside in a single `<script>` block.
