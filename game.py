from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pygame

from projectile_game.simulation import Projectile, SimulationConfig
from projectile_game.vector_math import magnitude, normalize

WIDTH, HEIGHT = 1100, 720
BACKGROUND_TOP = np.array([12, 18, 45])
BACKGROUND_BOTTOM = np.array([3, 5, 15])
TRAIL_COLOR = (120, 210, 255)
VELOCITY_COLOR = (255, 180, 90)
ACCEL_COLOR = (120, 255, 160)
GRID_COLOR = (40, 90, 130)
PILLAR_COLOR = (60, 180, 200)
RING_COLOR = (255, 120, 200)
CRYSTAL_COLOR = (120, 255, 230)
OBJECTIVE_ACTIVE_COLOR = (255, 220, 140)
OBJECTIVE_INACTIVE_COLOR = (90, 120, 220)
OBJECTIVE_FLASH_COLOR = (255, 200, 120)

GRID_SIZE = 900
GRID_STEP = 80
FPS_TARGET = 120
FOCUS_OFFSET = np.array([0.0, 0.0, 30.0], dtype=np.float64)
UNIT_Z = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _unit_ring_points(segments: int) -> np.ndarray:
    angles = np.linspace(0.0, math.tau, segments, endpoint=False)
    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)
    return np.stack((cos_vals, sin_vals, np.zeros_like(cos_vals)), axis=1)


RING_UNIT_CACHE: dict[int, np.ndarray] = {}


def get_ring_points(center: np.ndarray, radius: float, segments: int) -> np.ndarray:
    if segments not in RING_UNIT_CACHE:
        RING_UNIT_CACHE[segments] = _unit_ring_points(segments)
    unit = RING_UNIT_CACHE[segments]
    return center + unit * radius


def _build_grid_lines() -> list[tuple[np.ndarray, np.ndarray]]:
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for offset in range(-GRID_SIZE, GRID_SIZE + 1, GRID_STEP):
        lines.append(
            (
                np.array([offset, -GRID_SIZE, 0.0], dtype=np.float64),
                np.array([offset, GRID_SIZE, 0.0], dtype=np.float64),
            )
        )
        lines.append(
            (
                np.array([-GRID_SIZE, offset, 0.0], dtype=np.float64),
                np.array([GRID_SIZE, offset, 0.0], dtype=np.float64),
            )
        )
    return lines


GRID_LINES = _build_grid_lines()


AXIS_LINES = [
    (
        np.array([-GRID_SIZE, 0.0, 0.0], dtype=np.float64),
        np.array([GRID_SIZE, 0.0, 0.0], dtype=np.float64),
        (255, 100, 100),
    ),
    (
        np.array([0.0, -GRID_SIZE, 0.0], dtype=np.float64),
        np.array([0.0, GRID_SIZE, 0.0], dtype=np.float64),
        (100, 255, 120),
    ),
    (
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, GRID_SIZE], dtype=np.float64),
        (120, 140, 255),
    ),
]


PILLAR_CORNER_OFFSETS = [
    np.array([40.0, 40.0, 0.0]),
    np.array([-40.0, 40.0, 0.0]),
    np.array([-40.0, -40.0, 0.0]),
    np.array([40.0, -40.0, 0.0]),
]


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


@dataclass
class Camera:
    fov: float = 760.0
    radius: float = 920.0
    yaw: float = math.radians(35.0)
    pitch: float = math.radians(-25.0)
    height_offset: float = 140.0
    focus: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _position: np.ndarray = field(init=False, default_factory=lambda: np.zeros(3))
    _right: np.ndarray = field(init=False, default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    _up: np.ndarray = field(init=False, default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    _forward: np.ndarray = field(init=False, default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    _basis_ready: bool = field(init=False, default=False)

    def update_focus(self, target: np.ndarray, dt: float) -> None:
        smoothing = 6.0
        delta = (target - self.focus) * clamp(smoothing * dt, 0.0, 1.0)
        if np.linalg.norm(delta) > 1e-6:
            self.focus += delta
            self._basis_ready = False

    def position(self) -> np.ndarray:
        x = self.focus[0] + self.radius * math.cos(self.pitch) * math.cos(self.yaw)
        y = self.focus[1] + self.radius * math.cos(self.pitch) * math.sin(self.yaw)
        z = self.focus[2] + self.radius * math.sin(self.pitch) + self.height_offset
        return np.array([x, y, z], dtype=np.float64)

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._basis_ready:
            return self._position, self._right, self._up, self._forward

        position = self.position()
        direction = self.focus - position
        if np.linalg.norm(direction) < 1e-6:
            direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        forward = normalize(direction)
        world_up = UNIT_Z
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            right = np.cross(forward, world_up)
        right = normalize(right)
        up = normalize(np.cross(right, forward))

        self._position = position
        self._right = right
        self._up = up
        self._forward = forward
        self._basis_ready = True
        return position, right, up, forward

    def world_to_camera(self, point: np.ndarray) -> np.ndarray:
        position, right, up, forward = self._basis()
        relative = point - position
        return np.array([
            np.dot(relative, right),
            np.dot(relative, up),
            np.dot(relative, forward),
        ])

    def project(self, point: np.ndarray) -> tuple[int, int, float] | None:
        cam_point = self.world_to_camera(point)
        depth = cam_point[2]
        if depth <= 2.0:
            return None
        scale = self.fov / depth
        x = WIDTH / 2 + cam_point[0] * scale
        y = HEIGHT / 2 - cam_point[1] * scale
        return int(x), int(y), depth

    def handle_input(self, dt: float) -> None:
        keys = pygame.key.get_pressed()
        orbit_speed = 1.8
        pitch_speed = 1.2
        zoom_speed = 420.0
        height_speed = 220.0
        changed = False

        if keys[pygame.K_a]:
            self.yaw -= orbit_speed * dt
            changed = True
        if keys[pygame.K_d]:
            self.yaw += orbit_speed * dt
            changed = True
        if keys[pygame.K_w]:
            self.pitch = clamp(self.pitch + pitch_speed * dt, math.radians(-89), math.radians(85))
            changed = True
        if keys[pygame.K_s]:
            self.pitch = clamp(self.pitch - pitch_speed * dt, math.radians(-89), math.radians(85))
            changed = True
        if keys[pygame.K_q]:
            self.radius = clamp(self.radius - zoom_speed * dt, 300.0, 1600.0)
            changed = True
        if keys[pygame.K_e]:
            self.radius = clamp(self.radius + zoom_speed * dt, 300.0, 1600.0)
            changed = True
        if keys[pygame.K_r]:
            self.height_offset = clamp(self.height_offset + height_speed * dt, 40.0, 420.0)
            changed = True
        if keys[pygame.K_f]:
            self.height_offset = clamp(self.height_offset - height_speed * dt, 40.0, 420.0)
            changed = True

        if changed:
            self._basis_ready = False

    def reset_view(self) -> None:
        self.yaw = math.radians(35.0)
        self.pitch = math.radians(-25.0)
        self.radius = 920.0
        self.height_offset = 140.0
        self._basis_ready = False


@dataclass
class ObjectiveTracker:
    capture_radius: float = 140.0
    pattern_extent: float = 620.0
    altitude_range: tuple[float, float] = (90.0, 420.0)
    count: int = 5
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(1337))
    targets: list[np.ndarray] = field(default_factory=list)
    current_index: int = 0
    score: int = 0
    laps: int = 0
    lap_time: float = 0.0
    best_lap: float | None = None

    def __post_init__(self) -> None:
        if not self.targets:
            self.targets = self._generate_targets()

    def _generate_targets(self) -> list[np.ndarray]:
        positions: list[np.ndarray] = []
        for _ in range(self.count):
            radius = float(self.rng.uniform(220.0, self.pattern_extent))
            angle = float(self.rng.uniform(0.0, math.tau))
            altitude = float(self.rng.uniform(*self.altitude_range))
            x = math.cos(angle) * radius
            y = math.sin(angle) * radius
            positions.append(np.array([x, y, altitude], dtype=np.float64))
        return positions

    @property
    def current_target(self) -> np.ndarray:
        return self.targets[self.current_index]

    def advance_target(self) -> None:
        self.current_index += 1
        if self.current_index >= len(self.targets):
            self.laps += 1
            if self.best_lap is None or self.lap_time < self.best_lap:
                self.best_lap = self.lap_time
            self.targets = self._generate_targets()
            self.current_index = 0
            self.lap_time = 0.0

    def update(self, position: np.ndarray, dt: float) -> np.ndarray | None:
        self.lap_time += dt
        target = self.current_target
        if np.linalg.norm(position - target) <= self.capture_radius:
            capture_point = target.copy()
            self.score += 150 + int(25 * self.current_index)
            self.advance_target()
            return capture_point
        return None


@dataclass
class CaptureFlash:
    position: np.ndarray
    age: float = 0.0
    duration: float = 0.6

    def progress(self) -> float:
        return clamp(self.age / self.duration, 0.0, 1.0)


@dataclass
class ControlState:
    lock_direction: bool = False


def build_background() -> pygame.Surface:
    strip = pygame.Surface((1, HEIGHT))
    for y in range(HEIGHT):
        t = y / max(HEIGHT - 1, 1)
        color = BACKGROUND_TOP * (1 - t) + BACKGROUND_BOTTOM * t
        strip.set_at((0, y), tuple(color.astype(int)))
    return pygame.transform.smoothscale(strip, (WIDTH, HEIGHT))


def draw_line3d(
    surface: pygame.Surface,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, int, int],
    camera: Camera,
    width: int = 1,
    fade: bool = True,
) -> None:
    start_proj = camera.project(start)
    end_proj = camera.project(end)
    if not start_proj or not end_proj:
        return
    sx, sy, sd = start_proj
    ex, ey, ed = end_proj
    shade = 1.0
    if fade:
        depth = (sd + ed) / 2.0
        shade = clamp(1.4 - depth * 0.0012, 0.25, 1.0)
    tinted = tuple(int(c * shade) for c in color)
    pygame.draw.line(surface, tinted, (sx, sy), (ex, ey), width)


def draw_axes(surface: pygame.Surface, camera: Camera) -> None:
    for start, end, color in AXIS_LINES:
        draw_line3d(surface, start, end, color, camera, 2)


def draw_floor_grid(surface: pygame.Surface, camera: Camera) -> None:
    for start, end in GRID_LINES:
        draw_line3d(surface, start, end, GRID_COLOR, camera)


PILLAR_POSITIONS = [
    np.array([450.0, 0.0, 0.0]),
    np.array([-350.0, -280.0, 0.0]),
    np.array([150.0, 320.0, 0.0]),
    np.array([-520.0, 420.0, 0.0]),
]


def _build_pillar_geometry() -> list[tuple[list[np.ndarray], list[np.ndarray]]]:
    geometry: list[tuple[list[np.ndarray], list[np.ndarray]]] = []
    for base in PILLAR_POSITIONS:
        height = 260.0 + (base[0] % 200)
        corners = [base + offset for offset in PILLAR_CORNER_OFFSETS]
        top = [corner + np.array([0.0, 0.0, height]) for corner in corners]
        geometry.append((corners, top))
    return geometry


PILLAR_GEOMETRY = _build_pillar_geometry()


def draw_pillars(surface: pygame.Surface, camera: Camera) -> None:
    for corners, top in PILLAR_GEOMETRY:
        for i in range(4):
            draw_line3d(surface, corners[i], corners[(i + 1) % 4], PILLAR_COLOR, camera)
            draw_line3d(surface, top[i], top[(i + 1) % 4], PILLAR_COLOR, camera)
            draw_line3d(surface, corners[i], top[i], PILLAR_COLOR, camera)


GATE_LEVELS = [120.0, 260.0, 420.0]


def draw_ring(
    surface: pygame.Surface,
    camera: Camera,
    center: np.ndarray,
    radius: float,
    color: tuple[int, int, int] = RING_COLOR,
    segments: int = 48,
) -> None:
    points = get_ring_points(center, radius, segments)
    for i in range(segments):
        draw_line3d(surface, points[i], points[(i + 1) % segments], color, camera, 2)


def draw_rings(surface: pygame.Surface, camera: Camera) -> None:
    for idx, level in enumerate(GATE_LEVELS):
        center = np.array([0.0, 0.0, level + 40 * math.sin(pygame.time.get_ticks() * 0.001 + idx)])
        draw_ring(surface, camera, center, 320.0 + idx * 60.0)


CRYSTAL_POSITIONS = [
    np.array([250.0, -200.0, 180.0]),
    np.array([-420.0, 160.0, 220.0]),
    np.array([120.0, 420.0, 300.0]),
]

CRYSTAL_TIP_OFFSETS = np.array(
    [
        [0.0, 0.0, 90.0],
        [54.0, 0.0, -36.0],
        [-54.0, 0.0, -36.0],
        [0.0, 54.0, -36.0],
        [0.0, -54.0, -36.0],
    ],
    dtype=np.float64,
)

CRYSTAL_EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 3),
    (3, 2),
    (2, 4),
    (4, 1),
]


def draw_crystal(surface: pygame.Surface, camera: Camera, center: np.ndarray) -> None:
    tips = center + CRYSTAL_TIP_OFFSETS
    for start_idx, end_idx in CRYSTAL_EDGES:
        draw_line3d(surface, tips[start_idx], tips[end_idx], CRYSTAL_COLOR, camera, 2)


def draw_crystals(surface: pygame.Surface, camera: Camera) -> None:
    for position in CRYSTAL_POSITIONS:
        wobble = math.sin(pygame.time.get_ticks() * 0.001 + position[0] * 0.01) * 20.0
        draw_crystal(surface, camera, position + UNIT_Z * wobble)


def draw_objective_targets(surface: pygame.Surface, camera: Camera, tracker: ObjectiveTracker) -> None:
    for idx, target in enumerate(tracker.targets):
        prominence = 1.0 if idx == tracker.current_index else 0.4
        color = tuple(
            int(OBJECTIVE_INACTIVE_COLOR[i] * (1 - prominence) + OBJECTIVE_ACTIVE_COLOR[i] * prominence)
            for i in range(3)
        )
        ring_radius = 120.0 if idx == tracker.current_index else 90.0
        draw_ring(surface, camera, target, ring_radius, color=color, segments=60)
        draw_ring(surface, camera, target, ring_radius * 1.2, color=color, segments=60)
    ground_point = target.copy()
    ground_point[2] = 0.0
    draw_line3d(surface, target, ground_point, color, camera, 1)


def draw_capture_flashes(surface: pygame.Surface, camera: Camera, flashes: list[CaptureFlash]) -> None:
    for flash in flashes:
        t = flash.progress()
        if t >= 1.0:
            continue
        glow_radius = 80.0 + 260.0 * t
        pulse_color = tuple(
            int(OBJECTIVE_FLASH_COLOR[i] * (1 - t * 0.5)) for i in range(3)
        )
        draw_ring(surface, camera, flash.position, glow_radius, color=pulse_color, segments=70)


def draw_trail(surface: pygame.Surface, trail: deque[np.ndarray], camera: Camera) -> None:
    if len(trail) < 2:
        return
    projected = [camera.project(point) for point in trail]
    path = [(sx, sy) for item in projected if item for sx, sy, _ in [item]]
    if len(path) >= 2:
        pygame.draw.lines(surface, TRAIL_COLOR, False, path, 2)


def draw_projectile(surface: pygame.Surface, projectile: Projectile, camera: Camera) -> None:
    projection = camera.project(projectile.state.position)
    if not projection:
        return
    px, py, depth = projection
    size = clamp(14_000 / depth, 8.0, 24.0)
    pygame.draw.circle(surface, (255, 255, 255), (px, py), int(size))
    pygame.draw.circle(surface, TRAIL_COLOR, (px, py), int(size * 1.6), 2)


def draw_vectors(surface: pygame.Surface, projectile: Projectile, camera: Camera) -> None:
    position = projectile.state.position
    vel_display = position + normalize(projectile.state.velocity) * 60.0
    acc_display = position + normalize(projectile.state.acceleration) * 90.0
    draw_line3d(surface, position, vel_display, VELOCITY_COLOR, camera, 3, fade=False)
    draw_line3d(surface, position, acc_display, ACCEL_COLOR, camera, 2, fade=False)


def draw_hud(
    surface: pygame.Surface,
    projectile: Projectile,
    camera: Camera,
    font: pygame.font.Font,
    objective: ObjectiveTracker | None = None,
    control_state: ControlState | None = None,
    fps: float | None = None,
) -> None:
    hud_lines = [
        "3D Projectile Steering",
        "← / → rotate acceleration",
        "A/D yaw | W/S pitch | Q/E zoom | R/F camera height | C reset",
        f"Speed: {projectile.config.speed:.1f} u/s | |a|: {projectile.config.acceleration_magnitude:.1f} u/s^2",
        f"Radius: {projectile.config.speed**2 / projectile.config.acceleration_magnitude:.1f} u",
        f"|w|: {magnitude(projectile.state.control_axis):.2f}",
        f"Cam r={camera.radius:.0f} pitch={math.degrees(camera.pitch):.1f}°",
    ]
    if fps is not None:
        hud_lines.append(f"FPS: {fps:.0f}/{FPS_TARGET}")
    if control_state is not None:
        hud_lines.append(
            f"Direction lock: {'ON' if control_state.lock_direction else 'OFF'} (L toggles)"
        )
    if objective:
        best = f"{objective.best_lap:.1f}s" if objective.best_lap is not None else "--"
        hud_lines.append(
            f"Target {objective.current_index + 1}/{len(objective.targets)} | Score {objective.score} | Laps {objective.laps}"
        )
        hud_lines.append(f"Lap time {objective.lap_time:.1f}s | Best {best}")
    for idx, text in enumerate(hud_lines):
        surface.blit(font.render(text, True, (230, 235, 245)), (16, 16 + idx * 20))


def handle_keydown(
    event: pygame.event.Event,
    trail: deque[np.ndarray],
    camera: Camera,
    control_state: ControlState,
) -> bool:
    if event.key == pygame.K_ESCAPE:
        return False
    if event.key == pygame.K_SPACE:
        trail.clear()
    if event.key == pygame.K_c:
        camera.reset_view()
    if event.key == pygame.K_l:
        control_state.lock_direction = not control_state.lock_direction
    return True


def handle_events(trail: deque[np.ndarray], camera: Camera, control_state: ControlState) -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if not handle_keydown(event, trail, camera, control_state):
                return False
    return True


def automatic_rotation_input(
    projectile: Projectile,
    objective: ObjectiveTracker | None = None,
) -> float:
    if objective is None:
        return 0.0
    forward = normalize(projectile.state.velocity)
    to_target = objective.current_target - projectile.state.position
    distance = magnitude(to_target)
    if distance < 1e-3:
        return 0.0
    to_target_dir = normalize(to_target)
    desired_axis = np.cross(forward, to_target_dir)
    axis_norm = magnitude(desired_axis)
    if axis_norm < 1e-6:
        return 0.0
    desired_axis /= axis_norm
    current_axis = projectile.state.control_axis
    current_norm = magnitude(current_axis)
    if current_norm < 1e-6:
        return 0.0
    current_axis = current_axis / current_norm
    sin_term = np.dot(np.cross(current_axis, desired_axis), forward)
    cos_term = clamp(np.dot(current_axis, desired_axis), -1.0, 1.0)
    angle = math.atan2(sin_term, cos_term)
    gain = 1.6
    return clamp(angle * gain, -1.0, 1.0)


def rotation_input_from_keys(
    control_state: ControlState,
    projectile: Projectile,
    objective: ObjectiveTracker | None,
) -> float:
    if control_state.lock_direction:
        return automatic_rotation_input(projectile, objective)
    keys = pygame.key.get_pressed()
    direction = 0.0
    if keys[pygame.K_LEFT]:
        direction += 1.0
    if keys[pygame.K_RIGHT]:
        direction -= 1.0
    return direction


def main() -> None:
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Steerable 3D Projectile")
    font = pygame.font.SysFont("JetBrains Mono", 18)
    clock = pygame.time.Clock()

    background = build_background()

    config = SimulationConfig()
    projectile = Projectile(
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([config.speed, 0.0, 0.0]),
        config=config,
    )

    trail: deque[np.ndarray] = deque(maxlen=config.trail_length)
    camera = Camera()
    camera.focus = projectile.state.position.copy()
    camera._basis_ready = False
    objective = ObjectiveTracker()
    capture_flashes: list[CaptureFlash] = []
    control_state = ControlState()

    running = True
    while running:
        dt = clock.tick(FPS_TARGET) / 1000.0

        running = handle_events(trail, camera, control_state)
        if not running:
            break

        camera.handle_input(dt)
        camera.update_focus(projectile.state.position + FOCUS_OFFSET, dt)

        rotation_input = rotation_input_from_keys(control_state, projectile, objective)
        projectile.rotate_acceleration(rotation_input, dt)
        projectile.update(dt)
        capture_point = objective.update(projectile.state.position, dt)
        if capture_point is not None:
            capture_flashes.append(CaptureFlash(position=capture_point))

        trail.append(projectile.state.position.copy())

        for flash in capture_flashes:
            flash.age += dt
        capture_flashes[:] = [flash for flash in capture_flashes if flash.age < flash.duration]
        fps_display = clock.get_fps()

        screen.blit(background, (0, 0))
        draw_floor_grid(screen, camera)
        draw_axes(screen, camera)
        draw_pillars(screen, camera)
        draw_rings(screen, camera)
        draw_crystals(screen, camera)
        draw_objective_targets(screen, camera, objective)
        draw_capture_flashes(screen, camera, capture_flashes)
        draw_trail(screen, trail, camera)
        draw_projectile(screen, projectile, camera)
        draw_vectors(screen, projectile, camera)
        draw_hud(screen, projectile, camera, font, objective, control_state, fps_display)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
