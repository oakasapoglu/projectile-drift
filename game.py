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

GRID_SIZE = 900
GRID_STEP = 80


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

    def update_focus(self, target: np.ndarray, dt: float) -> None:
        smoothing = 6.0
        self.focus += (target - self.focus) * clamp(smoothing * dt, 0.0, 1.0)

    def position(self) -> np.ndarray:
        x = self.focus[0] + self.radius * math.cos(self.pitch) * math.cos(self.yaw)
        y = self.focus[1] + self.radius * math.cos(self.pitch) * math.sin(self.yaw)
        z = self.focus[2] + self.radius * math.sin(self.pitch) + self.height_offset
        return np.array([x, y, z], dtype=np.float64)

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        position = self.position()
        forward = normalize(self.focus - position)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, world_up)
        right = normalize(right)
        up = normalize(np.cross(right, forward))
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

        if keys[pygame.K_a]:
            self.yaw -= orbit_speed * dt
        if keys[pygame.K_d]:
            self.yaw += orbit_speed * dt
        if keys[pygame.K_w]:
            self.pitch = clamp(self.pitch + pitch_speed * dt, math.radians(-89), math.radians(85))
        if keys[pygame.K_s]:
            self.pitch = clamp(self.pitch - pitch_speed * dt, math.radians(-89), math.radians(85))
        if keys[pygame.K_q]:
            self.radius = clamp(self.radius - zoom_speed * dt, 300.0, 1600.0)
        if keys[pygame.K_e]:
            self.radius = clamp(self.radius + zoom_speed * dt, 300.0, 1600.0)
        if keys[pygame.K_r]:
            self.height_offset = clamp(self.height_offset + height_speed * dt, 40.0, 420.0)
        if keys[pygame.K_f]:
            self.height_offset = clamp(self.height_offset - height_speed * dt, 40.0, 420.0)

    def reset_view(self) -> None:
        self.yaw = math.radians(35.0)
        self.pitch = math.radians(-25.0)
        self.radius = 920.0
        self.height_offset = 140.0


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
    origin = np.zeros(3)
    axis_vectors = {
        "x": np.array([GRID_SIZE, 0, 0]),
        "y": np.array([0, GRID_SIZE, 0]),
        "z": np.array([0, 0, GRID_SIZE]),
    }
    colors = {
        "x": (255, 100, 100),
        "y": (100, 255, 120),
        "z": (120, 140, 255),
    }
    for axis, vec in axis_vectors.items():
        draw_line3d(surface, origin - vec, origin + vec, colors[axis], camera, 2)


def draw_floor_grid(surface: pygame.Surface, camera: Camera) -> None:
    for offset in range(-GRID_SIZE, GRID_SIZE + 1, GRID_STEP):
        start_x = np.array([offset, -GRID_SIZE, 0])
        end_x = np.array([offset, GRID_SIZE, 0])
        start_y = np.array([-GRID_SIZE, offset, 0])
        end_y = np.array([GRID_SIZE, offset, 0])
        draw_line3d(surface, start_x, end_x, GRID_COLOR, camera)
        draw_line3d(surface, start_y, end_y, GRID_COLOR, camera)


PILLAR_POSITIONS = [
    np.array([450.0, 0.0, 0.0]),
    np.array([-350.0, -280.0, 0.0]),
    np.array([150.0, 320.0, 0.0]),
    np.array([-520.0, 420.0, 0.0]),
]


def draw_pillars(surface: pygame.Surface, camera: Camera) -> None:
    for base in PILLAR_POSITIONS:
        height = 260.0 + (base[0] % 200)
        corners = [
            base + np.array([40, 40, 0]),
            base + np.array([-40, 40, 0]),
            base + np.array([-40, -40, 0]),
            base + np.array([40, -40, 0]),
        ]
        top = [c + np.array([0, 0, height]) for c in corners]
        for i in range(4):
            draw_line3d(surface, corners[i], corners[(i + 1) % 4], PILLAR_COLOR, camera)
            draw_line3d(surface, top[i], top[(i + 1) % 4], PILLAR_COLOR, camera)
            draw_line3d(surface, corners[i], top[i], PILLAR_COLOR, camera)


GATE_LEVELS = [120.0, 260.0, 420.0]


def draw_ring(surface: pygame.Surface, camera: Camera, center: np.ndarray, radius: float, segments: int = 48) -> None:
    points = []
    for i in range(segments):
        angle = (i / segments) * math.tau
        points.append(
            center + np.array([math.cos(angle) * radius, math.sin(angle) * radius, 0.0])
        )
    for i in range(segments):
        draw_line3d(surface, points[i], points[(i + 1) % segments], RING_COLOR, camera, 2)


def draw_rings(surface: pygame.Surface, camera: Camera) -> None:
    for idx, level in enumerate(GATE_LEVELS):
        center = np.array([0.0, 0.0, level + 40 * math.sin(pygame.time.get_ticks() * 0.001 + idx)])
        draw_ring(surface, camera, center, 320.0 + idx * 60.0)


CRYSTAL_POSITIONS = [
    np.array([250.0, -200.0, 180.0]),
    np.array([-420.0, 160.0, 220.0]),
    np.array([120.0, 420.0, 300.0]),
]


def draw_crystal(surface: pygame.Surface, camera: Camera, center: np.ndarray, size: float = 90.0) -> None:
    tips = [
        center + np.array([0.0, 0.0, size]),
        center + np.array([size * 0.6, 0.0, -size * 0.4]),
        center + np.array([-size * 0.6, 0.0, -size * 0.4]),
        center + np.array([0.0, size * 0.6, -size * 0.4]),
        center + np.array([0.0, -size * 0.6, -size * 0.4]),
    ]
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 3),
        (3, 2),
        (2, 4),
        (4, 1),
    ]
    for start_idx, end_idx in edges:
        draw_line3d(surface, tips[start_idx], tips[end_idx], CRYSTAL_COLOR, camera, 2)


def draw_crystals(surface: pygame.Surface, camera: Camera) -> None:
    for position in CRYSTAL_POSITIONS:
        wobble = math.sin(pygame.time.get_ticks() * 0.001 + position[0] * 0.01) * 20.0
        draw_crystal(surface, camera, position + np.array([0.0, 0.0, wobble]))


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
    for idx, text in enumerate(hud_lines):
        surface.blit(font.render(text, True, (230, 235, 245)), (16, 16 + idx * 20))


def handle_events(trail: deque[np.ndarray], camera: Camera) -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            if event.key == pygame.K_SPACE:
                trail.clear()
            if event.key == pygame.K_c:
                camera.reset_view()
    return True


def rotation_input_from_keys() -> float:
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

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        running = handle_events(trail, camera)
        if not running:
            break

        camera.handle_input(dt)
        camera.update_focus(projectile.state.position + np.array([0.0, 0.0, 30.0]), dt)

        rotation_input = rotation_input_from_keys()
        projectile.rotate_acceleration(rotation_input, dt)
        projectile.update(dt)

        trail.append(projectile.state.position.copy())

        screen.blit(background, (0, 0))
        draw_floor_grid(screen, camera)
        draw_axes(screen, camera)
        draw_pillars(screen, camera)
        draw_rings(screen, camera)
        draw_crystals(screen, camera)
        draw_trail(screen, trail, camera)
        draw_projectile(screen, projectile, camera)
        draw_vectors(screen, projectile, camera)
        draw_hud(screen, projectile, camera, font)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
