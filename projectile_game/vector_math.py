"""Lightweight vector helpers for 3D projectile dynamics."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

Vector = np.ndarray


def to_vector(value: Iterable[float] | Vector) -> Vector:
    """Convert any iterable to a float64 numpy vector."""
    return np.asarray(list(value), dtype=np.float64)


def magnitude(vec: Vector) -> float:
    return float(np.linalg.norm(vec))


def normalize(vec: Vector) -> Vector:
    norm = magnitude(vec)
    if norm == 0:
        raise ValueError("Cannot normalize the zero vector")
    return vec / norm


def project(vec: Vector, onto: Vector) -> Vector:
    """Project vec onto onto."""
    onto_norm = magnitude(onto)
    if onto_norm == 0:
        return np.zeros(3, dtype=np.float64)
    onto_hat = onto / onto_norm
    return np.dot(vec, onto_hat) * onto_hat


def perpendicular_component(vec: Vector, basis: Vector) -> Vector:
    return vec - project(vec, basis)


def rotate_about_axis(vec: Vector, axis: Vector, angle: float) -> Vector:
    """Rotate vec around axis (Rodrigues' rotation)."""
    axis_norm = magnitude(axis)
    if axis_norm == 0 or abs(angle) < 1e-9:
        return vec.copy()

    k = axis / axis_norm
    v = vec
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (
        v * cos_a
        + np.cross(k, v) * sin_a
        + k * np.dot(k, v) * (1 - cos_a)
    )
