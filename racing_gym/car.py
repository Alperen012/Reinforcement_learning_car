"""Car dynamics: a kinematic bicycle model with a tyre-grip limit.

Deliberately simple (one function, no physics engine) so that it is easy to
read, easy to port (``web/js/sim.js`` mirrors it line by line) and fast enough
to simulate thousands of steps per second in pure Python.

Units: positions in px, time in steps (the renderer shows 30 steps per second).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import CarConfig


@dataclass
class CarState:
    x: float
    y: float
    yaw: float  # radians, 0 = +x, pi/2 = +y (screen down)
    speed: float = 0.0  # px/step along the heading, negative = reversing
    steer: float = 0.0  # current front wheel angle, radians
    slip: float = 0.0  # yaw rate the tyres could not deliver this step (understeer), rad/step


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def step_car(s: CarState, steer_cmd: float, throttle: float, p: CarConfig) -> None:
    """Advance ``s`` in place by one step. ``steer_cmd`` and ``throttle`` are in [-1, 1]."""
    steer_cmd = min(1.0, max(-1.0, steer_cmd))
    throttle = min(1.0, max(-1.0, throttle))

    # Steering: the wheel turns towards the command at a limited rate.
    max_steer = math.radians(p.max_steer_deg)
    rate = math.radians(p.steer_rate_deg)
    s.steer += min(rate, max(-rate, steer_cmd * max_steer - s.steer))

    # Longitudinal: throttle, brake (negative throttle while rolling forward) or reverse.
    if throttle >= 0.0:
        accel = throttle * p.engine_accel
    elif s.speed > 0.0:
        accel = throttle * p.brake_accel
    else:
        accel = throttle * p.engine_accel
    v = s.speed
    drag = 0.0 if abs(v) < 1e-9 else math.copysign(p.rolling_drag, v) + p.aero_drag * v * abs(v)
    new_v = v + accel - drag
    if accel == 0.0 and v != 0.0 and (new_v > 0.0) != (v > 0.0):
        new_v = 0.0  # drag stops the car, it never pushes it backwards
    s.speed = min(p.max_speed, max(-p.max_reverse_speed, new_v))

    # Lateral: the yaw rate the geometry asks for, limited by tyre grip (a = v * yaw_rate <= grip).
    yaw_rate = s.speed / p.wheel_base * math.tan(s.steer)
    if abs(s.speed) > 1e-9:
        limit = p.grip / abs(s.speed)
        clamped = min(limit, max(-limit, yaw_rate))
        s.slip = yaw_rate - clamped
        yaw_rate = clamped
    else:
        s.slip = 0.0
    s.yaw = wrap_angle(s.yaw + yaw_rate)
    s.x += s.speed * math.cos(s.yaw)
    s.y += s.speed * math.sin(s.yaw)


def car_corners(s: CarState, p: CarConfig) -> list[tuple[float, float]]:
    """Four corners plus the front-centre point of the collision box."""
    c, sn = math.cos(s.yaw), math.sin(s.yaw)
    hl, hw = p.length / 2.0, p.width / 2.0
    pts = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw), (hl, 0.0)]
    return [(s.x + lx * c - ly * sn, s.y + lx * sn + ly * c) for lx, ly in pts]
