"""Closed race tracks: a smooth centreline plus a constant width.

Everything the environment needs is derived from the centreline:

* ``mask``      boolean drivable-surface raster used for collisions and lidar,
* ``cum_s``     arc length along the centreline, used for the progress reward,
* ``tangents``  / ``normals`` for heading error and lateral offset observations.

The same algorithm is mirrored in ``web/js/sim.js`` so the browser playground
produces identical tracks (see ``tests/test_web_parity.py``).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

WORLD_SIZE = 1000  # tracks live in a 1000 x 1000 px world
SPACING = 4.0  # distance between resampled centreline points (px)


def catmull_rom_closed(points: np.ndarray, samples_per_segment: int = 40) -> np.ndarray:
    """Dense closed Catmull-Rom spline through ``points`` (uniform parameterisation)."""
    p = np.asarray(points, dtype=np.float64)
    n = len(p)
    t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)[:, None]
    t2, t3 = t * t, t * t * t
    out = []
    for i in range(n):
        p0, p1, p2, p3 = p[(i - 1) % n], p[i], p[(i + 1) % n], p[(i + 2) % n]
        out.append(
            0.5
            * (
                2 * p1
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
        )
    return np.concatenate(out)


def resample_closed(points: np.ndarray, spacing: float = SPACING) -> np.ndarray:
    """Resample a closed polyline to (approximately) equal ``spacing``."""
    closed = np.vstack([points, points[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    n = max(8, int(round(total / spacing)))
    targets = np.arange(n) * (total / n)
    x = np.interp(targets, cum, closed[:, 0])
    y = np.interp(targets, cum, closed[:, 1])
    return np.stack([x, y], axis=1)


class Track:
    """A closed track. Construct from control points; use :func:`make_track` for named tracks."""

    def __init__(self, name: str, control_points, width: float, size: int = WORLD_SIZE):
        if width <= 0:
            raise ValueError("track width must be positive")
        self.name = name
        self.width = float(width)
        self.size = int(size)
        self.control_points = np.asarray(control_points, dtype=np.float64)
        if self.control_points.ndim != 2 or self.control_points.shape[1] != 2 or len(self.control_points) < 4:
            raise ValueError("a track needs at least 4 (x, y) control points")

        c = resample_closed(catmull_rom_closed(self.control_points))
        self.centerline = c
        nxt = np.roll(c, -1, axis=0)
        seg = nxt - c
        seg_len = np.linalg.norm(seg, axis=1)
        self.tangents = seg / seg_len[:, None]
        self.normals = np.stack([-self.tangents[:, 1], self.tangents[:, 0]], axis=1)
        self.headings = np.arctan2(self.tangents[:, 1], self.tangents[:, 0])
        self.cum_s = np.concatenate([[0.0], np.cumsum(seg_len)[:-1]])
        self.length = float(seg_len.sum())
        self.mask = self._rasterize()

    # ------------------------------------------------------------------ geometry
    def _rasterize(self) -> np.ndarray:
        """Boolean ``mask[y, x]``: True where the car may drive."""
        r = self.width / 2.0
        ri = int(math.ceil(r))
        oy, ox = np.mgrid[-ri : ri + 1, -ri : ri + 1]
        disc = (ox * ox + oy * oy) <= r * r
        mask = np.zeros((self.size, self.size), dtype=bool)
        for px, py in np.round(self.centerline).astype(int):
            x0, x1 = px - ri, px + ri + 1
            y0, y1 = py - ri, py + ri + 1
            cx0, cy0 = max(x0, 0), max(y0, 0)
            cx1, cy1 = min(x1, self.size), min(y1, self.size)
            if cx0 >= cx1 or cy0 >= cy1:
                continue
            mask[cy0:cy1, cx0:cx1] |= disc[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
        return mask

    def is_drivable(self, x: float, y: float) -> bool:
        ix, iy = math.floor(x), math.floor(y)
        if ix < 0 or iy < 0 or ix >= self.size or iy >= self.size:
            return False
        return bool(self.mask[iy, ix])

    def nearest_index(self, x: float, y: float, hint: int | None = None, window: int = 40) -> int:
        """Closest centreline index. With ``hint`` only a local window is searched (fast path)."""
        n = len(self.centerline)
        if hint is None:
            idx = np.arange(n)
        else:
            idx = np.arange(hint - window, hint + window + 1) % n
        d = (self.centerline[idx, 0] - x) ** 2 + (self.centerline[idx, 1] - y) ** 2
        return int(idx[int(np.argmin(d))])

    def heading_at(self, index: int) -> float:
        return float(self.headings[index % len(self.headings)])

    def index_ahead(self, index: int, distance: float) -> int:
        return (index + int(round(distance / SPACING))) % len(self.centerline)

    def min_clearance(self) -> float:
        """Smallest distance between centreline points that are far apart along the track.

        If this is below the track width, two parts of the track overlap.
        """
        c = self.centerline
        n = len(c)
        # Points further apart along the track than a half-circle of radius `width`
        # should never be closer than `width` in space.
        min_gap = int(math.pi * self.width / SPACING)
        j = np.arange(n)
        best = float("inf")
        for i in range(0, n, 2):
            d = np.hypot(c[:, 0] - c[i, 0], c[:, 1] - c[i, 1])
            gap = np.minimum((j - i) % n, (i - j) % n)
            d = d[gap > min_gap]
            if len(d):
                best = min(best, float(d.min()))
        return best

    def max_curvature(self) -> float:
        """Largest |dheading/ds| (1/px). 1/max_curvature is the tightest corner radius."""
        dh = np.diff(np.unwrap(np.append(self.headings, self.headings[0])))
        return float(np.max(np.abs(dh)) / SPACING)


# ---------------------------------------------------------------------- presets
def _oval() -> Track:
    # The original project's circuit (a rounded rectangle), kept for continuity.
    pts = [(200, 500), (200, 300), (230, 230), (300, 200), (500, 200), (700, 200), (770, 230), (800, 300),
           (800, 500), (800, 700), (770, 770), (700, 800), (500, 800), (300, 800), (230, 770), (200, 700)]
    return Track("oval", pts, width=110)


def _classic() -> Track:
    pts = [(150, 520), (160, 300), (260, 170), (430, 160), (520, 280), (600, 360), (760, 300), (860, 400),
           (850, 600), (720, 700), (760, 840), (600, 880), (420, 820), (330, 700), (200, 760)]
    return Track("classic", pts, width=90)


def _hairpin() -> Track:
    pts = [(130, 500), (150, 220), (300, 130), (420, 220), (450, 520), (520, 600), (580, 520), (600, 220),
           (720, 130), (860, 220), (870, 600), (820, 840), (620, 880), (420, 860), (240, 860), (140, 720)]
    return Track("hairpin", pts, width=80)


def _snake() -> Track:
    pts = [(120, 500), (130, 200), (260, 120), (380, 250), (500, 130), (620, 250), (740, 130), (870, 220),
           (880, 520), (870, 800), (740, 880), (620, 750), (500, 870), (380, 750), (260, 880), (130, 800)]
    return Track("snake", pts, width=85)


PRESETS: dict[str, Callable[[], Track]] = {
    "oval": _oval,
    "classic": _classic,
    "hairpin": _hairpin,
    "snake": _snake,
}


class _Lcg:
    """Tiny deterministic RNG shared with the JavaScript port (numbers must match bit for bit)."""

    def __init__(self, seed: int):
        self.state = (int(seed) * 2654435761 + 1013904223) % 4294967296

    def random(self) -> float:
        self.state = (1664525 * self.state + 1013904223) % 4294967296
        return self.state / 4294967296


def random_track(seed: int, width: float = 80.0, max_tries: int = 50) -> Track:
    """Procedural star-shaped track. Same ``seed`` -> same track, in Python and in the browser.

    Control points sit at evenly spaced angles around the world centre with random
    radii. Candidates whose sides overlap or whose corners are too tight to drive
    are rejected and the generator moves on to the next candidate.
    """
    rng = _Lcg(seed)
    for _ in range(max_tries):
        n = 8 + int(rng.random() * 6)  # 8..13 control points
        pts = []
        for k in range(n):
            ang = 2 * math.pi * (k + 0.35 * (rng.random() - 0.5)) / n
            rad = 200 + 230 * rng.random()
            pts.append((500 + rad * math.cos(ang), 500 + rad * math.sin(ang)))
        track = Track(f"random:{seed}", pts, width=width)
        corner_radius = 1.0 / max(track.max_curvature(), 1e-9)
        if track.min_clearance() > width * 1.15 and corner_radius > 30.0:
            return track
    raise RuntimeError(f"could not generate a valid track for seed {seed}")


def make_track(spec: str, width: float | None = None) -> Track:
    """Build a track from a spec string: a preset name (``"classic"``) or ``"random:<seed>"``."""
    spec = spec.strip().lower()
    if spec.startswith("random:"):
        try:
            seed = int(spec.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"bad track spec {spec!r}; expected 'random:<int>'") from exc
        return random_track(seed, width=width or 80.0)
    if spec not in PRESETS:
        raise ValueError(f"unknown track {spec!r}. Presets: {sorted(PRESETS)}, or 'random:<seed>'")
    track = PRESETS[spec]()
    if width is not None and width != track.width:
        track = Track(track.name, track.control_points, width=width)
    return track
