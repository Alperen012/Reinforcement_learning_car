"""Pygame renderer: the track, the car, its lidar, and a panel explaining the reward."""

from __future__ import annotations

import math
import os

import numpy as np

from .car import car_corners

VIEW = 760  # track view size in px (the 1000 px world is scaled down)
PANEL = 400
GRASS = (34, 52, 38)
ASPHALT = (52, 54, 62)
EDGE = (225, 225, 225)
TEXT = (235, 235, 240)
DIM = (150, 155, 170)
POS = (90, 210, 120)
NEG = (235, 95, 85)


def track_image(track) -> np.ndarray:
    """RGB image ``[y, x, 3]`` of a track: grass, asphalt and white edge lines."""
    m = track.mask
    img = np.empty(m.shape + (3,), dtype=np.uint8)
    img[:] = GRASS
    img[m] = ASPHALT
    inner = m.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)):
        inner &= np.roll(np.roll(m, dy, axis=0), dx, axis=1)
    img[m & ~inner] = EDGE
    return img


class Renderer:
    def __init__(self, human: bool, fps: int = 30):
        if not human:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame

        self.pg = pygame
        pygame.init()
        self.human = human
        self.fps = fps
        size = (VIEW + PANEL, VIEW)
        self.screen = pygame.display.set_mode(size) if human else pygame.Surface(size)
        if human:
            pygame.display.set_caption("Racing Gym")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 16)
        self.big = pygame.font.SysFont("consolas,menlo,monospace", 22, bold=True)
        self._bg_track = None
        self._bg = None
        self.overlay: list[str] = []  # extra lines shown at the top of the panel
        self.show_panel = True

    def _background(self, track):
        if self._bg_track is not track:
            img = track_image(track)
            # start/finish line: checkered strip across the track at index 0
            surf = self.pg.surfarray.make_surface(img.swapaxes(0, 1))
            c, n = track.centerline[0], track.normals[0]
            half = track.width / 2
            for k in range(-int(half), int(half), 6):
                p = (c[0] + n[0] * k, c[1] + n[1] * k)
                col = (240, 240, 240) if (k // 6) % 2 == 0 else (20, 20, 20)
                self.pg.draw.circle(surf, col, (int(p[0]), int(p[1])), 3)
            self._bg = self.pg.transform.smoothscale(surf, (VIEW, VIEW))
            self._bg_track = track
        return self._bg

    def draw(self, env):
        pg = self.pg
        if self.human:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    raise SystemExit(0)
        track, car, p = env.track, env.car, env.car_params
        k = VIEW / track.size
        scr = self.screen
        scr.blit(self._background(track), (0, 0))

        # lidar rays: green when far, red when close
        for d, off in zip(env.lidar, env.ray_offsets):
            frac = d / env.config.ray_length
            col = (int(255 * (1 - frac)), int(80 + 175 * frac), 90)
            a = car.yaw + off
            end = (car.x + d * math.cos(a), car.y + d * math.sin(a))
            pg.draw.line(scr, col, (car.x * k, car.y * k), (end[0] * k, end[1] * k), 1)
            pg.draw.circle(scr, col, (int(end[0] * k), int(end[1] * k)), 3)

        corners = car_corners(car, p)[:4]
        body = (255, 170, 60) if not env.crashed else NEG
        pg.draw.polygon(scr, body, [(x * k, y * k) for x, y in corners])
        nose = (car.x + math.cos(car.yaw) * p.length * 0.5, car.y + math.sin(car.yaw) * p.length * 0.5)
        pg.draw.line(scr, (30, 30, 30), (car.x * k, car.y * k), (nose[0] * k, nose[1] * k), 2)

        if self.show_panel:
            self._panel(env)
        else:  # compact mode (GIFs with several cars): label on the track view itself
            y = 10
            for i, line in enumerate(self.overlay or [env.track.name]):
                font = self.big if i == 0 else self.font
                col = NEG if line == "CRASHED" else TEXT
                shadow = font.render(line, True, (0, 0, 0))
                scr.blit(shadow, (14, y + 2))
                scr.blit(font.render(line, True, col), (12, y))
                y += 30
            scr.blit(self.font.render(f"laps {env.laps}   speed {env.car.speed:4.1f}", True, TEXT), (12, VIEW - 26))
        if self.human:
            pg.display.flip()
            self.clock.tick(self.fps)
            return None
        frame = np.transpose(pg.surfarray.array3d(scr), (1, 0, 2))
        return (frame if self.show_panel else frame[:, :VIEW]).copy()

    def _panel(self, env):
        pg, scr = self.pg, self.screen
        x0 = VIEW
        pg.draw.rect(scr, (18, 20, 28), (x0, 0, PANEL, VIEW))
        y = 12

        def text(s, col=TEXT, font=None):
            nonlocal y
            scr.blit((font or self.font).render(s, True, col), (x0 + 14, y))
            y += 22 if font is None else 30

        for line in self.overlay:
            text(line, (255, 210, 120))
        text(f"{env.track.name}", font=self.big)
        text(f"step {env.steps:5d}   laps {env.laps}")
        text(f"speed {env.car.speed:+5.2f} / {env.car_params.max_speed:.1f} px/step")
        lap_pct = 100 * ((env.distance / env.track.length) % 1.0) if env.distance > 0 else 0.0
        text(f"lap progress {lap_pct:5.1f}%")
        if env.car.slip > 1e-3:
            text("understeer: tyres at grip limit", NEG)
        y += 6
        text("reward this step / episode", DIM)
        for name in env.last_terms:
            now, tot = env.last_terms[name], env.episode_terms[name]
            if env.reward_cfg.__dict__[name] == 0:
                continue
            self._bar(x0 + 14, y + 4, now, 1.0)
            text(f"            {name:<9}{now:+6.2f} {tot:+8.1f}", POS if now >= 0 else NEG)
        total = sum(env.episode_terms.values())
        text(f"            {'return':<9}       {total:+8.1f}")
        y += 6
        text("observation", DIM)
        obs = env._get_obs()
        for name, v in zip(env.obs_names, obs):
            self._bar(x0 + 14, y + 4, float(v), 1.0)
            text(f"            {name:<16}{v:+.2f}", DIM)
            if y > VIEW - 24:
                break

    def _bar(self, x, y, value, scale):
        w = int(min(1.0, abs(value) / scale) * 40)
        col = POS if value >= 0 else NEG
        self.pg.draw.rect(self.screen, (40, 44, 56), (x, y, 80, 12))
        if value >= 0:
            self.pg.draw.rect(self.screen, col, (x + 40, y, w, 12))
        else:
            self.pg.draw.rect(self.screen, col, (x + 40 - w, y, w, 12))

    def close(self):
        if self.human:
            self.pg.display.quit()
