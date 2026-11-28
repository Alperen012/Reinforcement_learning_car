// Browser port of racing_gym (track.py, car.py, env.py).
// Kept deliberately parallel to the Python code so the two can be read side by side;
// tests/test_web_parity.py checks that both produce the same tracks and trajectories.
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.RG = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  const WORLD_SIZE = 1000;
  const SPACING = 4.0;
  const TAU = 2 * Math.PI;

  // ------------------------------------------------------------------ config defaults
  const CAR_DEFAULTS = {
    max_speed: 8.0, max_reverse_speed: 3.0, engine_accel: 0.35, brake_accel: 0.6, rolling_drag: 0.02,
    aero_drag: 0.004, wheel_base: 24.0, max_steer_deg: 32.0, steer_rate_deg: 6.0, grip: 0.9,
    length: 28.0, width: 14.0,
  };
  const CAR_PRESETS = {
    balanced: {},
    grippy: { max_speed: 6.0, grip: 1.6, engine_accel: 0.3 },
    rocket: { max_speed: 11.0, engine_accel: 0.5, grip: 0.7 },
  };
  const REWARD_DEFAULTS = { progress: 1.0, time: -0.05, crash: -20.0, lap: 10.0, center: 0.0, steering: 0.0 };
  const ENV_DEFAULTS = {
    track: "classic", track_width: null, car: "balanced", action_type: "continuous", n_rays: 9,
    ray_length: 250.0, lookahead: [60.0, 120.0, 240.0], random_start: true, max_steps: 2000,
    stall_steps: 200, target_laps: 0,
  };
  const DISCRETE_ACTIONS = [];
  for (const t of [1.0, 0.0, -1.0]) for (const s of [-1.0, 0.0, 1.0]) DISCRETE_ACTIONS.push([s, t]);
  const REWARD_TERMS = ["progress", "time", "crash", "lap", "center", "steering"];

  function carConfig(car) {
    if (typeof car === "object" && car !== null) return Object.assign({}, CAR_DEFAULTS, car);
    if (!(car in CAR_PRESETS)) throw new Error("unknown car preset " + car);
    return Object.assign({}, CAR_DEFAULTS, CAR_PRESETS[car]);
  }

  // ------------------------------------------------------------------ track
  function catmullRomClosed(p, samples) {
    samples = samples || 40;
    const n = p.length, out = [];
    for (let i = 0; i < n; i++) {
      const p0 = p[(i - 1 + n) % n], p1 = p[i], p2 = p[(i + 1) % n], p3 = p[(i + 2) % n];
      for (let k = 0; k < samples; k++) {
        const t = k / samples, t2 = t * t, t3 = t2 * t;
        const pt = [0, 0];
        for (let d = 0; d < 2; d++) {
          pt[d] = 0.5 * (2 * p1[d] + (-p0[d] + p2[d]) * t + (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * t2 +
            (-p0[d] + 3 * p1[d] - 3 * p2[d] + p3[d]) * t3);
        }
        out.push(pt);
      }
    }
    return out;
  }

  function resampleClosed(points, spacing) {
    const closed = points.concat([points[0]]);
    const cum = [0];
    for (let i = 1; i < closed.length; i++) {
      cum.push(cum[i - 1] + Math.hypot(closed[i][0] - closed[i - 1][0], closed[i][1] - closed[i - 1][1]));
    }
    const total = cum[cum.length - 1];
    const n = Math.max(8, roundHalfEven(total / spacing));
    const out = [];
    let j = 0;
    for (let k = 0; k < n; k++) {
      const t = k * (total / n);
      while (j < cum.length - 2 && cum[j + 1] <= t) j++;
      const f = cum[j + 1] > cum[j] ? (t - cum[j]) / (cum[j + 1] - cum[j]) : 0;
      out.push([closed[j][0] + f * (closed[j + 1][0] - closed[j][0]), closed[j][1] + f * (closed[j + 1][1] - closed[j][1])]);
    }
    return out;
  }

  // np.round rounds half to even; Math.round rounds half up. Match numpy exactly.
  function roundHalfEven(x) {
    const r = Math.round(x);
    return (Math.abs(x % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r;
  }

  class Track {
    constructor(name, controlPoints, width, size) {
      this.name = name;
      this.width = width;
      this.size = size || WORLD_SIZE;
      this.controlPoints = controlPoints;
      const c = resampleClosed(catmullRomClosed(controlPoints), SPACING);
      const n = c.length;
      this.centerline = c;
      this.tangents = []; this.normals = []; this.headings = []; this.cumS = [];
      let s = 0;
      for (let i = 0; i < n; i++) {
        const nx = c[(i + 1) % n];
        const dx = nx[0] - c[i][0], dy = nx[1] - c[i][1], L = Math.hypot(dx, dy);
        this.tangents.push([dx / L, dy / L]);
        this.normals.push([-dy / L, dx / L]);
        this.headings.push(Math.atan2(dy, dx));
        this.cumS.push(s);
        s += L;
      }
      this.length = s;
      this.mask = this._rasterize();
    }

    _rasterize() {
      const r = this.width / 2, ri = Math.ceil(r), size = this.size;
      const mask = new Uint8Array(size * size);
      for (const p of this.centerline) {
        const px = roundHalfEven(p[0]), py = roundHalfEven(p[1]);
        for (let oy = -ri; oy <= ri; oy++) {
          const y = py + oy;
          if (y < 0 || y >= size) continue;
          for (let ox = -ri; ox <= ri; ox++) {
            const x = px + ox;
            if (x < 0 || x >= size || ox * ox + oy * oy > r * r) continue;
            mask[y * size + x] = 1;
          }
        }
      }
      return mask;
    }

    isDrivable(x, y) {
      const ix = Math.floor(x), iy = Math.floor(y);
      if (ix < 0 || iy < 0 || ix >= this.size || iy >= this.size) return false;
      return this.mask[iy * this.size + ix] === 1;
    }

    nearestIndex(x, y, hint, window) {
      const n = this.centerline.length;
      window = window || 40;
      let best = -1, bestD = Infinity;
      const lo = hint == null ? 0 : hint - window, hi = hint == null ? n - 1 : hint + window;
      for (let k = lo; k <= hi; k++) {
        const i = ((k % n) + n) % n, c = this.centerline[i];
        const d = (c[0] - x) ** 2 + (c[1] - y) ** 2;
        if (d < bestD) { bestD = d; best = i; }
      }
      return best;
    }

    headingAt(i) { const n = this.headings.length; return this.headings[((i % n) + n) % n]; }
    indexAhead(i, dist) { return (i + roundHalfEven(dist / SPACING)) % this.centerline.length; }

    minClearance() {
      const c = this.centerline, n = c.length, minGap = Math.floor(Math.PI * this.width / SPACING);
      let best = Infinity;
      for (let i = 0; i < n; i += 2) {
        for (let j = 0; j < n; j++) {
          const gap = Math.min(((j - i) % n + n) % n, ((i - j) % n + n) % n);
          if (gap <= minGap) continue;
          const d = Math.hypot(c[j][0] - c[i][0], c[j][1] - c[i][1]);
          if (d < best) best = d;
        }
      }
      return best;
    }

    maxCurvature() {
      const h = this.headings, n = h.length;
      let best = 0;
      for (let i = 0; i < n; i++) {
        let d = h[(i + 1) % n] - h[i];
        if (Math.abs(d) >= Math.PI) d = ((d + Math.PI) % TAU + TAU) % TAU - Math.PI;
        best = Math.max(best, Math.abs(d));
      }
      return best / SPACING;
    }
  }

  const PRESETS = {
    oval: () => new Track("oval", [[200, 500], [200, 300], [230, 230], [300, 200], [500, 200], [700, 200], [770, 230], [800, 300],
      [800, 500], [800, 700], [770, 770], [700, 800], [500, 800], [300, 800], [230, 770], [200, 700]], 110),
    classic: () => new Track("classic", [[150, 520], [160, 300], [260, 170], [430, 160], [520, 280], [600, 360], [760, 300], [860, 400],
      [850, 600], [720, 700], [760, 840], [600, 880], [420, 820], [330, 700], [200, 760]], 90),
    hairpin: () => new Track("hairpin", [[130, 500], [150, 220], [300, 130], [420, 220], [450, 520], [520, 600], [580, 520], [600, 220],
      [720, 130], [860, 220], [870, 600], [820, 840], [620, 880], [420, 860], [240, 860], [140, 720]], 80),
    snake: () => new Track("snake", [[120, 500], [130, 200], [260, 120], [380, 250], [500, 130], [620, 250], [740, 130], [870, 220],
      [880, 520], [870, 800], [740, 880], [620, 750], [500, 870], [380, 750], [260, 880], [130, 800]], 85),
  };

  class Lcg {
    constructor(seed) { this.state = Number((BigInt(Math.trunc(seed)) * 2654435761n + 1013904223n) % 4294967296n); }
    random() { this.state = (1664525 * this.state + 1013904223) % 4294967296; return this.state / 4294967296; }
  }

  function randomTrack(seed, width, maxTries) {
    width = width || 80.0;
    maxTries = maxTries || 50;
    const rng = new Lcg(seed);
    for (let t = 0; t < maxTries; t++) {
      const n = 8 + Math.floor(rng.random() * 6), pts = [];
      for (let k = 0; k < n; k++) {
        const ang = TAU * (k + 0.35 * (rng.random() - 0.5)) / n;
        const rad = 200 + 230 * rng.random();
        pts.push([500 + rad * Math.cos(ang), 500 + rad * Math.sin(ang)]);
      }
      const track = new Track("random:" + seed, pts, width);
      const cornerRadius = 1 / Math.max(track.maxCurvature(), 1e-9);
      if (track.minClearance() > width * 1.15 && cornerRadius > 30) return track;
    }
    throw new Error("could not generate a valid track for seed " + seed);
  }

  const trackCache = new Map();
  function makeTrack(spec, width) {
    spec = String(spec).trim().toLowerCase();
    const key = spec + "|" + (width == null ? "" : width);
    if (trackCache.has(key)) return trackCache.get(key);
    let track;
    if (spec.startsWith("random:")) {
      const seed = Number(spec.slice(7));  // same rule as Python: an integer in [0, 2**31)
      if (!Number.isInteger(seed) || seed < 0 || seed >= 2 ** 31) throw new Error("bad track " + spec);
      track = randomTrack(seed, width || 80.0);
    } else if (spec in PRESETS) {
      track = PRESETS[spec]();
      if (width != null && width !== track.width) track = new Track(track.name, track.controlPoints, width);
    } else throw new Error("unknown track " + spec);
    trackCache.set(key, track);
    return track;
  }

  // ------------------------------------------------------------------ car
  function wrapAngle(a) { return ((a + Math.PI) % TAU + TAU) % TAU - Math.PI; }
  // Python's % is a floored modulo; ((x % m) + m) % m reproduces it for our purposes.

  function stepCar(s, steerCmd, throttle, p) {
    steerCmd = Math.min(1, Math.max(-1, steerCmd));
    throttle = Math.min(1, Math.max(-1, throttle));
    const maxSteer = p.max_steer_deg * Math.PI / 180, rate = p.steer_rate_deg * Math.PI / 180;
    s.steer += Math.min(rate, Math.max(-rate, steerCmd * maxSteer - s.steer));

    let accel;
    if (throttle >= 0) accel = throttle * p.engine_accel;
    else if (s.speed > 0) accel = throttle * p.brake_accel;
    else accel = throttle * p.engine_accel;
    const v = s.speed;
    const drag = Math.abs(v) < 1e-9 ? 0 : Math.sign(v) * p.rolling_drag + p.aero_drag * v * Math.abs(v);
    let nv = v + accel - drag;
    if (accel === 0 && v !== 0 && (nv > 0) !== (v > 0)) nv = 0;
    s.speed = Math.min(p.max_speed, Math.max(-p.max_reverse_speed, nv));

    let yawRate = s.speed / p.wheel_base * Math.tan(s.steer);
    if (Math.abs(s.speed) > 1e-9) {
      const limit = p.grip / Math.abs(s.speed);
      const clamped = Math.min(limit, Math.max(-limit, yawRate));
      s.slip = yawRate - clamped;
      yawRate = clamped;
    } else s.slip = 0;
    s.yaw = wrapAngle(s.yaw + yawRate);
    s.x += s.speed * Math.cos(s.yaw);
    s.y += s.speed * Math.sin(s.yaw);
  }

  function carCorners(s, p) {
    const c = Math.cos(s.yaw), sn = Math.sin(s.yaw), hl = p.length / 2, hw = p.width / 2;
    return [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw], [hl, 0]].map(([lx, ly]) => [s.x + lx * c - ly * sn, s.y + lx * sn + ly * c]);
  }

  // ------------------------------------------------------------------ environment
  function linspace(a, b, n) {
    if (n === 1) return [a];
    const step = (b - a) / (n - 1), out = [];
    for (let i = 0; i < n; i++) out.push(a + i * step);
    out[n - 1] = b;
    return out;
  }

  function observationNames(cfg) {
    const names = [];
    for (let i = 0; i < cfg.n_rays; i++) names.push("lidar_" + i);
    names.push("speed", "steer", "heading_err_sin", "heading_err_cos", "lateral_offset");
    for (const d of cfg.lookahead) names.push("curve_" + Math.trunc(d) + "px");
    return names;
  }

  class RaceEnv {
    constructor(config) {
      const cfg = Object.assign({}, ENV_DEFAULTS, config || {});
      cfg.reward = Object.assign({}, REWARD_DEFAULTS, (config && config.reward) || {});
      this.config = cfg;
      this.carParams = carConfig(cfg.car);
      this.rayOffsets = cfg.n_rays > 1 ? linspace(-Math.PI / 2, Math.PI / 2, cfg.n_rays) : [0];
      this.raySteps = [];
      for (let k = 0; 4 + 4 * k < cfg.ray_length + 1e-9; k++) this.raySteps.push(4 + 4 * k);
      this.obsNames = observationNames(cfg);
      this.nObs = this.obsNames.length;
      this.track = cfg.track === "random" ? null : makeTrack(cfg.track, cfg.track_width);
    }

    _lidar() {
      const c = this.car, out = [];
      for (const off of this.rayOffsets) {
        const a = c.yaw + off, ca = Math.cos(a), sa = Math.sin(a);
        let dist = this.config.ray_length;
        for (const d of this.raySteps) {
          if (!this.track.isDrivable(c.x + ca * d, c.y + sa * d)) { dist = d; break; }
        }
        out.push(dist);
      }
      return out;
    }

    _project() {
      const t = this.track;
      this.idx = t.nearestIndex(this.car.x, this.car.y, this.idx);
      const cl = t.centerline[this.idx], tg = t.tangents[this.idx], nm = t.normals[this.idx];
      const dx = this.car.x - cl[0], dy = this.car.y - cl[1];
      return [t.cumS[this.idx] + dx * tg[0] + dy * tg[1], dx * nm[0] + dy * nm[1]];
    }

    _crashed() { return carCorners(this.car, this.carParams).some(([x, y]) => !this.track.isDrivable(x, y)); }

    observe() {
      const cfg = this.config, p = this.carParams, t = this.track;
      const he = wrapAngle(this.car.yaw - t.headingAt(this.idx));
      const obs = this.lidar.map((d) => d / cfg.ray_length);
      obs.push(this.car.speed / p.max_speed, this.car.steer / (p.max_steer_deg * Math.PI / 180),
        Math.sin(he), Math.cos(he), this.lateral / (t.width / 2));
      for (const d of cfg.lookahead) obs.push(wrapAngle(t.headingAt(t.indexAhead(this.idx, d)) - t.headingAt(this.idx)) / Math.PI);
      const nr = cfg.n_rays;
      return obs.map((v, i) => {
        const lo = i < nr ? 0 : (i === nr + 4 ? -2 : -1), hi = i === nr + 4 ? 2 : 1;
        return Math.min(hi, Math.max(lo, Math.fround(v)));
      });
    }

    reset(opts) {
      opts = opts || {};
      if (opts.track) this.track = makeTrack(opts.track, this.config.track_width);
      else if (!this.track) this.track = randomTrack(1000 + Math.floor(Math.random() * 1e6), this.config.track_width || 80);
      const n = this.track.centerline.length;
      let start = opts.startIndex != null ? opts.startIndex : (this.config.random_start ? Math.floor(Math.random() * n) : 0);
      start = ((start % n) + n) % n;
      const c = this.track.centerline[start];
      this.car = { x: c[0], y: c[1], yaw: this.track.headingAt(start), speed: 0, steer: 0, slip: 0 };
      this.idx = start;
      [this.s, this.lateral] = this._project();
      this.lidar = this._lidar();
      this.steps = 0; this.distance = 0; this.bestDistance = 0; this.stepsSinceBest = 0; this.laps = 0;
      this.prevSteerCmd = 0; this.crashed = false; this.speedSum = 0; this.done = false;
      this.episodeTerms = {}; this.lastTerms = {};
      for (const k of REWARD_TERMS) { this.episodeTerms[k] = 0; this.lastTerms[k] = 0; }
      this.lastAction = [0, 0];
      return this.observe();
    }

    step(action) {
      let steerCmd, throttle;
      if (typeof action === "number") [steerCmd, throttle] = DISCRETE_ACTIONS[action];
      else { steerCmd = Math.min(1, Math.max(-1, action[0])); throttle = Math.min(1, Math.max(-1, action[1])); }
      this.lastAction = [steerCmd, throttle];
      stepCar(this.car, steerCmd, throttle, this.carParams);
      this.steps += 1;
      this.speedSum += this.car.speed;

      const oldS = this.s;
      [this.s, this.lateral] = this._project();
      const L = this.track.length;
      const ds = ((this.s - oldS + L / 2) % L + L) % L - L / 2;
      this.distance += ds;
      const newLaps = Math.max(0, Math.floor(this.distance / L) - this.laps);
      this.laps += newLaps;
      if (this.distance > this.bestDistance + 1e-6) { this.bestDistance = this.distance; this.stepsSinceBest = 0; }
      else this.stepsSinceBest += 1;

      this.crashed = this._crashed();
      const w = this.config.reward;
      const terms = {
        progress: w.progress * ds / 10,
        time: w.time,
        crash: this.crashed ? w.crash : 0,
        lap: w.lap * newLaps,
        center: w.center * (this.lateral / (this.track.width / 2)) ** 2,
        steering: w.steering * Math.abs(steerCmd - this.prevSteerCmd),
      };
      this.prevSteerCmd = steerCmd;
      let reward = 0;
      for (const k of REWARD_TERMS) { reward += terms[k]; this.episodeTerms[k] += terms[k]; }
      this.lastTerms = terms;
      const cfg = this.config;
      const terminated = this.crashed || (cfg.target_laps > 0 && this.laps >= cfg.target_laps);
      const truncated = !terminated && (this.steps >= cfg.max_steps || this.stepsSinceBest >= cfg.stall_steps);
      this.lidar = this._lidar();
      this.done = terminated || truncated;
      return { obs: this.observe(), reward, terminated, truncated };
    }

    episodeReturn() { return REWARD_TERMS.reduce((a, k) => a + this.episodeTerms[k], 0); }
  }

  // ------------------------------------------------------------------ policies
  function mlpForward(model, obs) {
    let x = obs;
    for (const layer of model.layers) {
      const y = new Array(layer.b.length);
      for (let i = 0; i < layer.b.length; i++) {
        let acc = layer.b[i];
        const row = layer.w[i];
        for (let j = 0; j < row.length; j++) acc += row[j] * x[j];
        y[i] = layer.act === "tanh" ? Math.tanh(acc) : layer.act === "relu" ? Math.max(0, acc) : acc;
      }
      x = y;
    }
    if (model.output === "clip") return x.map((v) => Math.min(1, Math.max(-1, v)));
    if (model.output === "tanh") return x.map(Math.tanh);
    if (model.output === "argmax") return x.indexOf(Math.max(...x));
    return x;
  }

  function heuristicAction(obs, nRays) {
    const hs = obs[nRays + 2], lat = obs[nRays + 4], curve = obs[nRays + 5];
    const steer = Math.min(1, Math.max(-1, -2.0 * hs - 0.8 * lat + 1.5 * curve));
    const throttle = obs[nRays] < 0.75 || Math.abs(curve) < 0.15 ? 1.0 : 0.2;
    return [steer, throttle];
  }

  return {
    WORLD_SIZE, SPACING, CAR_DEFAULTS, CAR_PRESETS, REWARD_DEFAULTS, ENV_DEFAULTS, DISCRETE_ACTIONS, REWARD_TERMS,
    Track, PRESETS, randomTrack, makeTrack, stepCar, carCorners, wrapAngle, RaceEnv, observationNames,
    mlpForward, heuristicAction, carConfig,
  };
});
