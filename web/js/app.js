// Racing Gym playground UI. Simulation lives in sim.js; exported agents in models/*.js.
(function () {
  "use strict";
  const RG = window.RG;
  const MODELS = window.RG_MODELS || {};
  const $ = (id) => document.getElementById(id);
  const COLORS = ["#ffb347", "#6ec6ff", "#c792ea", "#5fd38d", "#ff6b5e", "#f7e463", "#ff8ad8", "#9be7d8"];
  const STEPS_PER_SECOND = 30;
  const TOUCH = window.matchMedia("(pointer: coarse)").matches;  // phones and tablets

  const state = {
    trackSpec: "classic",
    racers: [],
    selected: 0,
    paused: false,
    speed: 1,
    keys: {},
    carOverrides: {},
    reward: Object.assign({}, RG.REWARD_DEFAULTS),
    nextColor: 0,
    nextId: 1,
  };

  // ------------------------------------------------------------------ racers
  function driverOptions() {
    const opts = [];
    for (const name of Object.keys(MODELS)) opts.push({ value: "model:" + name, label: `${name} (${MODELS[name].algo.toUpperCase()})` });
    opts.push({ value: "human", label: "You (arrow keys / WASD)" });
    opts.push({ value: "heuristic", label: "Hand-written controller (no learning)" });
    opts.push({ value: "random", label: "Random actions" });
    return opts;
  }

  function makeRacer(kind, stageIdx) {
    let envCfg = { track: state.trackSpec, random_start: false, max_steps: 4000, stall_steps: 200 };
    let label = kind, model = null;
    if (kind.startsWith("model:")) {
      model = MODELS[kind.slice(6)];
      envCfg = Object.assign({}, model.env, envCfg, { reward: Object.assign({}, state.reward) });
      if (stageIdx == null) stageIdx = model.stages.length - 1;
      label = model.name;
    } else {
      envCfg.reward = Object.assign({}, state.reward);
      label = { human: "You", heuristic: "Hand-written", random: "Random" }[kind];
    }
    const env = new RG.RaceEnv(envCfg);
    applyCarOverrides(env);
    const racer = {
      id: state.nextId++, kind, model, stageIdx: stageIdx || 0, env, label,
      color: COLORS[state.nextColor++ % COLORS.length], obs: null, trail: [], history: [], laps: 0, bestLaps: 0,
    };
    racer.obs = env.reset({ track: state.trackSpec, startIndex: 0 });
    return racer;
  }

  function racerName(r) {
    if (!r.model) return r.label;
    const st = r.model.stages[r.stageIdx];
    return r.model.stages.length > 1 ? `${r.label} · ${st.label}` : r.label;
  }

  function act(r) {
    const nRays = r.env.config.n_rays;
    if (r.kind === "human") {
      const k = state.keys;
      const steer = (k.ArrowRight || k.d ? 1 : 0) - (k.ArrowLeft || k.a ? 1 : 0);
      const throttle = (k.ArrowUp || k.w ? 1 : 0) - (k.ArrowDown || k.s ? 1 : 0);
      return [steer, throttle];
    }
    if (r.kind === "random") return [Math.random() * 2 - 1, Math.random() * 2 - 1];
    if (r.kind === "heuristic") return RG.heuristicAction(r.obs, nRays);
    const stage = r.model.stages[r.stageIdx];
    return RG.mlpForward({ layers: stage.layers, output: r.model.output }, r.obs);
  }

  function stepRacer(r) {
    const res = r.env.step(act(r));
    r.obs = res.obs;
    r.trail.push([r.env.car.x, r.env.car.y]);
    if (r.trail.length > 90) r.trail.shift();
    if (r.env.done) {
      r.history.push({ ret: r.env.episodeReturn(), crashed: r.env.crashed, laps: r.env.laps });
      if (r.history.length > 40) r.history.shift();
      r.bestLaps = Math.max(r.bestLaps, r.env.laps);
      if (r.model && $("stage-auto").checked && r.model.stages.length > 1) {
        r.stageIdx = (r.stageIdx + 1) % r.model.stages.length;
        if (state.racers[state.selected] === r) syncStageBox();
        renderRacerList();
      }
      r.obs = r.env.reset({ startIndex: 0 });
      r.trail = [];
    }
  }

  function restartAll() {
    for (const r of state.racers) {
      r.obs = r.env.reset({ track: state.trackSpec, startIndex: 0 });
      r.trail = [];
    }
  }

  // ------------------------------------------------------------------ track drawing
  let trackCanvas = null, trackFor = null;
  function trackImage(track) {
    if (trackFor === track) return trackCanvas;
    const size = track.size, c = document.createElement("canvas");
    c.width = c.height = size;
    const ctx = c.getContext("2d"), img = ctx.createImageData(size, size), m = track.mask, d = img.data;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const i = y * size + x, o = i * 4;
        let r = 34, g = 52, b = 38;
        if (m[i]) {
          const edge = x < 2 || y < 2 || x >= size - 2 || y >= size - 2 ||
            !m[i - 2] || !m[i + 2] || !m[i - 2 * size] || !m[i + 2 * size];
          if (edge) { r = g = b = 225; } else { r = 52; g = 54; b = 62; }
        } else if (((x >> 3) + (y >> 3)) % 2 === 0) { r = 37; g = 56; b = 41; }
        d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    // start / finish line
    const c0 = track.centerline[0], n0 = track.normals[0], half = track.width / 2;
    for (let k = -half; k < half; k += 6) {
      ctx.fillStyle = Math.floor((k + half) / 6) % 2 ? "#141414" : "#f0f0f0";
      ctx.fillRect(c0[0] + n0[0] * k - 3, c0[1] + n0[1] * k - 3, 6, 6);
    }
    trackCanvas = c; trackFor = track;
    return c;
  }

  function draw() {
    const cv = $("view"), ctx = cv.getContext("2d");
    const track = state.racers.length ? state.racers[0].env.track : RG.makeTrack(state.trackSpec);
    ctx.drawImage(trackImage(track), 0, 0);
    if ($("show-center").checked) {
      ctx.setLineDash([10, 12]); ctx.strokeStyle = "rgba(255,255,255,0.35)"; ctx.lineWidth = 2;
      ctx.beginPath();
      track.centerline.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
      ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);
    }
    state.racers.forEach((r, idx) => {
      const env = r.env, car = env.car, p = env.carParams;
      if ($("show-trails").checked && r.trail.length > 1) {
        ctx.strokeStyle = r.color; ctx.lineWidth = 3;
        for (let i = 1; i < r.trail.length; i++) {
          ctx.globalAlpha = (i / r.trail.length) * 0.6;
          ctx.beginPath(); ctx.moveTo(...r.trail[i - 1]); ctx.lineTo(...r.trail[i]); ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
      if ($("show-lidar").checked && idx === state.selected) {
        env.lidar.forEach((dist, i) => {
          const a = car.yaw + env.rayOffsets[i], f = dist / env.config.ray_length;
          const col = `rgb(${Math.round(255 * (1 - f))},${Math.round(80 + 175 * f)},90)`;
          ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1.5;
          const ex = car.x + dist * Math.cos(a), ey = car.y + dist * Math.sin(a);
          ctx.beginPath(); ctx.moveTo(car.x, car.y); ctx.lineTo(ex, ey); ctx.stroke();
          ctx.beginPath(); ctx.arc(ex, ey, 4, 0, 2 * Math.PI); ctx.fill();
        });
      }
      ctx.save();
      ctx.translate(car.x, car.y); ctx.rotate(car.yaw);
      ctx.fillStyle = env.crashed ? "#ff6b5e" : r.color;
      ctx.strokeStyle = idx === state.selected ? "#ffffff" : "rgba(0,0,0,0.6)";
      ctx.lineWidth = idx === state.selected ? 3 : 1.5;
      ctx.beginPath(); ctx.roundRect(-p.length / 2, -p.width / 2, p.length, p.width, 4); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "rgba(20,25,35,0.85)"; ctx.fillRect(p.length * 0.05, -p.width / 2 + 3, p.length * 0.22, p.width - 6);
      ctx.restore();
      ctx.font = "600 15px system-ui, sans-serif"; ctx.textAlign = "center";
      ctx.lineWidth = 4; ctx.strokeStyle = "rgba(0,0,0,0.7)"; ctx.fillStyle = "#fff";
      const tag = r.label + (r.model && r.model.stages.length > 1 && r.stageIdx < r.model.stages.length - 1 ? " ·" + r.model.stages[r.stageIdx].label.replace("after ", "") : "");
      ctx.strokeText(tag, car.x, car.y - 22); ctx.fillText(tag, car.x, car.y - 22);
    });
  }

  // ------------------------------------------------------------------ side panels
  function renderRacerList() {
    // On-screen arrows: touch devices only, and only while someone is driving.
    $("touch-pad").hidden = !(TOUCH && state.racers.some((r) => r.kind === "human"));
    const ul = $("racer-list");
    ul.innerHTML = "";
    state.racers.forEach((r, i) => {
      const li = document.createElement("li");
      li.className = i === state.selected ? "selected" : "";
      const last = r.history.length ? r.history[r.history.length - 1] : null;
      li.innerHTML = `<span class="dot" style="background:${r.color}"></span><span>${racerName(r)}</span>` +
        `<span class="stat">lap ${r.env.laps}${r.bestLaps ? ` · best ${r.bestLaps}` : ""}${last && last.crashed ? " · last: crash" : ""}</span>` +
        `<button class="x" title="Remove" aria-label="Remove ${racerName(r)}">✕</button>`;
      li.addEventListener("click", (e) => {
        if (e.target.classList.contains("x")) { removeRacer(i); return; }
        select(i);
      });
      ul.appendChild(li);
    });
  }

  function removeRacer(i) {
    state.racers.splice(i, 1);
    state.selected = Math.max(0, Math.min(state.selected, state.racers.length - 1));
    renderRacerList(); buildInspector(); syncStageBox();
  }

  function select(i) { state.selected = i; renderRacerList(); buildInspector(); syncStageBox(); }

  function syncStageBox() {
    const r = state.racers[state.selected];
    const box = $("stage-box");
    if (!r || !r.model || r.model.stages.length < 2) { box.hidden = true; return; }
    box.hidden = false;
    $("stage-racer").textContent = r.label;
    const s = $("stage");
    s.max = r.model.stages.length - 1; s.value = r.stageIdx;
    const st = r.model.stages[r.stageIdx];
    $("stage-label").textContent = st.label;
    $("stage-return").textContent = st.eval_return == null ? "" : `eval return ${st.eval_return.toFixed(0)}`;
  }

  function barRows(container, rows) {
    container.innerHTML = "";
    return rows.map((name) => {
      const n = document.createElement("span"); n.className = "name"; n.textContent = name;
      const b = document.createElement("span"); b.className = "bar"; const fill = document.createElement("i"); b.appendChild(fill);
      const v = document.createElement("span"); v.className = "val";
      container.append(n, b, v);
      return { fill, v };
    });
  }
  function setBar(row, value, scale, text) {
    const f = Math.max(-1, Math.min(1, value / scale)) * 50;
    row.fill.style.left = (f >= 0 ? 50 : 50 + f) + "%";
    row.fill.style.width = Math.abs(f) + "%";
    row.fill.style.background = value >= 0 ? "var(--pos)" : "var(--neg)";
    row.v.textContent = text;
  }

  let ui = null;
  function buildInspector() {
    const r = state.racers[state.selected];
    $("insp-name").textContent = r ? racerName(r) : "the agent";
    if (!r) { ui = null; return; }
    ui = {
      action: barRows($("insp-action"), ["steer", "throttle"]),
      reward: barRows($("insp-reward"), RG.REWARD_TERMS.concat(["return"])),
      obs: barRows($("insp-obs"), r.env.obsNames),
    };
  }

  function updateInspector() {
    const r = state.racers[state.selected];
    if (!r || !ui) return;
    const env = r.env;
    setBar(ui.action[0], env.lastAction[0], 1, (env.lastAction[0] < 0 ? "left " : env.lastAction[0] > 0 ? "right " : "") + env.lastAction[0].toFixed(2));
    setBar(ui.action[1], env.lastAction[1], 1, (env.lastAction[1] < 0 ? "brake " : "gas ") + env.lastAction[1].toFixed(2));
    RG.REWARD_TERMS.forEach((k, i) => setBar(ui.reward[i], env.lastTerms[k], 1, `${fmt(env.lastTerms[k])} · ${fmt(env.episodeTerms[k], 1)}`));
    const ret = env.episodeReturn();
    setBar(ui.reward[RG.REWARD_TERMS.length], ret, 500, fmt(ret, 1));
    r.obs.forEach((v, i) => setBar(ui.obs[i], v, 1, v.toFixed(2)));
    drawHistory(r);
  }
  const fmt = (v, d) => (v >= 0 ? "+" : "") + v.toFixed(d == null ? 2 : d);

  function drawHistory(r) {
    const cv = $("history"), ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    const hist = r.history.slice(-30), cur = r.env.episodeReturn();
    const vals = hist.map((h) => h.ret).concat([cur]);
    const max = Math.max(10, ...vals), min = Math.min(0, ...vals);
    const bw = W / 31, y0 = H - ((0 - min) / (max - min)) * (H - 4) - 2;
    vals.forEach((v, i) => {
      const crashed = i < hist.length ? hist[i].crashed : false;
      ctx.fillStyle = i === vals.length - 1 ? "rgba(255,179,71,0.8)" : crashed ? "#ff6b5e" : "#5fd38d";
      const y = H - ((v - min) / (max - min)) * (H - 4) - 2;
      ctx.fillRect(i * bw + 1, Math.min(y, y0), bw - 2, Math.max(1, Math.abs(y0 - y)));
    });
  }

  function sliders(container, specs, onChange) {
    container.innerHTML = "";
    for (const s of specs) {
      const l = document.createElement("label"); l.textContent = s.label; l.htmlFor = "sl-" + s.key;
      const input = document.createElement("input");
      Object.assign(input, { type: "range", id: "sl-" + s.key, min: s.min, max: s.max, step: s.step, value: s.value });
      const out = document.createElement("output"); out.textContent = s.value;
      input.addEventListener("input", () => { out.textContent = input.value; onChange(s.key, parseFloat(input.value)); });
      container.append(l, input, out);
    }
  }

  function applyCarOverrides(env) { Object.assign(env.carParams, state.carOverrides); }

  // ------------------------------------------------------------------ setup
  function init() {
    // Deep links: ?track=hairpin or ?track=random:42, ?speed=2, ?warmup=600 (simulate N steps before showing)
    const params = new URLSearchParams(location.search);
    if (params.get("track")) {
      try { RG.makeTrack(params.get("track")); state.trackSpec = params.get("track").toLowerCase(); } catch (e) { /* keep default */ }
    }
    const trackSel = $("track");
    for (const name of Object.keys(RG.PRESETS)) trackSel.add(new Option(name, name));
    trackSel.add(new Option("random (seed →)", "random"));
    if (state.trackSpec.startsWith("random:")) { trackSel.value = "random"; $("seed").value = state.trackSpec.slice(7); }
    else trackSel.value = state.trackSpec;
    const setTrack = () => {
      state.trackSpec = trackSel.value === "random" ? "random:" + (parseInt($("seed").value, 10) || 0) : trackSel.value;
      setBanner("Building track…");
      setTimeout(() => { restartAll(); setBanner(""); }, 10);
    };
    trackSel.addEventListener("change", setTrack);
    $("seed").addEventListener("change", () => { trackSel.value = "random"; setTrack(); });
    $("btn-random").addEventListener("click", () => {
      $("seed").value = Math.floor(Math.random() * 1000); trackSel.value = "random"; setTrack();
    });

    const kindSel = $("add-kind");
    for (const o of driverOptions()) kindSel.add(new Option(o.label, o.value));

    // Bring your own agent: a file written by `racing-gym export` (JS wrapper) or its plain JSON.
    async function loadAgentFile(file) {
      try {
        const text = await file.text();
        const json = text.trim().startsWith("{") ? text : text.slice(text.indexOf("= {") + 2, text.lastIndexOf("; })"));
        const model = JSON.parse(json);
        if (!Array.isArray(model.stages) || !model.stages.length || !model.env) throw new Error("not a racing-gym export");
        let name = model.name || file.name.replace(/\.(js|json)$/, "");
        while (MODELS[name]) name += "'";
        model.name = name;
        MODELS[name] = model;
        kindSel.add(new Option(`${name} (${String(model.algo).toUpperCase()}, yours)`, "model:" + name));
        state.racers.push(makeRacer("model:" + name));
        select(state.racers.length - 1);
        setBanner(`Loaded ${name}: it is racing now. Drag its training stage slider!`);
        setTimeout(() => setBanner(""), 4000);
      } catch (err) {
        console.warn("agent upload failed:", err);
        setBanner("That file is not a Racing Gym agent. Make one with `racing-gym export` or the Colab notebook.");
        setTimeout(() => setBanner(""), 6000);
      }
    }
    $("upload").addEventListener("change", (e) => { if (e.target.files[0]) loadAgentFile(e.target.files[0]); e.target.value = ""; });
    const drop = document.querySelector(".upload");
    for (const target of [drop, $("view")]) {
      target.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("drag"); });
      target.addEventListener("dragleave", () => drop.classList.remove("drag"));
      target.addEventListener("drop", (e) => {
        e.preventDefault(); drop.classList.remove("drag");
        if (e.dataTransfer.files[0]) loadAgentFile(e.dataTransfer.files[0]);
      });
    }
    $("btn-add").addEventListener("click", () => {
      state.racers.push(makeRacer(kindSel.value));
      select(state.racers.length - 1);
    });

    $("stage").addEventListener("input", (e) => {
      const r = state.racers[state.selected];
      r.stageIdx = parseInt(e.target.value, 10);
      r.obs = r.env.reset({ startIndex: 0 }); r.trail = []; r.history = []; r.bestLaps = 0;
      syncStageBox(); renderRacerList(); buildInspector();
    });

    $("btn-play").addEventListener("click", togglePause);
    $("btn-reset").addEventListener("click", restartAll);
    $("sim-speed").addEventListener("change", (e) => { state.speed = parseFloat(e.target.value); });

    const car = RG.CAR_DEFAULTS;
    sliders($("car-sliders"), [
      { key: "max_speed", label: "top speed", min: 3, max: 14, step: 0.5, value: car.max_speed },
      { key: "engine_accel", label: "engine", min: 0.1, max: 0.8, step: 0.05, value: car.engine_accel },
      { key: "grip", label: "tyre grip", min: 0.3, max: 2.5, step: 0.05, value: car.grip },
      { key: "max_steer_deg", label: "steering lock°", min: 10, max: 45, step: 1, value: car.max_steer_deg },
    ], (k, v) => { state.carOverrides[k] = v; state.racers.forEach((r) => applyCarOverrides(r.env)); });
    const rw = RG.REWARD_DEFAULTS;
    sliders($("reward-sliders"), [
      { key: "progress", label: "progress", min: 0, max: 3, step: 0.1, value: rw.progress },
      { key: "time", label: "time", min: -0.5, max: 0, step: 0.01, value: rw.time },
      { key: "crash", label: "crash", min: -100, max: 0, step: 1, value: rw.crash },
      { key: "lap", label: "lap", min: 0, max: 100, step: 1, value: rw.lap },
      { key: "center", label: "stay centred", min: -2, max: 0, step: 0.05, value: rw.center },
      { key: "steering", label: "smooth steering", min: -2, max: 0, step: 0.05, value: rw.steering },
    ], (k, v) => { state.reward[k] = v; state.racers.forEach((r) => { r.env.config.reward[k] = v; }); });

    window.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
      if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", " "].includes(e.key) && state.racers.some((r) => r.kind === "human")) e.preventDefault();
      if (e.key === " ") { togglePause(); e.preventDefault(); return; }
      if (k === "r" && !state.racers.some((r) => r.kind === "human")) { restartAll(); return; }
      state.keys[k] = true;
    });
    window.addEventListener("keyup", (e) => { state.keys[e.key.length === 1 ? e.key.toLowerCase() : e.key] = false; });
    // Touch arrows press the same keys as the keyboard.
    document.querySelectorAll("#touch-pad button").forEach((b) => {
      const set = (down) => (e) => { e.preventDefault(); state.keys[b.dataset.key] = down; };
      b.addEventListener("pointerdown", set(true));
      ["pointerup", "pointercancel", "pointerleave"].forEach((t) => b.addEventListener(t, set(false)));
      b.addEventListener("contextmenu", (e) => e.preventDefault());
    });
    $("view").addEventListener("click", (e) => {
      const rect = e.target.getBoundingClientRect();
      const scale = 1000 / Math.min(rect.width, rect.height);
      const x = (e.clientX - rect.left - (rect.width - Math.min(rect.width, rect.height)) / 2) * scale;
      const y = (e.clientY - rect.top - (rect.height - Math.min(rect.width, rect.height)) / 2) * scale;
      let best = -1, bestD = 40 * 40;
      state.racers.forEach((r, i) => { const d = (r.env.car.x - x) ** 2 + (r.env.car.y - y) ** 2; if (d < bestD) { bestD = d; best = i; } });
      if (best >= 0) select(best);
    });

    // Default line-up: a trained agent racing an early (still crashing) version of itself.
    const names = Object.keys(MODELS).sort((a, b) => (b === "PPO") - (a === "PPO"));  // PPO first if present
    if (names.length) {
      const m = MODELS[names[0]];
      state.racers.push(makeRacer("model:" + names[0]));
      if (m.stages.length > 1) state.racers.push(makeRacer("model:" + names[0], Math.min(1, m.stages.length - 2)));
    } else {
      state.racers.push(makeRacer("heuristic"));
      state.racers.push(makeRacer("random"));
      setBanner("No exported agents found in web/models. Train one and run `racing-gym export`.");
    }
    select(0);
    const speed = parseFloat(params.get("speed"));
    if (speed > 0) { state.speed = speed; $("sim-speed").value = String(speed); }
    const warmup = Math.min(20000, parseInt(params.get("warmup"), 10) || 0);
    for (let i = 0; i < warmup; i++) state.racers.forEach(stepRacer);
    requestAnimationFrame(loop);
  }

  function togglePause() {
    state.paused = !state.paused;
    $("btn-play").textContent = state.paused ? "▶ Play" : "⏸ Pause";
  }
  function setBanner(text) { const b = $("banner"); b.textContent = text; b.hidden = !text; }

  let last = performance.now(), acc = 0, frame = 0;
  function loop(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (!state.paused) {
      acc += dt * STEPS_PER_SECOND * state.speed;
      let n = Math.floor(acc);
      acc -= n;
      n = Math.min(n, 60);
      for (let i = 0; i < n; i++) state.racers.forEach(stepRacer);
    }
    draw();
    if (frame++ % 3 === 0) updateInspector();
    if (frame % 30 === 0) renderRacerList();
    requestAnimationFrame(loop);
  }

  init();
  // For tinkering in the browser console: RG_APP.racers[0].env.car, RG_APP.racers[0].obs, ...
  window.RG_APP = state;
})();
