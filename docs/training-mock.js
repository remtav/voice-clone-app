/* Static demo only: an in-browser stand-in for the trainer service.
 *
 * training.js is a byte-for-byte copy of the real app's script; this file
 * answers its /api/training and /api/models calls by intercepting fetch(), and
 * simulates a run (queue → GPU hand-off → training → validation → publish).
 * A real run on an RTX 3090 takes 3–6 h; here it takes about a minute, while
 * the step counts and "time left" show the real-scale numbers. */
(() => {
  "use strict";

  const realFetch = window.fetch.bind(window);
  const now = () => Date.now() / 1000;
  const rid = () => Math.random().toString(16).slice(2, 14).padEnd(12, "0");
  const GPU = { name: "NVIDIA GeForce RTX 3090", total_gb: 24.0 };
  const DEFAULTS = { // same as app/training.py
    fr_ca: { epochs: 4, lora_rank: 16, learning_rate: 2e-5, grad_accum: 8 },
    personal: { epochs: 2, lora_rank: 16, learning_rate: 1e-5, grad_accum: 8 },
  };
  const CLIPS = { fr_ca: 9952, personal: 4 * 312 + 2986 }; // corpus; your clips ×3 + 30 % of it
  const SECONDS_PER_STEP = 0.33;   // what a 3090 does, used for the displayed ETA
  const TRAIN_DEMO_SECONDS = 45;   // how long the demo's "Training" stage really lasts
  const ACTIVE = new Set(["queued", "waiting_gpu", "running", "cancelling"]);

  const lossAt = (p, seed = 0) => 1.05 + 1.85 * Math.exp(-3.2 * p) + 0.04 * Math.sin(37 * p + seed);
  const seededHistory = (total) => Array.from({ length: 60 }, (_, i) => [Math.round(((i + 1) / 60) * total), lossAt((i + 1) / 60, 1)]);

  const seeded = (() => {
    const perEpoch = Math.round(CLIPS.fr_ca * 0.95);
    const total = 4 * perEpoch;
    const finished = now() - 2 * 86400;
    return {
      id: rid(), name: "Quebec accent", recipe: "fr_ca", params: { ...DEFAULTS.fr_ca, output_name: "t3_quebec_accent_4f2a91" },
      status: "done", stage: null, message: "Model ready: t3_quebec_accent_4f2a91.safetensors",
      step: total, total_steps: total, epoch: 4, epochs: 4, loss: lossAt(1, 1), eta_seconds: 0,
      output_model: "t3_quebec_accent_4f2a91.safetensors",
      created_at: finished - 4.7 * 3600, started_at: finished - 4.6 * 3600, finished_at: finished,
      history: seededHistory(total),
      log: [
        "=== Downloading the Quebec corpus (1.8 GB) ===", "Fetching 9993 files: 100%|██████████| 9993/9993",
        "=== Preparing the Quebec corpus ===", `train: ${CLIPS.fr_ca} clips, 12.0 h, holdout: 40 clips`,
        "=== Setting up the training toolkit ===", "=== Training ===",
        `Train samples: ${perEpoch}, Validation samples: ${CLIPS.fr_ca - perEpoch}`,
        "Injected 210 LoRA layers",
        ...[1, 2, 3, 4].map((e) => `Saved checkpoint to /data/finetune/runs/…/checkpoint_epoch${e - 1}_step${e * perEpoch}.pt`),
        "=== Exporting the checkpoint ===", "=== Validating the checkpoint ===",
        "t3_quebec_accent_4f2a91.safetensors: 292 tensors, 2.14 GB", "changed vs v3: 210 tensor(s)",
        "strict load into T3(T3Config.multilingual()): OK", "smoke: smoke_00.wav (2.1s)", "smoke: smoke_01.wav (2.9s)",
        "smoke: smoke_02.wav (3.6s)", "OK", "=== Publishing to data/models ===",
      ],
    };
  })();

  const state = {
    active: "v3",
    models: [{ name: seeded.output_model, size_bytes: 2143989752, created_at: seeded.finished_at }],
    runs: [seeded],
  };

  // ---------------------------------------------------------------- simulation
  function stagesFor(run) {
    const pre = run.recipe === "fr_ca" ? [] : [["Mixing your clips with the Quebec corpus", 2]];
    return [...pre, ["Setting up the training toolkit", 2], ["Training", TRAIN_DEMO_SECONDS],
      ["Exporting the checkpoint", 3], ["Validating the checkpoint", 5], ["Publishing to data/models", 1]];
  }

  function advance(run) {
    const t = now();
    if (run.status === "queued" && t - run.created_at > 1.5) {
      run.status = "waiting_gpu";
      run.message = "Waiting for the GPU: the app still holds its model (it finishes the current generation first)";
      run.log.push("The app is releasing its model…");
      run.waitedAt = t;
    } else if (run.status === "waiting_gpu" && t - run.waitedAt > 3) {
      run.status = "running";
      run.message = null;
      run.started_at = t;
      run.stageIndex = -1;
      nextStage(run);
    } else if (run.status === "cancelling" && t - run.cancelledAt > 1.5) {
      finish(run, "cancelled", "Cancelled");
      run.log.push("Run cancelled: training processes stopped.");
    } else if (run.status === "running") {
      const [label, seconds] = stagesFor(run)[run.stageIndex];
      const elapsed = t - run.stageStart;
      if (label === "Training") train(run, Math.min(1, elapsed / seconds));
      if (elapsed >= seconds) nextStage(run);
    }
  }

  function nextStage(run) {
    run.stageIndex += 1;
    const stages = stagesFor(run);
    if (run.stageIndex >= stages.length) {
      const output = `${run.params.output_name}.safetensors`;
      state.models.unshift({ name: output, size_bytes: 2143989752, created_at: now() });
      run.output_model = output;
      run.log.push("smoke: smoke_00.wav (2.0s)", "smoke: smoke_01.wav (2.6s)", "smoke: smoke_02.wav (3.4s)", "OK");
      finish(run, "done", `Model ready: ${output}`);
      return;
    }
    const [label] = stages[run.stageIndex];
    run.stage = label;
    run.stageStart = now();
    run.log.push(`=== ${label} ===`);
    if (label === "Training") {
      const perEpoch = Math.round(CLIPS[run.recipe] * 0.95);
      run.epochs = run.params.epochs;
      run.total_steps = run.epochs * perEpoch;
      run.message = "demo: hours of training compressed into a minute";
      run.log.push(`Train samples: ${perEpoch}, Validation samples: ${CLIPS[run.recipe] - perEpoch}`, "Injected 210 LoRA layers");
    } else {
      run.message = null;
    }
    if (label === "Validating the checkpoint") {
      run.log.push(`${run.params.output_name}.safetensors: 292 tensors, 2.14 GB`, "changed vs v3: 210 tensor(s)",
        "strict load into T3(T3Config.multilingual()): OK");
    }
  }

  function train(run, progress) {
    const perEpoch = run.total_steps / run.epochs;
    run.step = Math.round(progress * run.total_steps);
    run.epoch = Math.min(run.epochs, Math.floor(run.step / perEpoch) + 1);
    run.loss = lossAt(progress, run.seed);
    run.eta_seconds = Math.round((run.total_steps - run.step) * SECONDS_PER_STEP);
    const last = run.history[run.history.length - 1];
    if (!last || run.step - last[0] >= run.total_steps / 60) {
      run.history.push([run.step, run.loss]);
      const nInEpoch = run.step - (run.epoch - 1) * perEpoch;
      const pct = Math.round((100 * nInEpoch) / perEpoch);
      run.log.push(`Epoch ${run.epoch}/${run.epochs}: ${pct}%| ${nInEpoch}/${perEpoch} [${SECONDS_PER_STEP}s/it, loss=${run.loss.toFixed(4)}]`);
    }
  }

  function finish(run, status, message) {
    Object.assign(run, { status, message, stage: null, finished_at: now() });
    if (status === "done") run.eta_seconds = 0;
  }

  // ---------------------------------------------------------------- API
  const json = (body, status = 200) =>
    new Response(status === 204 ? null : JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  const error = (status, detail) => json({ detail }, status);
  const activeRun = () => state.runs.find((r) => ACTIVE.has(r.status)) || null;

  function publicRun(run) {
    const { history, log, waitedAt, stageIndex, stageStart, cancelledAt, seed, ...rest } = run;
    const progress = run.total_steps ? run.step / run.total_steps : (run.status === "done" ? 1 : 0);
    return { ...rest, progress, has_metrics: history.length > 0 };
  }

  function modelsPayload() {
    return { active: state.active, model_id: `Chatterbox multilingual (${state.active})`, official: ["v3", "v2"],
      custom: state.models, switchable: true, error: null };
  }

  function route(method, url, body) {
    const path = url.pathname;
    let m;
    if (method === "GET" && path === "/api/training") {
      const active = activeRun();
      const training = active && ["running", "cancelling"].includes(active.status);
      return json({
        trainer: { online: true, last_seen_seconds: 3, busy: active ? active.id : null, allow_cpu: false,
          gpu: { ...GPU, free_gb: training ? 6.4 : 22.8 } },
        active: active && publicRun(active),
        runs: state.runs.slice().sort((a, b) => b.created_at - a.created_at).map(publicRun),
        defaults: DEFAULTS,
        datasets: { qc: { ready: true, clips: CLIPS.fr_ca }, own: { ready: true, clips: 312 } },
        models: state.models,
      });
    }
    if (method === "GET" && path === "/api/models") return json(modelsPayload());
    if (method === "POST" && path === "/api/models/active") {
      const name = body.model;
      if (!["v3", "v2"].includes(name) && !state.models.some((x) => x.name === name)) return error(404, `Model '${name}' not found in data/models`);
      state.active = name;
      return json({ model: name, pending: true });
    }
    if (method === "DELETE" && (m = path.match(/^\/api\/models\/(.+)$/))) {
      const name = decodeURIComponent(m[1]);
      if (name === state.active) return error(409, "This model is active; switch to another one first");
      state.models = state.models.filter((x) => x.name !== name);
      return json(null, 204);
    }
    if (method === "POST" && path === "/api/training/runs") {
      const active = activeRun();
      if (active) return error(409, `Run '${active.name}' is still ${active.status}; one run at a time`);
      if (body.recipe === "personal" && !body.base_model) return error(400, "Pick the fine-tuned model to start from (base_model)");
      const name = (body.name || "").trim() || (body.recipe === "fr_ca" ? "Quebec accent" : "My voice");
      const params = { ...DEFAULTS[body.recipe] };
      for (const key of ["epochs", "lora_rank", "learning_rate"]) if (body[key] !== undefined) params[key] = body[key];
      if (body.base_model) params.base_model = body.base_model;
      const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "") || "model";
      params.output_name = `t3_${slug}_${rid().slice(0, 6)}`;
      const run = { id: rid(), name, recipe: body.recipe, params, status: "queued", stage: null, message: null, step: 0,
        total_steps: 0, epoch: 0, epochs: params.epochs, loss: null, eta_seconds: null, output_model: null,
        created_at: now(), started_at: null, finished_at: null, history: [], log: [], seed: Math.random() * 6 };
      state.runs.push(run);
      return json(publicRun(run), 201);
    }
    if ((m = path.match(/^\/api\/training\/runs\/([0-9a-f]+)(\/log)?$/))) {
      const run = state.runs.find((r) => r.id === m[1]);
      if (!run) return error(404, "Run not found");
      if (method === "GET" && m[2]) return json({ lines: run.log.slice(-(Number(url.searchParams.get("lines")) || 200)) });
      if (method === "GET") return json(publicRun(run));
      if (method === "DELETE") {
        if (run.status === "queued") finish(run, "cancelled", "Cancelled before it started");
        else if (["waiting_gpu", "running"].includes(run.status)) Object.assign(run, { status: "cancelling", cancelledAt: now() });
        else if (!ACTIVE.has(run.status)) state.runs = state.runs.filter((r) => r !== run);
        return json(null, 204);
      }
    }
    return error(404, "Not available in the static demo");
  }

  window.fetch = async (input, options = {}) => {
    const url = new URL(typeof input === "string" ? input : input.url, window.location.href);
    if (!url.pathname.startsWith("/api/")) return realFetch(input, options);
    let body = {};
    try { body = options.body ? JSON.parse(options.body) : {}; } catch (_) { /* not JSON */ }
    return route((options.method || "GET").toUpperCase(), url, body);
  };

  setInterval(() => state.runs.forEach(advance), 500);

  // ---------------------------------------------------------------- loss chart
  // The app serves the toolkit's training_metrics.png; the demo draws the loss as an SVG.
  function lossChart(run) {
    const pts = run.history;
    const w = 720, h = 360, pad = 48;
    const maxStep = Math.max(...pts.map((p) => p[0]), 1);
    const losses = pts.map((p) => p[1]);
    const lo = Math.min(...losses) - 0.1, hi = Math.max(...losses) + 0.1;
    const x = (s) => pad + ((w - 2 * pad) * s) / maxStep;
    const y = (l) => h - pad - ((h - 2 * pad) * (l - lo)) / (hi - lo);
    const line = pts.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join(" ");
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" font-family="system-ui,sans-serif">
      <rect width="100%" height="100%" fill="#14161c"/>
      <text x="${pad}" y="28" fill="#e6e8ee" font-size="16" font-weight="600">${run.name} · training loss</text>
      <line x1="${pad}" y1="${h - pad}" x2="${w - pad}" y2="${h - pad}" stroke="#2a2f3d"/>
      <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${h - pad}" stroke="#2a2f3d"/>
      <text x="${pad}" y="${h - 16}" fill="#8b91a5" font-size="12">step 0</text>
      <text x="${w - pad}" y="${h - 16}" fill="#8b91a5" font-size="12" text-anchor="end">step ${maxStep}</text>
      <text x="${pad - 8}" y="${y(hi - 0.1) + 4}" fill="#8b91a5" font-size="12" text-anchor="end">${(hi - 0.1).toFixed(2)}</text>
      <text x="${pad - 8}" y="${y(lo + 0.1) + 4}" fill="#8b91a5" font-size="12" text-anchor="end">${(lo + 0.1).toFixed(2)}</text>
      <path d="${line}" fill="none" stroke="#6c8cff" stroke-width="2"/>
      <text x="${w - pad}" y="28" fill="#8b91a5" font-size="12" text-anchor="end">static demo · simulated</text>
    </svg>`;
  }

  document.addEventListener("click", (event) => {
    const link = event.target.closest('a[href*="/metrics.png"]');
    if (!link) return;
    event.preventDefault();
    const id = link.getAttribute("href").split("/")[4];
    const run = state.runs.find((r) => r.id === id);
    if (!run) return;
    const blob = new Blob([lossChart(run)], { type: "image/svg+xml" });
    window.open(URL.createObjectURL(blob), "_blank", "noopener");
  }, true);

  window.demoTrainingActive = () => Boolean(activeRun());
})();
