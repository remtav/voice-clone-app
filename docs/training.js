/* Fine-tuning card: trainer status, active model, training runs.
 * Talks to /api/training and /api/models; the trainer service does the work. */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const ACTIVE = new Set(["queued", "waiting_gpu", "running", "cancelling"]);
  const STATUS_LABEL = {
    queued: "queued", waiting_gpu: "waiting for GPU", running: "running", cancelling: "cancelling",
    done: "done", failed: "failed", cancelled: "cancelled",
  };
  const RECIPE_LABEL = { fr_ca: "Quebec accent", personal: "My voice" };
  const state = { data: null, models: null, openLogs: new Set(), busy: false };

  function toast(message, kind = "error") {
    const node = $("#toast");
    node.textContent = message;
    node.className = `toast ${kind}`;
    setTimeout(() => node.classList.add("hidden"), kind === "error" ? 6000 : 3000);
  }

  async function api(path, options = {}) {
    const res = await fetch(path, { credentials: "same-origin", ...options });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
      } catch (_) { /* not JSON */ }
      throw new Error(detail);
    }
    return res.status === 204 ? null : res.json();
  }
  const postJSON = (path, body) =>
    api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined && v !== false) node.setAttribute(k, v);
    }
    for (const child of children) if (child) node.append(child);
    return node;
  }

  function fmtDuration(seconds) {
    if (seconds === null || seconds === undefined) return "";
    const s = Math.max(0, Math.round(seconds));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    if (h) return `${h} h ${String(m).padStart(2, "0")} min`;
    if (m) return `${m} min`;
    return `${s} s`;
  }
  const fmtWhen = (t) => new Date(t * 1000).toLocaleString([], { dateStyle: "short", timeStyle: "short" });
  const fmtSize = (bytes) => `${(bytes / 1e9).toFixed(1)} GB`;

  // ---------------------------------------------------------------- data
  async function refresh() {
    if ($("#app-view").classList.contains("hidden") || state.busy) return; // not logged in yet
    state.busy = true;
    try {
      const [data, models] = await Promise.all([api("/api/training"), api("/api/models")]);
      state.data = data;
      state.models = models;
      render();
      await refreshLogs();
    } catch (_) {
      /* polling: stay quiet, the status bar already reports connection problems */
    } finally {
      state.busy = false;
    }
  }

  function render() {
    renderTrainer();
    renderModels();
    renderForm();
    renderRuns();
    renderGenerationLock();
  }

  function renderTrainer() {
    const node = $("#trainer-status");
    const t = state.data.trainer;
    if (!t.online) {
      node.className = "trainer-status small err";
      node.textContent = "Trainer offline — start it with: docker compose up -d trainer";
    } else if (t.gpu) {
      node.className = "trainer-status small ok";
      node.textContent = `Trainer online · ${t.gpu.name} · ${t.gpu.free_gb}/${t.gpu.total_gb} GB free`;
    } else {
      node.className = "trainer-status small err";
      node.textContent = t.allow_cpu ? "Trainer online · CPU only (development mode)" : "Trainer online · no GPU visible";
    }
  }

  function renderModels() {
    const m = state.models;
    const select = $("#model-select");
    const previous = select.value;
    select.innerHTML = "";
    select.append(el("option", { value: "v3", text: "Official v3 (Chatterbox multilingual)" }));
    select.append(el("option", { value: "v2", text: "Official v2" }));
    for (const model of m.custom) {
      select.append(el("option", { value: model.name, text: `${model.name} · ${fmtSize(model.size_bytes)} · ${fmtWhen(model.created_at)}` }));
    }
    const known = [...select.options].map((o) => o.value);
    select.value = select.dataset.touched === "1" && known.includes(previous) ? previous : (m.active || "v3");
    select.disabled = !m.switchable;
    syncModelButtons();
    const err = $("#model-error");
    err.textContent = m.error ? `Could not switch model: ${m.error}` : "";
    err.classList.toggle("hidden", !m.error);
  }

  function syncModelButtons() {
    const m = state.models;
    if (!m) return;
    const chosen = $("#model-select").value;
    $("#model-apply").disabled = !m.switchable || chosen === m.active;
    $("#model-delete").classList.toggle("hidden", chosen === m.active || chosen === "v3" || chosen === "v2");
  }

  function renderForm() {
    const d = state.data;
    const recipe = $("#tr-recipe").value;
    const defaults = d.defaults[recipe];
    $("#tr-epochs").placeholder = `default ${defaults.epochs}`;
    $("#tr-lr").placeholder = `default ${defaults.learning_rate}`;
    $("#tr-name").placeholder = RECIPE_LABEL[recipe];
    document.querySelectorAll(".personal-only").forEach((n) => n.classList.toggle("hidden", recipe !== "personal"));

    const base = $("#tr-base");
    const keep = base.value;
    base.innerHTML = "";
    for (const model of d.models) base.append(el("option", { value: model.name, text: model.name }));
    if ([...base.options].some((o) => o.value === keep)) base.value = keep;

    let hint = "";
    let blocked = false;
    if (recipe === "fr_ca") {
      hint = d.datasets.qc.ready
        ? `Quebec corpus ready (${d.datasets.qc.clips} clips).`
        : "First run: the trainer downloads and prepares the Quebec corpus (1.8 GB, CC0).";
    } else if (!d.models.length) {
      hint = "Train a Quebec accent model first: this recipe starts from one.";
      blocked = true;
    } else if (!d.datasets.own.ready) {
      hint = "No recordings of your voice yet: run scripts.finetune.segment_recording into data/finetune/me (see README).";
      blocked = true;
    } else {
      hint = `${d.datasets.own.clips} clips of your voice ready.`;
    }
    if (d.active) hint = `A run is already in progress ("${d.active.name}"): one at a time.`;
    $("#tr-hint").textContent = hint;
    $("#tr-start").disabled = Boolean(d.active) || blocked;
  }

  function runMeta(run) {
    const bits = [];
    if (run.total_steps) bits.push(`step ${run.step}/${run.total_steps}`);
    if (run.epochs && run.epoch) bits.push(`epoch ${run.epoch}/${run.epochs}`);
    if (run.loss !== null && run.loss !== undefined) bits.push(`loss ${Number(run.loss).toFixed(3)}`);
    if (run.status === "running" && run.eta_seconds) bits.push(`≈ ${fmtDuration(run.eta_seconds)} left`);
    if (run.started_at && run.finished_at) bits.push(`took ${fmtDuration(run.finished_at - run.started_at)}`);
    return bits.join(" · ");
  }

  function renderRuns() {
    const list = $("#run-list");
    const runs = state.data.runs;
    $("#run-empty").classList.toggle("hidden", runs.length > 0);
    list.innerHTML = "";
    for (const run of runs) list.append(renderRun(run));
  }

  function renderRun(run) {
    const active = ACTIVE.has(run.status);
    const pct = Math.round((run.progress || 0) * 100);
    const indeterminate = active && !run.total_steps;
    const stageText = [run.stage, run.message].filter(Boolean).join(" — ");
    const actions = [];
    actions.push(el("span", { class: "job-params", text: `${RECIPE_LABEL[run.recipe] || run.recipe} · ${fmtWhen(run.created_at)}` }));
    actions.push(el("span", { class: "spacer" }));
    if (run.status === "done" && run.output_model && state.models.active !== run.output_model) {
      actions.push(el("button", { class: "btn btn-primary small", type: "button", text: "Use this model",
        onclick: () => activate(run.output_model) }));
    }
    if (run.has_metrics) {
      actions.push(el("a", { class: "btn btn-ghost small", href: `/api/training/runs/${run.id}/metrics.png`,
        target: "_blank", rel: "noopener", text: "Loss chart" }));
    }
    actions.push(el("button", { class: "btn btn-ghost small", type: "button",
      text: state.openLogs.has(run.id) ? "Hide log" : "Log", onclick: () => toggleLog(run.id) }));
    actions.push(el("button", { class: "btn btn-ghost small", type: "button",
      text: active ? (run.status === "cancelling" ? "Cancelling…" : "Cancel") : "Delete",
      disabled: run.status === "cancelling", onclick: () => removeRun(run, active) }));

    return el("li", { class: "job-item", "data-run": run.id }, [
      el("div", { class: "job-head" }, [
        el("span", { class: "voice", text: run.name }),
        el("span", { class: `badge ${run.status}`, text: STATUS_LABEL[run.status] || run.status }),
        run.output_model && run.status === "done" ? el("span", { class: "muted small", text: run.output_model }) : null,
      ]),
      stageText ? el("div", { class: `run-stage${run.status === "failed" ? " job-error" : ""}`, text: stageText }) : null,
      active || run.total_steps
        ? el("div", { class: `progress${indeterminate ? " indeterminate" : ""}` },
          [el("div", { style: indeterminate ? null : `width:${pct}%` })])
        : null,
      runMeta(run) ? el("div", { class: "run-meta", text: runMeta(run) }) : null,
      state.openLogs.has(run.id) ? el("pre", { class: "run-log", "data-log": run.id, text: "Loading…" }) : null,
      el("div", { class: "job-actions" }, actions),
    ]);
  }

  function renderGenerationLock() {
    const run = state.data.active;
    document.body.classList.toggle("training-active", Boolean(run));
    const note = $("#gen-training-note");
    note.classList.toggle("hidden", !run);
    if (run) {
      note.textContent = `Training "${run.name}" is using the GPU (${STATUS_LABEL[run.status]}, `
        + `${Math.round((run.progress || 0) * 100)}%). Generation resumes when it ends.`;
    }
  }

  async function refreshLogs() {
    for (const id of state.openLogs) {
      const pre = document.querySelector(`pre[data-log="${id}"]`);
      if (!pre) continue;
      try {
        const { lines } = await api(`/api/training/runs/${id}/log?lines=120`);
        const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 8;
        pre.textContent = lines.length ? lines.join("\n") : "No output yet.";
        if (atBottom) pre.scrollTop = pre.scrollHeight;
      } catch (err) {
        pre.textContent = err.message;
      }
    }
  }

  // ---------------------------------------------------------------- actions
  function toggleLog(id) {
    if (state.openLogs.has(id)) state.openLogs.delete(id);
    else state.openLogs.add(id);
    renderRuns();
    refreshLogs();
  }

  async function removeRun(run, active) {
    const question = active
      ? `Cancel "${run.name}"? The toolkit cannot resume: progress is lost.`
      : `Delete "${run.name}" and its working files? ${run.output_model ? "The published model is kept." : ""}`;
    if (!confirm(question)) return;
    try {
      await api(`/api/training/runs/${run.id}`, { method: "DELETE" });
      state.openLogs.delete(run.id);
      await refresh();
    } catch (err) {
      toast(err.message);
    }
  }

  async function activate(model) {
    try {
      await postJSON("/api/models/active", { model });
      toast(`Switching to ${model}…`, "ok");
      $("#model-select").dataset.touched = "";
      setTimeout(refresh, 1500);
    } catch (err) {
      toast(err.message);
    }
  }

  async function deleteModel() {
    const model = $("#model-select").value;
    if (!confirm(`Delete ${model}? This removes the checkpoint file.`)) return;
    try {
      await api(`/api/models/${encodeURIComponent(model)}`, { method: "DELETE" });
      $("#model-select").dataset.touched = "";
      await refresh();
    } catch (err) {
      toast(err.message);
    }
  }

  async function startRun(event) {
    event.preventDefault();
    const recipe = $("#tr-recipe").value;
    const body = { recipe, name: $("#tr-name").value.trim() };
    if ($("#tr-epochs").value) body.epochs = Number($("#tr-epochs").value);
    if ($("#tr-rank").value) body.lora_rank = Number($("#tr-rank").value);
    if ($("#tr-lr").value) body.learning_rate = Number($("#tr-lr").value);
    if (recipe === "personal") body.base_model = $("#tr-base").value;
    try {
      const run = await postJSON("/api/training/runs", body);
      toast(`Training "${run.name}" queued`, "ok");
      $("#training-new").open = false;
      await refresh();
    } catch (err) {
      toast(err.message);
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    $("#training-form").addEventListener("submit", startRun);
    $("#tr-recipe").addEventListener("change", () => state.data && renderForm());
    $("#model-select").addEventListener("change", (e) => { e.target.dataset.touched = "1"; syncModelButtons(); });
    $("#model-apply").addEventListener("click", () => activate($("#model-select").value));
    $("#model-delete").addEventListener("click", deleteModel);
    setTimeout(refresh, 800);
    setInterval(() => { if (!document.hidden) refresh(); }, 4000);
  });
})();
