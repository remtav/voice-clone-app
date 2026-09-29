(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const ACTIVE = new Set(["queued", "running", "cancelling"]);

  const state = {
    config: null,
    voices: [],
    jobs: new Map(), // id -> job
    jobNodes: new Map(), // id -> <li>
    selectedVoice: null,
    settingsTouched: false,
    recordedBlob: null,
    mediaRecorder: null,
    recordTimer: null,
    pollTimer: null,
    statusTimer: null,
  };

  // ---------------------------------------------------------------- utils
  let toastTimer = null;
  function toast(message, kind = "error") {
    const el = $("#toast");
    el.textContent = message;
    el.className = `toast ${kind}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.add("hidden"), kind === "error" ? 6000 : 3000);
  }

  async function api(path, options = {}) {
    const res = await fetch(path, { credentials: "same-origin", ...options });
    if (res.status === 401 && path !== "/api/login") {
      showLogin();
      throw new Error("Authentication required");
    }
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
      } catch (_) { /* not JSON */ }
      throw new Error(detail);
    }
    if (res.status === 204) return null;
    return res.json();
  }

  const postJSON = (path, body) =>
    api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

  function fmtSeconds(s) {
    if (s == null || Number.isNaN(s)) return "";
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${String(sec).padStart(2, "0")}`;
  }
  const fmtWhen = (ts) => new Date(ts * 1000).toLocaleString([], { dateStyle: "short", timeStyle: "short" });
  const langName = (code) => (state.config?.engine.languages?.[code] || code || "").toString();

  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined) node.setAttribute(k, v);
    }
    for (const child of children) node.append(child);
    return node;
  }

  // ---------------------------------------------------------------- boot
  async function init() {
    try {
      state.config = await api("/api/config");
    } catch (err) {
      toast(`Cannot reach the server: ${err.message}`);
      return;
    }
    if (state.config.auth_required && !state.config.authenticated) showLogin();
    else boot();
  }

  function showLogin() {
    $("#login-view").classList.remove("hidden");
    $("#app-view").classList.add("hidden");
    $("#logout-btn").classList.add("hidden");
    stopPolling();
    clearInterval(state.statusTimer);
    setTimeout(() => $("#login-password").focus(), 0);
  }

  async function boot() {
    $("#login-view").classList.add("hidden");
    $("#app-view").classList.remove("hidden");
    if (state.config.auth_required) $("#logout-btn").classList.remove("hidden");

    fillLanguageSelects();
    const limits = state.config.limits;
    $("#char-max").textContent = limits.max_text_chars;
    $("#max-ref-seconds").textContent = limits.max_reference_seconds;
    updateCharCount();

    await Promise.all([loadVoices(), loadJobs()]);
    defaultPresetFor(state.voices.find((v) => v.id === state.selectedVoice));
    refreshStatus();
    clearInterval(state.statusTimer);
    state.statusTimer = setInterval(refreshStatus, 5000);
    schedulePoll();
  }

  function fillLanguageSelects() {
    const engine = state.config.engine;
    const entries = Object.entries(engine.languages || { en: "English" }).sort((a, b) => a[1].localeCompare(b[1]));
    for (const id of ["#voice-language", "#gen-language"]) {
      const select = $(id);
      select.innerHTML = "";
      for (const [code, name] of entries) select.append(el("option", { value: code, text: `${name} (${code})` }));
      select.value = entries.some(([c]) => c === "en") ? "en" : entries[0]?.[0];
      select.disabled = !engine.multilingual;
    }
  }

  // ---------------------------------------------------------------- status
  async function refreshStatus() {
    const bar = $("#status-bar");
    try {
      const s = await api("/api/status");
      const parts = [];
      let cls = "ok";
      if (s.load_error) {
        cls = "err";
        parts.push(`Model failed to load: ${s.load_error}`);
      } else if (s.model_loading) {
        cls = "busy";
        parts.push("Loading model… (first start downloads several GB)");
      } else if (!s.loaded) {
        cls = "busy";
        parts.push("Model not loaded yet (loads on first generation)");
      } else {
        parts.push(`Ready · ${s.model_id || `${s.name} ${s.variant}`}`);
      }
      if (s.device) parts.push(s.device);
      if (s.gpu) parts.push(`${s.gpu.name} ${(s.gpu.memory_reserved_mb / 1024).toFixed(1)}/${(s.gpu.memory_total_mb / 1024).toFixed(0)} GB`);
      if (s.current_job_id) {
        cls = cls === "err" ? cls : "busy";
        parts.push(`generating ${s.current_job_id}`);
      }
      if (s.queue_size) parts.push(`${s.queue_size} queued`);
      bar.className = `status ${cls}`;
      bar.innerHTML = "";
      bar.append(el("span", { class: "dot" }), document.createTextNode(parts.join(" · ")));
    } catch (err) {
      bar.className = "status err";
      bar.innerHTML = "";
      bar.append(el("span", { class: "dot" }), document.createTextNode(`Offline: ${err.message}`));
    }
  }

  // ---------------------------------------------------------------- voices
  async function loadVoices() {
    state.voices = await api("/api/voices");
    if (!state.voices.some((v) => v.id === state.selectedVoice)) {
      state.selectedVoice = state.voices[0]?.id ?? null;
    }
    renderVoices();
  }

  let previewAudio = null;
  function togglePreview(button, src) {
    if (previewAudio && previewAudio.src.endsWith(src) && !previewAudio.paused) {
      previewAudio.pause();
      button.textContent = "▶";
      return;
    }
    if (previewAudio) {
      previewAudio.pause();
      document.querySelectorAll(".voice-item .play").forEach((b) => (b.textContent = "▶"));
    }
    previewAudio = new Audio(src);
    previewAudio.addEventListener("ended", () => (button.textContent = "▶"));
    previewAudio.play().catch((err) => toast(`Playback failed: ${err.message}`));
    button.textContent = "■";
  }

  function renderVoices() {
    const list = $("#voice-list");
    list.innerHTML = "";
    $("#voice-empty").classList.toggle("hidden", state.voices.length > 0);
    for (const voice of state.voices) {
      const meta = [voice.language ? langName(voice.language) : null, fmtSeconds(voice.duration)].filter(Boolean).join(" · ");
      const item = el("li", { class: `voice-item${voice.id === state.selectedVoice ? " selected" : ""}`, onclick: () => selectVoice(voice.id) }, [
        el("button", {
          class: "icon-btn play btn",
          type: "button",
          title: "Play reference",
          text: "▶",
          onclick: (e) => { e.stopPropagation(); togglePreview(e.currentTarget, `/api/voices/${voice.id}/audio`); },
        }),
        el("div", { class: "info" }, [el("div", { class: "name", text: voice.name }), el("div", { class: "meta", text: meta })]),
        el("button", {
          class: "icon-btn btn",
          type: "button",
          title: "Rename",
          text: "✎",
          onclick: async (e) => {
            e.stopPropagation();
            const name = prompt("Voice name", voice.name);
            if (name && name.trim() && name.trim() !== voice.name) {
              try {
                await api(`/api/voices/${voice.id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: name.trim() }) });
                await loadVoices();
              } catch (err) { toast(err.message); }
            }
          },
        }),
        el("button", {
          class: "icon-btn btn",
          type: "button",
          title: "Delete voice",
          text: "🗑",
          onclick: async (e) => {
            e.stopPropagation();
            if (!confirm(`Delete voice "${voice.name}"? Generated audio is kept.`)) return;
            try {
              await api(`/api/voices/${voice.id}`, { method: "DELETE" });
              await loadVoices();
              toast("Voice deleted", "ok");
            } catch (err) { toast(err.message); }
          },
        }),
      ]);
      list.append(item);
    }
    renderVoiceSelect();
  }

  function renderVoiceSelect() {
    const select = $("#gen-voice");
    select.innerHTML = "";
    for (const voice of state.voices) select.append(el("option", { value: voice.id, text: voice.name }));
    if (state.selectedVoice) select.value = state.selectedVoice;
    $("#gen-btn").disabled = state.voices.length === 0;
  }

  function selectVoice(id) {
    state.selectedVoice = id;
    const voice = state.voices.find((v) => v.id === id);
    if (voice?.language && state.config.engine.multilingual) $("#gen-language").value = voice.language;
    defaultPresetFor(voice);
    renderVoices();
  }

  // ---------------------------------------------------------------- recording
  function setRecordedBlob(blob) {
    state.recordedBlob = blob;
    const preview = $("#record-preview");
    if (blob) {
      preview.src = URL.createObjectURL(blob);
      preview.classList.remove("hidden");
    } else {
      preview.removeAttribute("src");
      preview.classList.add("hidden");
    }
    updateSaveButton();
  }

  function updateSaveButton() {
    const tab = document.querySelector(".tab.active")?.dataset.tab;
    const hasInput = tab === "record" ? !!state.recordedBlob : ($("#voice-file").files?.length ?? 0) > 0;
    $("#voice-save").disabled = !hasInput;
  }

  async function toggleRecording() {
    const button = $("#record-btn");
    if (state.mediaRecorder && state.mediaRecorder.state === "recording") {
      state.mediaRecorder.stop();
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      toast("Recording is not supported in this browser (needs HTTPS or localhost). Use Upload instead.");
      return;
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    } catch (err) {
      toast(`Microphone access denied: ${err.message}`);
      return;
    }
    const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"].find((m) => MediaRecorder.isTypeSupported(m)) || "";
    const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    const chunks = [];
    const started = Date.now();
    recorder.addEventListener("dataavailable", (e) => e.data.size && chunks.push(e.data));
    recorder.addEventListener("stop", () => {
      stream.getTracks().forEach((t) => t.stop());
      clearInterval(state.recordTimer);
      button.textContent = "● Record";
      button.classList.remove("recording");
      setRecordedBlob(new Blob(chunks, { type: recorder.mimeType || mime || "audio/webm" }));
      if (!$("#voice-name").value) $("#voice-name").focus();
    });
    state.mediaRecorder = recorder;
    setRecordedBlob(null);
    recorder.start(250);
    button.textContent = "■ Stop";
    button.classList.add("recording");
    const maxMs = state.config.limits.max_reference_seconds * 1000;
    state.recordTimer = setInterval(() => {
      const elapsed = Date.now() - started;
      $("#record-timer").textContent = fmtSeconds(elapsed / 1000);
      if (elapsed >= maxMs && recorder.state === "recording") recorder.stop();
    }, 200);
  }

  async function saveVoice(event) {
    event.preventDefault();
    const tab = document.querySelector(".tab.active")?.dataset.tab;
    const form = new FormData();
    if (tab === "record") {
      if (!state.recordedBlob) return toast("Record something first");
      const type = state.recordedBlob.type || "";
      const ext = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";
      form.append("file", state.recordedBlob, `recording.${ext}`);
    } else {
      const file = $("#voice-file").files?.[0];
      if (!file) return toast("Choose a file first");
      form.append("file", file, file.name);
    }
    form.append("name", $("#voice-name").value.trim());
    form.append("language", $("#voice-language").value || "");

    const button = $("#voice-save");
    button.disabled = true;
    button.textContent = "Saving…";
    try {
      const voice = await api("/api/voices", { method: "POST", body: form });
      toast(`Voice "${voice.name}" saved (${fmtSeconds(voice.duration)})`, "ok");
      $("#voice-name").value = "";
      $("#voice-file").value = "";
      setRecordedBlob(null);
      $("#record-timer").textContent = "0:00";
      state.selectedVoice = voice.id;
      await loadVoices();
      selectVoice(voice.id);
    } catch (err) {
      toast(err.message);
    } finally {
      button.textContent = "Save voice";
      updateSaveButton();
    }
  }

  // ---------------------------------------------------------------- generation
  function updateCharCount() {
    const len = $("#gen-text").value.length;
    const max = state.config?.limits.max_text_chars ?? 5000;
    $("#char-count").textContent = len;
    $("#char-count").parentElement.classList.toggle("over", len > max);
  }

  async function generate(event) {
    event.preventDefault();
    const text = $("#gen-text").value.trim();
    if (!text) return toast("Enter some text first");
    const voiceId = $("#gen-voice").value;
    if (!voiceId) return toast("Add a voice first");
    const seedRaw = $("#seed").value.trim();
    const body = {
      voice_id: voiceId,
      text,
      language: $("#gen-language").value || null,
      exaggeration: parseFloat($("#exaggeration").value),
      cfg_weight: parseFloat($("#cfg").value),
      temperature: parseFloat($("#temp").value),
      seed: seedRaw === "" ? null : parseInt(seedRaw, 10),
    };
    const button = $("#gen-btn");
    button.disabled = true;
    try {
      const job = await postJSON("/api/generate", body);
      upsertJob(job, true);
      toast("Queued", "ok");
      schedulePoll();
      refreshStatus();
    } catch (err) {
      toast(err.message);
    } finally {
      button.disabled = false;
    }
  }

  // ---------------------------------------------------------------- jobs
  async function loadJobs() {
    const jobs = await api("/api/jobs?limit=100");
    const seen = new Set(jobs.map((j) => j.id));
    for (const id of [...state.jobs.keys()]) {
      if (!seen.has(id)) removeJobNode(id);
    }
    for (const job of jobs.slice().reverse()) upsertJob(job, false);
    // Keep newest first.
    const list = $("#job-list");
    const ordered = jobs.map((j) => state.jobNodes.get(j.id)).filter(Boolean);
    list.replaceChildren(...ordered);
    $("#job-empty").classList.toggle("hidden", jobs.length > 0);
  }

  function removeJobNode(id) {
    state.jobNodes.get(id)?.remove();
    state.jobNodes.delete(id);
    state.jobs.delete(id);
    $("#job-empty").classList.toggle("hidden", state.jobs.size > 0);
  }

  function upsertJob(job, prepend) {
    const previous = state.jobs.get(job.id);
    state.jobs.set(job.id, job);
    let node = state.jobNodes.get(job.id);
    if (!node) {
      node = buildJobNode(job);
      state.jobNodes.set(job.id, node);
      if (prepend) $("#job-list").prepend(node);
      else $("#job-list").append(node);
    }
    if (!previous || previous.status !== job.status || previous.progress_done !== job.progress_done || previous.progress_total !== job.progress_total) {
      updateJobNode(node, job, previous);
    }
    $("#job-empty").classList.add("hidden");
  }

  function buildJobNode(job) {
    const textNode = el("div", { class: "job-text", text: job.text });
    const node = el("li", { class: "job-item", "data-id": job.id }, [
      el("div", { class: "job-head" }, [
        el("span", { class: "voice", text: job.voice_name }),
        el("span", { class: "badge", text: job.status }),
        el("span", { class: "lang muted small", text: job.language ? langName(job.language) : "" }),
        el("span", { class: "when", text: fmtWhen(job.created_at) }),
      ]),
      textNode,
      el("button", {
        class: `job-text-toggle${job.text.length > 220 ? "" : " hidden"}`,
        type: "button",
        text: "Show more",
        onclick: (e) => {
          const expanded = textNode.classList.toggle("expanded");
          e.currentTarget.textContent = expanded ? "Show less" : "Show more";
        },
      }),
      el("div", { class: "progress hidden" }, [el("div")]),
      el("div", { class: "job-error hidden" }),
      el("div", { class: "job-audio" }),
      el("div", { class: "job-actions" }, [
        el("span", { class: "job-params", text: describeParams(job) }),
        el("span", { class: "spacer" }),
        el("a", { class: "btn small download hidden", href: `/api/jobs/${job.id}/audio?download=1`, text: "Download WAV" }),
        el("button", { class: "btn small reuse", type: "button", text: "Reuse text", onclick: () => reuseJob(job.id) }),
        el("button", { class: "btn small remove", type: "button", text: "Delete", onclick: () => deleteJob(job.id) }),
      ]),
    ]);
    return node;
  }

  function describeParams(job) {
    const p = job.params || {};
    const bits = [`exag ${Number(p.exaggeration ?? 0.5).toFixed(2)}`, `cfg ${Number(p.cfg_weight ?? 0.5).toFixed(2)}`, `temp ${Number(p.temperature ?? 0.8).toFixed(2)}`];
    if (p.seed != null) bits.push(`seed ${p.seed}`);
    return bits.join(" · ");
  }

  function updateJobNode(node, job, previous) {
    const badge = node.querySelector(".badge");
    badge.textContent = job.status;
    badge.className = `badge ${job.status}`;

    const progress = node.querySelector(".progress");
    if (job.status === "running" || job.status === "queued" || job.status === "cancelling") {
      progress.classList.remove("hidden");
      const total = job.progress_total || 0;
      if (total > 0) {
        progress.classList.remove("indeterminate");
        progress.firstElementChild.style.width = `${Math.round((100 * (job.progress_done || 0)) / total)}%`;
      } else {
        progress.classList.add("indeterminate");
      }
    } else {
      progress.classList.add("hidden");
    }

    const error = node.querySelector(".job-error");
    error.classList.toggle("hidden", job.status !== "failed");
    if (job.status === "failed") error.textContent = job.error || "Generation failed";

    const audioBox = node.querySelector(".job-audio");
    if (job.status === "done" && job.has_audio && !audioBox.firstChild) {
      const audio = el("audio", { controls: "", preload: "metadata", src: `/api/jobs/${job.id}/audio` });
      audioBox.append(audio);
      if (previous && ACTIVE.has(previous.status)) {
        const secs = job.duration ? ` (${fmtSeconds(job.duration)})` : "";
        toast(`Audio ready${secs}`, "ok");
      }
    }
    node.querySelector(".download").classList.toggle("hidden", !(job.status === "done" && job.has_audio));
    node.querySelector(".remove").textContent = ACTIVE.has(job.status) ? "Cancel" : "Delete";
  }

  function reuseJob(id) {
    const job = state.jobs.get(id);
    if (!job) return;
    $("#gen-text").value = job.text;
    updateCharCount();
    if (job.language && state.config.engine.multilingual) $("#gen-language").value = job.language;
    if (state.voices.some((v) => v.id === job.voice_id)) selectVoice(job.voice_id);
    const p = job.params || {};
    state.settingsTouched = true;
    setSliders(p);
    $("#seed").value = p.seed ?? "";
    window.scrollTo({ top: 0, behavior: "smooth" });
    $("#gen-text").focus();
  }

  async function deleteJob(id) {
    const job = state.jobs.get(id);
    if (!job) return;
    const active = ACTIVE.has(job.status);
    if (!active && !confirm("Delete this generation and its audio?")) return;
    try {
      await api(`/api/jobs/${id}`, { method: "DELETE" });
      if (active) {
        toast("Cancelling…", "ok");
        schedulePoll();
      } else {
        removeJobNode(id);
      }
    } catch (err) {
      toast(err.message);
    }
  }

  function hasActiveJobs() {
    for (const job of state.jobs.values()) if (ACTIVE.has(job.status)) return true;
    return false;
  }

  function schedulePoll() {
    stopPolling();
    if (!hasActiveJobs()) return;
    state.pollTimer = setTimeout(async () => {
      try {
        const active = [...state.jobs.values()].filter((j) => ACTIVE.has(j.status));
        const updates = await Promise.all(active.map((j) => api(`/api/jobs/${j.id}`).catch(() => null)));
        for (const job of updates) if (job) upsertJob(job, false);
      } finally {
        schedulePoll();
      }
    }, 1500);
  }

  function stopPolling() {
    clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }

  // ---------------------------------------------------------------- auth
  async function login(event) {
    event.preventDefault();
    const error = $("#login-error");
    error.classList.add("hidden");
    try {
      await postJSON("/api/login", { password: $("#login-password").value });
      $("#login-password").value = "";
      state.config = await api("/api/config");
      boot();
    } catch (err) {
      error.textContent = err.message;
      error.classList.remove("hidden");
    }
  }

  async function logout() {
    try { await postJSON("/api/logout", {}); } catch (_) { /* ignore */ }
    showLogin();
  }


  // ---------------------------------------------------------------- presets
  // [slider, output, job param, preset data attribute]
  const SLIDERS = [
    ["#exaggeration", "#exaggeration-val", "exaggeration", "exag"],
    ["#cfg", "#cfg-val", "cfg_weight", "cfg"],
    ["#temp", "#temp-val", "temperature", "temp"],
  ];

  function setSliders(values) {
    for (const [slider, output, key] of SLIDERS) {
      if (values[key] == null) continue;
      $(slider).value = values[key];
      $(output).textContent = Number(values[key]).toFixed(2);
    }
    markPreset();
  }

  function applyPreset(btn) {
    setSliders(Object.fromEntries(SLIDERS.map(([, , key, attr]) => [key, btn.dataset[attr]])));
  }

  // French voices default to "Faithful accent" (keeps a regional accent such as
  // Quebec French) until the user picks settings themselves.
  function defaultPresetFor(voice) {
    if (state.settingsTouched) return;
    const name = voice?.language === "fr" ? "faithful" : "neutral";
    const btn = document.querySelector(`.preset[data-preset="${name}"]`);
    if (btn) applyPreset(btn);
  }

  function markPreset() {
    document.querySelectorAll(".preset").forEach((btn) => {
      const match = SLIDERS.every(([slider, , , attr]) => Math.abs(Number($(slider).value) - Number(btn.dataset[attr])) < 1e-6);
      btn.classList.toggle("active", match);
    });
  }

  // ---------------------------------------------------------------- wiring
  document.addEventListener("DOMContentLoaded", () => {
    $("#login-form").addEventListener("submit", login);
    $("#logout-btn").addEventListener("click", logout);
    $("#record-btn").addEventListener("click", toggleRecording);
    $("#voice-form").addEventListener("submit", saveVoice);
    $("#voice-file").addEventListener("change", updateSaveButton);
    $("#generate-form").addEventListener("submit", generate);
    $("#gen-text").addEventListener("input", updateCharCount);
    $("#gen-voice").addEventListener("change", (e) => selectVoice(e.target.value));
    $("#refresh-jobs").addEventListener("click", () => loadJobs().catch((err) => toast(err.message)));

    for (const [slider, output] of SLIDERS) {
      $(slider).addEventListener("input", (e) => {
        $(output).textContent = Number(e.target.value).toFixed(2);
        state.settingsTouched = true;
        markPreset();
      });
    }
    document.querySelectorAll(".preset").forEach((btn) => btn.addEventListener("click", () => {
      state.settingsTouched = true;
      applyPreset(btn);
    }));
    markPreset();

    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
        document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("hidden", p.id !== `tab-${tab.dataset.tab}`));
        updateSaveButton();
      });
    });

    $("#gen-text").addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") $("#generate-form").requestSubmit();
    });

    init();
  });
})();
