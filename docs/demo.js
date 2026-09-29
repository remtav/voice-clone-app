/*
 * Static demo of the Voice Clone front-end for GitHub Pages.
 *
 * This is a MAQUETTE: it renders the real UI with canned data and simulates
 * generation entirely in the browser. There is NO backend, no network call,
 * and nothing is stored anywhere. "Generated" audio is a short tone synthesised
 * with the Web Audio math below, so the players and downloads actually work.
 */
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const ACTIVE = new Set(["queued", "running"]);

  const LANGUAGES = {
    ar: "Arabic", da: "Danish", de: "German", el: "Greek", en: "English",
    es: "Spanish", fi: "Finnish", fr: "French", he: "Hebrew", hi: "Hindi",
    it: "Italian", ja: "Japanese", ko: "Korean", ms: "Malay", nl: "Dutch",
    no: "Norwegian", pl: "Polish", pt: "Portuguese", ru: "Russian", sv: "Swedish",
    sw: "Swahili", tr: "Turkish", zh: "Chinese",
  };
  const LIMITS = { max_text_chars: 5000, max_reference_seconds: 30 };

  const state = {
    voices: [],
    jobs: new Map(),
    jobNodes: new Map(),
    selectedVoice: null,
    recordedTone: null,
    recording: false,
    recordTimer: null,
    seq: 0,
  };

  const langName = (code) => LANGUAGES[code] || code || "";
  const newId = () => (++state.seq).toString(36) + Math.random().toString(36).slice(2, 6);
  const fmtSeconds = (s) => {
    if (s == null || Number.isNaN(s)) return "";
    const m = Math.floor(s / 60);
    return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  };
  const fmtWhen = (ts) => new Date(ts).toLocaleString([], { dateStyle: "short", timeStyle: "short" });

  let toastTimer = null;
  function toast(message, kind = "ok") {
    const elx = $("#toast");
    elx.textContent = message;
    elx.className = `toast ${kind}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => elx.classList.add("hidden"), 3000);
  }

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

  // ---- In-browser WAV synthesis (so the demo audio actually plays) --------
  function toneWav(seconds, freq = 210, sr = 22050) {
    const n = Math.floor(seconds * sr);
    const buffer = new ArrayBuffer(44 + n * 2);
    const view = new DataView(buffer);
    const wstr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
    wstr(0, "RIFF"); view.setUint32(4, 36 + n * 2, true); wstr(8, "WAVE");
    wstr(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
    view.setUint16(22, 1, true); view.setUint32(24, sr, true); view.setUint32(28, sr * 2, true);
    view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    wstr(36, "data"); view.setUint32(40, n * 2, true);
    for (let i = 0; i < n; i++) {
      const t = i / sr;
      // A couple of harmonics with a soft vibrato and an envelope: pleasant-ish.
      const vib = 1 + 0.02 * Math.sin(2 * Math.PI * 5 * t);
      let s = 0.6 * Math.sin(2 * Math.PI * freq * vib * t);
      s += 0.25 * Math.sin(2 * Math.PI * freq * 2 * t);
      const env = Math.min(1, t * 12, (seconds - t) * 6);
      view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, s * env)) * 0.5 * 32767, true);
    }
    return URL.createObjectURL(new Blob([buffer], { type: "audio/wav" }));
  }

  // ---- Boot ---------------------------------------------------------------
  function fillLanguageSelects() {
    const entries = Object.entries(LANGUAGES).sort((a, b) => a[1].localeCompare(b[1]));
    for (const id of ["#voice-language", "#gen-language"]) {
      const select = $(id);
      select.innerHTML = "";
      for (const [code, name] of entries) select.append(el("option", { value: code, text: `${name} (${code})` }));
      select.value = "en";
    }
  }

  function seedData() {
    const now = Date.now();
    state.voices = [
      { id: newId(), name: "My voice", language: "en", duration: 11, tone: 210 },
      { id: newId(), name: "Narrator", language: "fr", duration: 14, tone: 160 },
      { id: newId(), name: "Abuela", language: "es", duration: 9, tone: 240 },
    ];
    state.selectedVoice = state.voices[0].id;

    const v = state.voices;
    const sample = (voice, text, language, ageMin, dur, params) => {
      const id = newId();
      const job = {
        id, voice_id: voice.id, voice_name: voice.name, text, language,
        params: { exaggeration: 0.5, cfg_weight: 0.5, temperature: 0.8, seed: null, ...params },
        status: "done", created_at: now - ageMin * 60000, duration: dur,
        audio: toneWav(dur, voice.tone), progress_done: 1, progress_total: 1, has_audio: true,
      };
      state.jobs.set(id, job);
    };
    sample(v[0], "Hello from my own GPU. This whole app runs locally and is served over a Cloudflare tunnel.", "en", 4, 6.2, {});
    sample(v[1], "Bonjour à toutes et à tous. Ceci est une démonstration de clonage de voix multilingue.", "fr", 22, 7.1, { exaggeration: 0.7, cfg_weight: 0.3 });
  }

  // ---- Voices -------------------------------------------------------------
  let previewAudio = null;
  function togglePreview(button, voice) {
    if (previewAudio && !previewAudio.paused) {
      previewAudio.pause();
      document.querySelectorAll(".voice-item .play").forEach((b) => (b.textContent = "▶"));
      if (previewAudio.dataset && previewAudio.dataset.id === voice.id) return;
    }
    previewAudio = new Audio(voice.tone ? toneWav(Math.min(voice.duration, 4), voice.tone) : undefined);
    previewAudio.dataset.id = voice.id;
    previewAudio.addEventListener("ended", () => (button.textContent = "▶"));
    previewAudio.play().catch(() => toast("Preview needs a click first", "error"));
    button.textContent = "■";
  }

  function renderVoices() {
    const list = $("#voice-list");
    list.innerHTML = "";
    $("#voice-empty").classList.toggle("hidden", state.voices.length > 0);
    for (const voice of state.voices) {
      const meta = [voice.language ? langName(voice.language) : null, fmtSeconds(voice.duration)].filter(Boolean).join(" · ");
      list.append(el("li", { class: `voice-item${voice.id === state.selectedVoice ? " selected" : ""}`, onclick: () => selectVoice(voice.id) }, [
        el("button", { class: "icon-btn play btn", type: "button", title: "Play reference", text: "▶",
          onclick: (e) => { e.stopPropagation(); togglePreview(e.currentTarget, voice); } }),
        el("div", { class: "info" }, [el("div", { class: "name", text: voice.name }), el("div", { class: "meta", text: meta })]),
        el("button", { class: "icon-btn btn", type: "button", title: "Rename", text: "✎",
          onclick: (e) => {
            e.stopPropagation();
            const name = prompt("Voice name", voice.name);
            if (name && name.trim()) { voice.name = name.trim().slice(0, 120); renderVoices(); }
          } }),
        el("button", { class: "icon-btn btn", type: "button", title: "Delete voice", text: "🗑",
          onclick: (e) => {
            e.stopPropagation();
            if (!confirm(`Delete voice "${voice.name}"? (Demo only.)`)) return;
            state.voices = state.voices.filter((x) => x.id !== voice.id);
            if (state.selectedVoice === voice.id) state.selectedVoice = state.voices[0]?.id ?? null;
            renderVoices();
            toast("Voice deleted");
          } }),
      ]));
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
    if (voice?.language) $("#gen-language").value = voice.language;
    renderVoices();
  }

  // ---- Add a voice --------------------------------------------------------
  function updateSaveButton() {
    const tab = document.querySelector(".tab.active")?.dataset.tab;
    const hasInput = tab === "record" ? !!state.recordedTone : ($("#voice-file").files?.length ?? 0) > 0;
    $("#voice-save").disabled = !hasInput;
  }

  function toggleRecording() {
    const button = $("#record-btn");
    if (state.recording) { stopRecording(); return; }
    state.recording = true;
    state.recordedTone = null;
    $("#record-preview").classList.add("hidden");
    const started = Date.now();
    button.textContent = "■ Stop";
    button.classList.add("recording");
    state.recordTimer = setInterval(() => {
      const elapsed = (Date.now() - started) / 1000;
      $("#record-timer").textContent = fmtSeconds(elapsed);
      if (elapsed >= LIMITS.max_reference_seconds) stopRecording();
    }, 200);
  }

  function stopRecording() {
    if (!state.recording) return;
    state.recording = false;
    clearInterval(state.recordTimer);
    const button = $("#record-btn");
    button.textContent = "● Record";
    button.classList.remove("recording");
    const seconds = Math.max(2, Math.min(LIMITS.max_reference_seconds, parseFloat($("#record-timer").textContent.split(":").reduce((m, s) => m * 60 + +s, 0)) || 6));
    state.recordedTone = { seconds, tone: 190 + Math.floor(Math.random() * 80) };
    const preview = $("#record-preview");
    preview.src = toneWav(Math.min(seconds, 5), state.recordedTone.tone);
    preview.classList.remove("hidden");
    if (!$("#voice-name").value) $("#voice-name").focus();
    updateSaveButton();
  }

  function saveVoice(event) {
    event.preventDefault();
    const tab = document.querySelector(".tab.active")?.dataset.tab;
    let duration, tone;
    if (tab === "record") {
      if (!state.recordedTone) return toast("Record something first", "error");
      duration = state.recordedTone.seconds; tone = state.recordedTone.tone;
    } else {
      const file = $("#voice-file").files?.[0];
      if (!file) return toast("Choose a file first", "error");
      duration = 8 + (file.name.length % 8); tone = 170 + (file.name.length * 7) % 90;
    }
    const fallback = tab === "upload" ? ($("#voice-file").files[0].name.replace(/\.[^.]+$/, "")) : "Voice";
    const voice = {
      id: newId(),
      name: ($("#voice-name").value.trim() || fallback).slice(0, 120),
      language: $("#voice-language").value || null,
      duration, tone,
    };
    state.voices.unshift(voice);
    state.selectedVoice = voice.id;
    $("#voice-name").value = "";
    $("#voice-file").value = "";
    state.recordedTone = null;
    $("#record-timer").textContent = "0:00";
    $("#record-preview").classList.add("hidden");
    renderVoices();
    selectVoice(voice.id);
    updateSaveButton();
    toast(`Voice "${voice.name}" saved (${fmtSeconds(duration)})`);
  }

  // ---- Generation (simulated) --------------------------------------------
  function updateCharCount() {
    const len = $("#gen-text").value.length;
    $("#char-count").textContent = len;
    $("#char-count").parentElement.classList.toggle("over", len > LIMITS.max_text_chars);
  }

  function generate(event) {
    event.preventDefault();
    const text = $("#gen-text").value.trim();
    if (!text) return toast("Enter some text first", "error");
    const voice = state.voices.find((v) => v.id === $("#gen-voice").value);
    if (!voice) return toast("Add a voice first", "error");
    const seedRaw = $("#seed").value.trim();
    const job = {
      id: newId(), voice_id: voice.id, voice_name: voice.name, text,
      language: $("#gen-language").value || voice.language || "en",
      params: {
        exaggeration: parseFloat($("#exaggeration").value),
        cfg_weight: parseFloat($("#cfg").value),
        temperature: parseFloat($("#temp").value),
        seed: seedRaw === "" ? null : parseInt(seedRaw, 10),
      },
      status: "queued", created_at: Date.now(), progress_done: 0,
      progress_total: Math.max(2, Math.ceil(text.length / 180)), has_audio: false,
      _voiceTone: voice.tone,
    };
    upsertJob(job, true);
    toast("Queued");
    simulate(job);
  }

  function simulate(job) {
    setTimeout(() => {
      job.status = "running";
      upsertJob(job, false);
      let done = 0;
      const tick = setInterval(() => {
        done += 1;
        job.progress_done = done;
        upsertJob(job, false);
        if (done >= job.progress_total) {
          clearInterval(tick);
          const seconds = Math.max(2, Math.min(30, job.text.length / 14));
          job.status = "done";
          job.duration = seconds;
          job.audio = toneWav(seconds, job._voiceTone || 210);
          job.has_audio = true;
          upsertJob(job, false);
          toast(`Audio ready (${fmtSeconds(seconds)})`);
        }
      }, 550);
    }, 500);
  }

  // ---- Job rendering ------------------------------------------------------
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
    updateJobNode(node, job, previous);
    $("#job-empty").classList.add("hidden");
  }

  function removeJobNode(id) {
    state.jobNodes.get(id)?.remove();
    state.jobNodes.delete(id);
    state.jobs.delete(id);
    $("#job-empty").classList.toggle("hidden", state.jobs.size > 0);
  }

  function describeParams(job) {
    const p = job.params || {};
    const bits = [`exag ${(+p.exaggeration).toFixed(2)}`, `cfg ${(+p.cfg_weight).toFixed(2)}`, `temp ${(+p.temperature).toFixed(2)}`];
    if (p.seed != null) bits.push(`seed ${p.seed}`);
    return bits.join(" · ");
  }

  function buildJobNode(job) {
    const textNode = el("div", { class: "job-text", text: job.text });
    return el("li", { class: "job-item", "data-id": job.id }, [
      el("div", { class: "job-head" }, [
        el("span", { class: "voice", text: job.voice_name }),
        el("span", { class: "badge", text: job.status }),
        el("span", { class: "lang muted small", text: job.language ? langName(job.language) : "" }),
        el("span", { class: "when", text: fmtWhen(job.created_at) }),
      ]),
      textNode,
      el("button", { class: `job-text-toggle${job.text.length > 220 ? "" : " hidden"}`, type: "button", text: "Show more",
        onclick: (e) => { const ex = textNode.classList.toggle("expanded"); e.currentTarget.textContent = ex ? "Show less" : "Show more"; } }),
      el("div", { class: "progress hidden" }, [el("div")]),
      el("div", { class: "job-audio" }),
      el("div", { class: "job-actions" }, [
        el("span", { class: "job-params", text: describeParams(job) }),
        el("span", { class: "spacer" }),
        el("a", { class: "btn small download hidden", download: `${job.voice_name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}-${job.id}.wav`, text: "Download WAV" }),
        el("button", { class: "btn small reuse", type: "button", text: "Reuse text", onclick: () => reuseJob(job.id) }),
        el("button", { class: "btn small remove", type: "button", text: "Delete", onclick: () => deleteJob(job.id) }),
      ]),
    ]);
  }

  function updateJobNode(node, job) {
    const badge = node.querySelector(".badge");
    badge.textContent = job.status;
    badge.className = `badge ${job.status}`;

    const progress = node.querySelector(".progress");
    if (ACTIVE.has(job.status)) {
      progress.classList.remove("hidden");
      const total = job.progress_total || 0;
      if (total > 0 && job.status === "running") {
        progress.classList.remove("indeterminate");
        progress.firstElementChild.style.width = `${Math.round((100 * (job.progress_done || 0)) / total)}%`;
      } else {
        progress.classList.add("indeterminate");
      }
    } else {
      progress.classList.add("hidden");
    }

    const audioBox = node.querySelector(".job-audio");
    if (job.status === "done" && job.has_audio && !audioBox.firstChild) {
      audioBox.append(el("audio", { controls: "", preload: "metadata", src: job.audio }));
    }
    const download = node.querySelector(".download");
    download.classList.toggle("hidden", !(job.status === "done" && job.has_audio));
    if (job.audio) download.href = job.audio;
    node.querySelector(".remove").textContent = ACTIVE.has(job.status) ? "Cancel" : "Delete";
  }

  function reuseJob(id) {
    const job = state.jobs.get(id);
    if (!job) return;
    $("#gen-text").value = job.text;
    updateCharCount();
    if (job.language) $("#gen-language").value = job.language;
    if (state.voices.some((v) => v.id === job.voice_id)) selectVoice(job.voice_id);
    const p = job.params || {};
    setSliders(p);
    $("#seed").value = p.seed ?? "";
    window.scrollTo({ top: 0, behavior: "smooth" });
    $("#gen-text").focus();
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

  function markPreset() {
    document.querySelectorAll(".preset").forEach((btn) => {
      const match = SLIDERS.every(([slider, , , attr]) => Math.abs(Number($(slider).value) - Number(btn.dataset[attr])) < 1e-6);
      btn.classList.toggle("active", match);
    });
  }

  function deleteJob(id) {
    const job = state.jobs.get(id);
    if (!job) return;
    if (ACTIVE.has(job.status)) { job.status = "cancelled"; upsertJob(job, false); toast("Cancelled"); return; }
    if (!confirm("Delete this generation? (Demo only.)")) return;
    removeJobNode(id);
  }

  // ---- Wiring -------------------------------------------------------------
  function boot() {
    $("#login-view").classList.add("hidden");
    $("#app-view").classList.remove("hidden");
    renderVoices();
    // Newest first.
    const jobs = [...state.jobs.values()].sort((a, b) => b.created_at - a.created_at);
    state.jobs.clear();
    for (const job of jobs) upsertJob(job, false);
  }

  document.addEventListener("DOMContentLoaded", () => {
    $("#char-max").textContent = LIMITS.max_text_chars;
    $("#max-ref-seconds").textContent = LIMITS.max_reference_seconds;
    fillLanguageSelects();
    seedData();
    boot();

    $("#demo-banner-close").addEventListener("click", () => $("#demo-banner").remove());
    $("#logout-btn").addEventListener("click", () => {
      $("#app-view").classList.add("hidden");
      $("#login-view").classList.remove("hidden");
      setTimeout(() => $("#login-password").focus(), 0);
    });
    $("#login-form").addEventListener("submit", (e) => { e.preventDefault(); $("#login-password").value = ""; boot(); });
    $("#record-btn").addEventListener("click", toggleRecording);
    $("#voice-form").addEventListener("submit", saveVoice);
    $("#voice-file").addEventListener("change", updateSaveButton);
    $("#generate-form").addEventListener("submit", generate);
    $("#gen-text").addEventListener("input", updateCharCount);
    $("#gen-voice").addEventListener("change", (e) => selectVoice(e.target.value));
    $("#refresh-jobs").addEventListener("click", () => toast("This is a static demo — nothing to refresh"));

    for (const [slider, output] of SLIDERS) {
      $(slider).addEventListener("input", (e) => {
        $(output).textContent = Number(e.target.value).toFixed(2);
        markPreset();
      });
    }
    document.querySelectorAll(".preset").forEach((btn) => btn.addEventListener("click", () => applyPreset(btn)));
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
  });
})();
