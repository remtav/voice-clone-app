/*
 * Contextual onboarding for the Voice Clone UI. Dependency-free and shared
 * verbatim by the real app (app/static) and the static demo (docs/).
 *
 * Pieces:
 *   - A first-visit welcome modal (3 steps), remembered in localStorage and
 *     reopenable from a "Help" button added to the top bar.
 *   - A "?" popover on each section header with contextual tips.
 *   - Friendly empty-state hints in the Voices and Generations panels.
 *
 * It reads no app state and calls no backend, so it is safe in both contexts.
 */
(() => {
  "use strict";

  const STORE_KEY = "vc_onboarded_v1";
  const $ = (s, r = document) => r.querySelector(s);

  const stored = (key) => { try { return localStorage.getItem(key); } catch { return null; } };
  const remember = (key, val) => { try { localStorage.setItem(key, val); } catch { /* private mode */ } };

  const SLIDES = [
    { icon: "🎙️", label: "Step 1 of 3", title: "Add a voice",
      body: "Record 5–15 seconds of clean, single-speaker speech, or upload a clip. Choose the language you're speaking — accents carry across languages." },
    { icon: "⌨️", label: "Step 2 of 3", title: "Write your text",
      body: "Pick a saved voice, then type or paste what you want it to say. Long passages are split into sentence-sized chunks automatically." },
    { icon: "▶️", label: "Step 3 of 3", title: "Generate & download",
      body: "Press <span class='kbd'>Generate</span> (or Ctrl/⌘+Enter), watch the progress, then play or download the WAV. Open <em>Advanced settings</em> to fine-tune expressiveness." },
  ];

  const SECTION_TIPS = {
    "Voices": "Your saved reference voices. Click one to select it for generation. Use ▶ to hear the reference, ✎ to rename, 🗑 to delete.",
    "Add a voice": "Record or upload 5–15s of clean, single-speaker audio in the language you want to clone. Only the first 30 seconds are kept.",
    "Generate speech": "Choose a voice and language, enter text, and Generate. Exaggeration controls emotion; lower the CFG weight for slower, more expressive delivery.",
    "Generations": "Your history. Each result has a player, Download WAV, Reuse text (restores the text and settings), and Delete.",
  };

  const EMPTY_HINTS = {
    "voice-empty": "<strong>No voices yet.</strong> Start below:<ol><li>Record or upload a short clip.</li><li>Name it and pick its language.</li><li>Save — it appears here.</li></ol>",
    "job-empty": "<strong>Nothing generated yet.</strong> Add a voice, type some text in <em>Generate speech</em>, then press Generate.",
  };

  // ---- Welcome modal ------------------------------------------------------
  let modal, dotsWrap, slideBox, backBtn, nextBtn, current = 0;

  function buildModal() {
    modal = document.createElement("div");
    modal.className = "ob-backdrop";
    modal.hidden = true;
    modal.innerHTML = `
      <div class="ob-modal" role="dialog" aria-modal="true" aria-labelledby="ob-slide-title">
        <button class="ob-close" type="button" aria-label="Close">✕</button>
        <div class="ob-slide">
          <div class="ob-slide-icon" aria-hidden="true"></div>
          <div class="ob-step-label"></div>
          <h3 id="ob-slide-title"></h3>
          <p></p>
        </div>
        <div class="ob-dots"></div>
        <div class="ob-actions">
          <button class="btn ob-back" type="button">Back</button>
          <span class="spacer"></span>
          <button class="btn btn-ghost ob-skip" type="button">Skip</button>
          <button class="btn btn-primary ob-next" type="button">Next</button>
        </div>
      </div>`;
    document.body.append(modal);

    slideBox = $(".ob-slide", modal);
    dotsWrap = $(".ob-dots", modal);
    backBtn = $(".ob-back", modal);
    nextBtn = $(".ob-next", modal);

    dotsWrap.innerHTML = SLIDES.map((_, i) => `<button type="button" data-i="${i}" aria-label="Step ${i + 1}"></button>`).join("");
    dotsWrap.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => go(+b.dataset.i)));

    $(".ob-close", modal).addEventListener("click", closeModal);
    $(".ob-skip", modal).addEventListener("click", closeModal);
    backBtn.addEventListener("click", () => go(current - 1));
    nextBtn.addEventListener("click", () => (current >= SLIDES.length - 1 ? closeModal() : go(current + 1)));
    modal.addEventListener("click", (e) => { if (e.target === modal) closeModal(); });
    document.addEventListener("keydown", (e) => {
      if (modal.hidden) return;
      if (e.key === "Escape") closeModal();
      else if (e.key === "ArrowRight" && current < SLIDES.length - 1) go(current + 1);
      else if (e.key === "ArrowLeft" && current > 0) go(current - 1);
    });
  }

  function go(i) {
    current = Math.max(0, Math.min(SLIDES.length - 1, i));
    const s = SLIDES[current];
    $(".ob-slide-icon", modal).textContent = s.icon;
    $(".ob-step-label", modal).textContent = s.label;
    $("#ob-slide-title", modal).textContent = s.title;
    $(".ob-slide p", modal).innerHTML = s.body;
    dotsWrap.querySelectorAll("button").forEach((b, idx) => b.classList.toggle("active", idx === current));
    backBtn.style.visibility = current === 0 ? "hidden" : "visible";
    nextBtn.textContent = current >= SLIDES.length - 1 ? "Got it" : "Next";
  }

  function openModal() { if (!modal) buildModal(); go(0); modal.hidden = false; $(".ob-next", modal).focus(); }
  function closeModal() { if (modal) modal.hidden = true; remember(STORE_KEY, "1"); }

  // ---- Help button --------------------------------------------------------
  function addHelpButton() {
    const bar = $(".topbar");
    if (!bar || $("#ob-help")) return;
    const btn = document.createElement("button");
    btn.id = "ob-help";
    btn.type = "button";
    btn.className = "btn btn-ghost";
    btn.textContent = "Help";
    btn.title = "How to use this app";
    btn.addEventListener("click", openModal);
    const logout = $("#logout-btn", bar);
    if (logout) bar.insertBefore(btn, logout); else bar.append(btn);
  }

  // ---- Section "?" popovers ----------------------------------------------
  let openPopover = null;

  function closePopover() {
    if (openPopover) { openPopover.el.remove(); openPopover.btn.setAttribute("aria-expanded", "false"); openPopover = null; }
  }

  function showPopover(btn, title, body) {
    closePopover();
    const pop = document.createElement("div");
    pop.className = "ob-popover";
    pop.innerHTML = `<h4></h4><p></p>`;
    $("h4", pop).textContent = title;
    $("p", pop).textContent = body;
    document.body.append(pop);
    const r = btn.getBoundingClientRect();
    let left = r.left;
    const maxLeft = window.innerWidth - pop.offsetWidth - 8;
    if (left > maxLeft) { pop.style.setProperty("--arrow-x", `${left - maxLeft + 14}px`); left = maxLeft; }
    pop.style.left = `${Math.max(8, left)}px`;
    pop.style.top = `${r.bottom + 8}px`;
    btn.setAttribute("aria-expanded", "true");
    openPopover = { el: pop, btn };
  }

  function addSectionTips() {
    const scope = $("#app-view") || document;
    scope.querySelectorAll(".card h2").forEach((h2) => {
      const label = h2.textContent.trim();
      const tip = SECTION_TIPS[label];
      if (!tip || h2.querySelector(".ob-tip-btn")) return;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "ob-tip-btn";
      btn.textContent = "?";
      btn.setAttribute("aria-expanded", "false");
      btn.setAttribute("aria-label", `About "${label}"`);
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (openPopover && openPopover.btn === btn) closePopover();
        else showPopover(btn, label, tip);
      });
      h2.append(btn);
    });
    document.addEventListener("click", (e) => {
      if (openPopover && !openPopover.el.contains(e.target) && e.target !== openPopover.btn) closePopover();
    });
    window.addEventListener("resize", closePopover);
    window.addEventListener("scroll", closePopover, true);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closePopover(); });
  }

  // ---- Empty-state hints --------------------------------------------------
  function enrichEmptyStates() {
    for (const [id, html] of Object.entries(EMPTY_HINTS)) {
      const node = document.getElementById(id);
      if (node && !node.classList.contains("ob-empty")) {
        node.classList.remove("muted");
        node.classList.add("ob-empty");
        node.innerHTML = html;
      }
    }
  }

  // ---- First-visit trigger ------------------------------------------------
  function maybeWelcome() {
    if (stored(STORE_KEY)) return;
    const app = $("#app-view");
    if (app && !app.classList.contains("hidden")) { openModal(); return; }
    if (!app) { openModal(); return; }
    // App view is gated (e.g. behind login): wait until it becomes visible.
    const obs = new MutationObserver(() => {
      if (!app.classList.contains("hidden")) {
        obs.disconnect();
        if (!stored(STORE_KEY)) openModal();
      }
    });
    obs.observe(app, { attributes: true, attributeFilter: ["class"] });
  }

  function init() {
    addHelpButton();
    addSectionTips();
    enrichEmptyStates();
    maybeWelcome();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
