const VOL_SVG = (extra) => `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>${extra}</svg>`;
const VOL_ON = VOL_SVG('<path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/>');
const VOL_OFF = VOL_SVG('<line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/>');
/* Shared helpers: webcam + prediction loop, local progress storage, gentle sounds, small UI bits. */
(function () {
  const E = window.EMOTIONS;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ------------------------------------------------------------ mobile menu
  const menuBtn = document.getElementById("menu-btn"), nav = document.getElementById("nav");
  menuBtn?.addEventListener("click", () => {
    const open = nav.classList.toggle("open");
    menuBtn.setAttribute("aria-expanded", String(open));
    menuBtn.textContent = open ? "✕" : "☰";
  });

  // Buttons like <button data-click="start"> forward the click to #start (used inside the camera box).
  document.addEventListener("click", e => {
    const t = e.target.closest("[data-click]");
    if (t) document.getElementById(t.dataset.click)?.click();
  });

  // ------------------------------------------------------------ sound (soft chime, can be muted)
  const Sound = {
    get on() { try { return localStorage.getItem("fl_sound") !== "off"; } catch { return true; } },
    set on(v) { try { localStorage.setItem("fl_sound", v ? "on" : "off"); } catch {} },
    ctx: null,
    play(kind) {
      if (!this.on) return;
      try {
        this.ctx = this.ctx || new (window.AudioContext || window.webkitAudioContext)();
        const notes = kind === "success" ? [523.25, 659.25, 783.99] : [392.0, 349.23];
        notes.forEach((f, i) => {
          const o = this.ctx.createOscillator(), g = this.ctx.createGain();
          o.type = "sine"; o.frequency.value = f;
          const t = this.ctx.currentTime + i * 0.13;
          g.gain.setValueAtTime(0.0001, t);
          g.gain.exponentialRampToValueAtTime(0.12, t + 0.02);
          g.gain.exponentialRampToValueAtTime(0.0001, t + 0.35);
          o.connect(g).connect(this.ctx.destination); o.start(t); o.stop(t + 0.4);
        });
      } catch {}
    },
  };
  const toggle = document.getElementById("sound-toggle");
  const paintToggle = () => { toggle.innerHTML = Sound.on ? VOL_ON : VOL_OFF; toggle.setAttribute("aria-pressed", String(Sound.on)); };
  toggle.addEventListener("click", () => { Sound.on = !Sound.on; paintToggle(); });
  paintToggle();

  // ------------------------------------------------------------ progress (browser only)
  const KEY = "fl_progress_v1";
  const Progress = {
    load() {
      try { return JSON.parse(localStorage.getItem(KEY)) || { events: [] }; } catch { return { events: [] }; }
    },
    add(event) {
      const data = this.load();
      data.events.push({ ...event, t: new Date().toISOString() });
      try { localStorage.setItem(KEY, JSON.stringify(data)); } catch {}
    },
    clear() { try { localStorage.removeItem(KEY); } catch {} },
  };

  // ------------------------------------------------------------ model status banner
  const Status = { unsure_below: 0.0 };
  fetch("/api/status").then(r => r.json()).then(s => {
    if (typeof s.unsure_below === "number") Status.unsure_below = s.unsure_below;
    if (!s.model_ready) {
      const b = document.getElementById("model-banner");
      b.firstElementChild.textContent = "Camera activities (Explore, Mirror game, Music) are unavailable because the expression model isn't installed. Everything else works. See the README to enable it.";
      if (s.error) console.warn("Expression model:", s.error);
      b.hidden = false;
    }
  }).catch(() => {});

  // ------------------------------------------------------------ celebration
  function confetti(emojis = ["⭐", "🌟", "✨", "🎉"]) {
    if (reduceMotion) return;
    const box = document.createElement("div");
    box.className = "confetti";
    for (let i = 0; i < 22; i++) {
      const s = document.createElement("span");
      s.textContent = emojis[i % emojis.length];
      s.style.left = `${Math.random() * 100}%`;
      s.style.animationDelay = `${Math.random() * 0.5}s`;
      s.style.fontSize = `${1.2 + Math.random() * 1.2}rem`;
      box.append(s);
    }
    document.body.append(box);
    setTimeout(() => box.remove(), 2600);
  }

  function pop(el) {
    el.classList.remove("pop"); void el.offsetWidth; el.classList.add("pop");
  }

  // ------------------------------------------------------------ webcam + prediction loop
  class EmotionCamera {
    /**
     * @param {HTMLElement} container  element with class "camera" (may contain .cam-empty)
     * @param {object} opts  { intervalMs, onResult(result), drawBoxes }
     */
    constructor(container, opts = {}) {
      this.container = container;
      this.opts = Object.assign({ intervalMs: 600, drawBoxes: true, onResult: () => {} }, opts);
      this.video = document.createElement("video");
      this.video.playsInline = true; this.video.muted = true;
      this.overlay = document.createElement("canvas");
      this.grab = document.createElement("canvas");
      this.guide = Object.assign(document.createElement("div"), { className: "face-guide" });
      this.pill = Object.assign(document.createElement("div"), { className: "cam-pill" });
      this.pill.setAttribute("role", "status");
      this.running = false; this.busy = false; this.stream = null;
    }
    status(text) { this.pill.textContent = text || ""; }
    async start() {
      if (this.running) return;
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser can't use the camera.");
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false });
      } catch (err) {
        const msg = err.name === "NotAllowedError"
          ? "Camera permission was blocked. Click the camera icon in the address bar and choose Allow."
          : err.name === "NotReadableError" ? "The camera is busy. Close other apps that use it (Zoom, Teams, Camera) and try again."
          : err.name === "NotFoundError" ? "No camera was found on this device." : String(err.message || err);
        throw new Error(msg);
      }
      this.video.srcObject = this.stream;
      await this.video.play();
      this.container.querySelector(".cam-empty")?.setAttribute("hidden", "");
      this.container.append(this.video, this.overlay, this.guide, this.pill);
      this.container.classList.add("live", "no-face");
      this.status("Looking for a face…");
      this.running = true;
      this.timer = setInterval(() => this.tick(), this.opts.intervalMs);
    }
    stop() {
      this.running = false;
      clearInterval(this.timer);
      this.stream?.getTracks().forEach(t => t.stop());
      this.stream = null;
      [this.video, this.overlay, this.guide, this.pill].forEach(el => el.remove());
      this.container.classList.remove("live", "no-face");
      this.container.querySelector(".cam-empty")?.removeAttribute("hidden");
    }
    async tick() {
      if (!this.running || this.busy || !this.video.videoWidth) return;
      this.busy = true;
      try {
        const w = 480, h = Math.round(480 * this.video.videoHeight / this.video.videoWidth);
        this.grab.width = w; this.grab.height = h;
        this.grab.getContext("2d").drawImage(this.video, 0, 0, w, h);
        const image = this.grab.toDataURL("image/jpeg", 0.75);
        const res = await fetch("/api/predict", {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || res.statusText);
        if (!this.running) return;
        const found = data.faces.length > 0;
        this.container.classList.toggle("no-face", !found);
        this.status(found ? "" : "No face detected. Face the camera in good light.");
        if (this.opts.drawBoxes) this.draw(data);
        this.opts.onResult(data);
      } catch (err) {
        this.status(String(err.message || err));
        this.opts.onResult({ error: String(err.message || err), faces: [] });
      } finally { this.busy = false; }
    }
    draw(data) {
      const c = this.overlay;
      c.width = data.width; c.height = data.height;
      const ctx = c.getContext("2d");
      ctx.clearRect(0, 0, c.width, c.height);
      data.faces.forEach((f, i) => {
        const { x, y, w, h } = f.box;
        const col = f.unsure ? "#c9d2d9" : (E[f.emotion]?.color || "#7fd1b9");
        ctx.lineWidth = 4;
        ctx.strokeStyle = i === 0 ? col : "rgba(255,255,255,.6)";
        roundRect(ctx, x, y, w, h, 18); ctx.stroke();
        // The canvas is mirrored with CSS; draw the label un-mirrored.
        ctx.save(); ctx.scale(-1, 1);
        ctx.font = "800 20px Nunito, system-ui, sans-serif";
        const label = f.unsure ? "🤔 Not sure" : `${E[f.emotion]?.emoji || ""} ${E[f.emotion]?.name || f.emotion}`;
        const tw = ctx.measureText(label).width + 18;
        const ly = Math.max(0, y - 38);
        ctx.fillStyle = i === 0 ? col : "rgba(31,38,44,.75)";
        roundRect(ctx, -(x + w), ly, tw, 32, 16); ctx.fill();
        ctx.fillStyle = "#15202a";
        if (i !== 0) ctx.fillStyle = "#fff";
        ctx.fillText(label, -(x + w) + 9, ly + 23);
        ctx.restore();
      });
    }
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }

  function renderBars(el, scores) {
    el.innerHTML = Object.keys(E).map(k => {
      const v = scores?.[k] ?? 0;
      return `<div class="bar" style="--c:${E[k].color}"><span>${E[k].emoji}</span><span>${E[k].name}</span>
        <div class="track"><div class="fill" style="width:${(v * 100).toFixed(0)}%"></div></div>
        <span class="v small">${(v * 100).toFixed(0)}%</span></div>`;
    }).join("");
  }

  window.FL = { Sound, Progress, Status, EmotionCamera, renderBars, confetti, pop, EMOTIONS: E, reduceMotion };
})();
