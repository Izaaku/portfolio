# Isaak Uribe — Engineering Portfolio

Software developer focused on real-time, ML-powered systems. Below are three products I designed,
built, and shipped end to end — two production web apps and a published Chrome extension —
including the hard parts: fault-tolerant service design, running full ML pipelines in the browser,
and low-level ONNX model debugging.

> Bilingual (C1 English) · PCAP-certified (Python Institute) · Querétaro, Mexico · Open to remote
> · [github.com/Izaaku](https://github.com/Izaaku)

---

## 🪶 Legendary Feather — Real-Time Voice Translation App

A consumer real-time translation app with an **offline mode that runs the entire
speech → translation → speech pipeline inside the browser** (WebAssembly, no server, private by
design), plus an online cascade of cloud and self-hosted engines.

**Highlight:** I debugged and patched a malformed **quantized ONNX model at the graph level**,
cutting in-browser Spanish speech recognition from ~13.8s to ~250–800ms (≈20–50× faster).

→ **[Read the case study](docs/legendary-feather.md)**

---

## 🗣️ Polyglot Talk Tour — Venue Translation Station

A voice-to-voice translation station for hotels and venues. I **replaced a managed Google
(Gemini Live) integration with a custom, fault-tolerant cascade** — browser speech recognition +
a self-hosted CTranslate2 engine + tiered neural TTS — reaching ~1s end-to-end at near-zero
marginal cost, and fixing a bug the managed product had.

→ **[Read the case study](docs/polyglot-talk-tour.md)**

---

## 🧩 Polyglot Desk — AI Writing Assistant (Chrome Extension)

A **Chrome extension published on the Chrome Web Store** — an AI writing assistant for support
agents: grammar and punctuation fixes, rewrite, hover-to-translate, voice dictation, and
predictive autocomplete across **30+ languages**, embedded directly into any browser text field
(Manifest V3, no tab-switching).

→ **[View on the Chrome Web Store](https://chromewebstore.google.com/detail/polyglot-desk/iahcimnkfmlhndlgpnfngnfecklajlnb)**

---

## Code highlights

A few representative files from these projects (the full products are private):

- [`highlights/cascade.py`](highlights/cascade.py) — the fault-tolerant voice-to-voice cascade
  (STT → local translation → tiered TTS) that replaced Gemini Live.
- [`highlights/empaquetar_moonshine_es.py`](highlights/empaquetar_moonshine_es.py) — the ONNX
  packaging/repair script that patches quantization scales into an `If`-node's subgraphs so a
  broken quantized decoder loads and runs.

---

## Tech I work with

**Languages:** Python, JavaScript, SQL, HTML, CSS
**Backend:** Flask, Flask-SocketIO / WebSockets, REST APIs, SQLAlchemy, PostgreSQL, gunicorn
**Browser:** Chrome Extensions (Manifest V3), WebAssembly, transformers.js
**ML / Inference:** ONNX / onnxruntime, CTranslate2, transformers.js, WebAssembly inference, Whisper
**Infra:** Docker, CI/CD (GitHub Actions), Fly.io, Railway, Git, Stripe

---

*Contact: izaaku16@gmail.com*
