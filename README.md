# Isaak Uribe — Engineering Portfolio

Software developer focused on real-time, ML-powered systems. Below are two production projects
I designed and built end to end — including the hard parts: fault-tolerant service design,
running full ML pipelines in the browser, and low-level ONNX model debugging.

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
**ML / Inference:** ONNX / onnxruntime, CTranslate2, transformers.js, WebAssembly inference, Whisper
**Infra:** Docker, CI/CD (GitHub Actions), Fly.io, Railway, Git, Stripe

---

*Contact: izaak16@live.com.mx*
