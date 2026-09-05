# Legendary Feather — Real-Time Voice Translation (with a fully offline mode)

A consumer real-time voice translation app: speak, and the other person hears you in their language,
face to face. It runs in two modes — an **online** mode that cascades cloud and self-hosted engines
for best quality, and an **offline** mode that runs the **entire speech→translation→speech pipeline
inside the browser**, with no server round-trip and no data leaving the device.

The offline mode is the interesting part. Getting Whisper-class speech recognition, neural machine
translation, and neural TTS to all run **client-side, in WebAssembly, fast enough to feel live** is
where most of the hard engineering went — including hand-patching a broken quantized ONNX model to
make it load and run at all.

> Online: ~99 languages. Offline: ~30+ languages fully on-device. Built and deployed solo.

---

## Two modes, one app

**Online mode** — a fault-tolerant cascade of engines rather than a single API. A self-hosted
CTranslate2 + Opus/Argos engine (free, on-box, int8 CPU) is the primary translator; cloud services
(OpenAI, DeepL, ElevenLabs, Gemini) fill quality gaps and the rare long tail. Real-time transport is
Flask-SocketIO over gevent websockets. Billing (Stripe, test/live toggle), JWT auth, and Postgres
back the product side.

**Offline mode** — the whole pipeline runs in the browser, nothing leaves the device:

```
   Mic ──▶  STT (in-browser)      ──▶  MT (in-browser WASM)  ──▶  TTS (in-browser)   ──▶  Speaker
            Moonshine ONNX             Bergamot                    Piper / Kokoro
            (transformers.js)          (Marian NMT, WASM)          (VITS, onnxruntime-web)
            Whisper-base fallback
```

Everything is client-side: ONNX models via `transformers.js` and `onnxruntime-web`, Marian NMT
compiled to WebAssembly (Bergamot), and VITS neural voices (Piper / Kokoro). No inference server, no
network dependency, no conversation ever transmitted. That last property is the whole point of the
offline mode — it works on a plane, in a foreign country with no data plan, and it's private by
construction.

---

## Engineering deep-dive: making a broken quantized ONNX model actually run

The offline Spanish speech-recognition path uses Moonshine (a fast, small ASR model). The quantized
build I needed had two independent problems:

**1. It was 35× too slow in the naive path.** The full-precision (fp32) decoder ran at ~13.8s per
utterance in-browser — unusable. The quantized decoder existed but wouldn't load.

**2. The quantized decoder was malformed.** Its `decoder_model_merged` graph is an ONNX `If` node
with two branches (a KV-cache-priming branch and a reuse branch). Standard ONNX quantizers **don't
descend into `If` subgraphs**, so the `scale` and `zero_point` initializers for the token-embedding
matrix lived only at the top level — not inside the branches that actually consume them.
`onnxruntime` rejected the model outright with `Missing required scale`.

**3. It also came out in English.** transformers.js reads `is_multilingual` from the model config;
left true, generation defaulted to English instead of Spanish.

**The fix** (see [`empaquetar_moonshine_es.py`](../highlights/empaquetar_moonshine_es.py)) loads the graph with the
`onnx` library, finds the `If` node, and for each branch subgraph copies the missing
`*_scale` / `*_zero_point` initializers **into the subgraph** where they're referenced — without
touching a single weight. It then forces `is_multilingual: false` in the config, verifies the patched
decoder loads in an `onnxruntime` session, and repackages everything for `transformers.js`.

**Result:** in-browser Spanish transcription dropped from ~13.8s to **~250–800ms** (roughly
20–50× faster), now correctly **in Spanish**, in a ~21 MB encoder + ~43 MB decoder package that loads
in any ONNX runtime. This is the kind of low-level model-plumbing fix that isn't in any tutorial —
it required reading the ONNX graph structure and understanding how quantization interacts with
control-flow nodes.

---

## Layered voice architecture

Offline voice coverage is organized as layers, so the app always has *some* voice for a language and
upgrades to a better one when available:

- **Layer 1** — hand-selected high-quality voices for the top languages.
- **Layer 2** — Piper (VITS) in-browser, ~27 languages, self-hosted WASM phonemizer (espeak-ng) +
  `onnxruntime-web`.
- **Layer 3 (online tail)** — Gemini TTS for the rare long-tail languages that have no offline voice.

Each layer is a graceful fallback for the one before it — the same resilience philosophy as the
online cascade.

---

## Tech stack

| Area | Choice |
|---|---|
| Backend | Python, Flask, Flask-SocketIO (gevent websockets), gunicorn |
| Auth / data | JWT (python-jose, bcrypt), PostgreSQL / SQLAlchemy |
| Billing | Stripe (test/live mode toggle) |
| Online translation | CTranslate2 + Opus/Argos (primary), OpenAI / DeepL / Gemini (fallback) |
| Online voice | ElevenLabs, Edge TTS, Gemini TTS |
| Offline STT | Moonshine + Whisper (ONNX, transformers.js / onnxruntime-web) |
| Offline MT | Bergamot (Marian NMT compiled to WebAssembly) |
| Offline TTS | Piper, Kokoro (VITS, onnxruntime-web) |
| Infra | Docker, Railway, PostHog analytics |

---

## What this project demonstrates

- Shipping and operating a **real product** end to end, solo — auth, billing, analytics, deploys.
- Running a **full ML inference pipeline client-side in WebAssembly** — a genuinely hard target.
- **Low-level ONNX / quantization debugging**: reading model graphs and patching control-flow
  subgraphs, not just calling an API.
- **Resilient system design**: layered fallbacks so no single engine or vendor is a point of failure.

---

_Part of Legendary Feather, a real-time translation product suite (this consumer app + venue stations)._
