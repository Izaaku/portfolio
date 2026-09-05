# Polyglot Talk Tour — Real-Time Voice Translation Station

A production, voice-to-voice translation system that lets a hotel or venue and an international
guest speak to each other in their own languages. A guest scans a QR code, taps a language, and
talks; staff hear the translation spoken back within about a second, and vice-versa.

**The engineering story:** this station originally ran on Google's Gemini Live Translate.
I replaced it with a **custom, fault-tolerant translation cascade** I built from independent,
mostly-free components. The result is faster, cheaper to run, keeps guest conversations off
Google's servers, and fixed a bug the managed product couldn't: Gemini Live's streaming would
sometimes clip the last word of a sentence.

> Live in production. ~1s end-to-end for common language pairs. ~30 languages with neural voices,
> ~99 with cloud voice fallback.

---

## Why this is interesting

Real-time translation is a "solved" problem if you're Google. The interesting engineering is doing
it **without** a single managed black-box API — as a resilient pipeline where every stage has a
free primary path and graceful fallbacks, so no one vendor outage or rate limit takes the station
down, and the common case costs essentially nothing to run.

- **Replaced a Google product** (Gemini Live Translate) with an in-house cascade — and improved on it.
- **Fixed the "clipped last word" bug** by design: the cascade returns each fully-formed sentence in
  one shot instead of streaming partial audio it has to guess the end of.
- **Runs the common path for free**: browser speech recognition + a self-hosted CPU translation
  engine + free neural TTS. Paid/cloud services are only touched for the rare long tail.
- **Privacy as an architectural property**, not a marketing line: the primary translation engine runs
  entirely on our own server. Guest conversations don't transit a third-party translation cloud.

---

## Architecture

Each spoken phrase is one request. The pipeline is three stages, and **every stage degrades gracefully**:

```
                 ┌─────────────────────────────────────────────────────────────┐
   Guest / Staff │  1. SPEECH → TEXT   2. TRANSLATE        3. TEXT → SPEECH      │
   speaks  ─────▶│                                                              │─────▶  translated
                 │  Browser SR  ┐      Local Opus/CT2 ┐    Edge TTS (neural) ┐   │        voice + text
                 │  (free)      │      (free, on-box) │    (free)            │   │
                 │  Whisper ────┘      MyMemory ──────┘    Google TTS ───────┤   │
                 │  (fallback,                             Gemini TTS ───────┘   │
                 │   no browser SR)                        (rare long tail)      │
                 └─────────────────────────────────────────────────────────────┘
```

**Stage 1 — Speech to text.** The browser's native `SpeechRecognition` does transcription for free
on devices that support it. Devices that don't (iOS Safari) fall back to OpenAI Whisper server-side.
The client sends already-transcribed text on the fast path, and only ships audio when it has to.

**Stage 2 — Translation.** A **self-hosted CTranslate2 + SentencePiece engine** running Opus/Argos
models on CPU (int8) is the primary. Models download on first use and are cached to a persistent
volume; an English-pivot handles pairs without a direct model. If the local engine can't serve a
pair, it falls back to the free MyMemory API, and finally to returning the source text rather than
failing.

**Stage 3 — Text to speech.** Microsoft Edge's neural voices (free, via the `edge-tts` Python API)
are primary, with a per-language male/female voice map. If Edge doesn't answer in time, it falls to
Google Translate TTS, and for the rare long-tail language to Gemini TTS. Voice gender is selectable
at the station.

### Reliability & performance details worth calling out

- **Fault tolerance at every layer.** Any single engine can be missing, rate-limited, or throw, and
  the request still completes down the fallback chain. There is no single point of failure.
- **Thread-safe async in a sync server.** Edge TTS is async; the app runs under gunicorn `gthread`.
  Each request gets its own `asyncio` event loop with a hard 6s timeout, so a hung TTS call cuts over
  to the next engine instead of blocking a worker.
- **Cold-start elimination.** A background daemon thread pre-downloads the translation models for the
  common language pairs on boot (download-to-disk only, no RAM cost), so the first guest doesn't pay
  the model-download latency. Configurable via env.
- **Model persistence across deploys.** Models are cached on a Fly.io volume mounted into the
  container, so a redeploy doesn't re-download gigabytes of model weights.

---

## Tech stack

| Layer | Choice |
|---|---|
| Backend | Python, Flask, gunicorn (`gthread`) |
| STT | Browser `SpeechRecognition` (primary), OpenAI Whisper (fallback) |
| Translation | CTranslate2 + SentencePiece (Opus/Argos, int8 CPU), MyMemory (fallback) |
| TTS | Edge TTS neural → Google TTS → Gemini TTS |
| Data / auth | Supabase |
| Infra | Docker, Fly.io (single machine + mounted volume), GitHub Actions CI/CD |

---

## Results

- **~1 second** end-to-end (speak → translated voice) on common pairs, in production.
- **~$0 marginal cost** on the common path — browser STT + on-box translation + free neural TTS.
- **Fixed** the last-word clipping that the previous managed (Gemini Live) implementation exhibited.
- **~30 languages** with gendered neural voices; **~99** reachable via the cloud fallback voice.

---

## Selected code

- [`highlights/cascade.py`](../highlights/cascade.py) — the whole cascade: STT, the local-first
  translate step, the three-tier TTS fallback, the per-request event-loop + timeout, and the boot
  prewarmer.
- `app/services/local_translation.py` (in the private repo) — the self-hosted
  CTranslate2 translation engine with English-pivot.
- `app/routes/tour.py` (in the private repo) — the station endpoint that drives the cascade.

---

_Part of Legendary Feather, a real-time translation product suite (consumer app + venue stations)._
