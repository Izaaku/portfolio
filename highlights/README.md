# Code highlights

Representative files from two private production projects, included so the engineering
is visible without exposing the full products.

- **cascade.py** — the fault-tolerant voice-to-voice translation cascade (STT → local
  translation → tiered TTS) that replaced Google's Gemini Live in Polyglot Talk Tour.
- **empaquetar_moonshine_es.py** — repairs a broken *quantized* ONNX decoder by copying the
  quantization `scale`/`zero_point` initializers into an `If`-node's subgraphs (where the
  standard quantizer never wrote them), then repackages it for in-browser use.
