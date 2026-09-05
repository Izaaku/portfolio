"""Cascada de traducción voz→voz para la estación del Tour.

REEMPLAZA a Gemini Live. Recibe audio de una frase, y en un solo request:
  1. Transcribe    — OpenAI Whisper (whisper-1)
  2. Traduce       — LOCAL Opus/CTranslate2 (gratis, sin nube) → MyMemory
  3. Sintetiza voz — Edge TTS (neural) → Google TTS → Gemini TTS (cola rara)

Devuelve la traducción COMPLETA de un jalón (texto + audio MP3 base64). Sin
streaming que adivinar → nunca corta la última palabra (el bug de Gemini Live).

Env: OPENAI_API_KEY, GEMINI_API_KEY (para la cola rara de voz).
Requiere: openai, ctranslate2, sentencepiece, langdetect, edge-tts, pydub
          (+ ffmpeg en el sistema, para el PCM→MP3 de Gemini TTS).
Todo es tolerante a fallos: si un motor no está o falla, cae al siguiente.
"""
import os
import io
import re
import time
import base64
import tempfile
import subprocess
import logging

log = logging.getLogger(__name__)

# ───────────────────────── STT — OpenAI Whisper ─────────────────────────
_WHISPER_LANG_TO_ISO = {
    'english': 'en', 'spanish': 'es', 'castilian': 'es', 'french': 'fr',
    'german': 'de', 'italian': 'it', 'portuguese': 'pt', 'dutch': 'nl',
    'russian': 'ru', 'chinese': 'zh', 'mandarin': 'zh', 'japanese': 'ja',
    'korean': 'ko', 'arabic': 'ar', 'hindi': 'hi', 'turkish': 'tr',
    'polish': 'pl', 'ukrainian': 'uk', 'greek': 'el', 'swedish': 'sv',
    'danish': 'da', 'norwegian': 'no', 'finnish': 'fi', 'czech': 'cs',
    'romanian': 'ro', 'hungarian': 'hu', 'thai': 'th', 'vietnamese': 'vi',
    'indonesian': 'id', 'hebrew': 'he', 'bulgarian': 'bg', 'catalan': 'ca',
}


def _norm_lang(lang):
    if not lang:
        return ''
    s = str(lang).strip().lower()
    if s in _WHISPER_LANG_TO_ISO:
        return _WHISPER_LANG_TO_ISO[s]
    return s if len(s) <= 3 else s[:2]


_openai_client = None


def _openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', ''))
    return _openai_client


def transcribe(audio_bytes, language=None):
    """audio → {text, lang, duration}. Whisper con hint de idioma opcional."""
    if not os.getenv('OPENAI_API_KEY', '') or not audio_bytes:
        return {'text': '', 'lang': '', 'duration': 0.0}
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
            f.write(audio_bytes)
            tmp = f.name
        params = {'model': 'whisper-1', 'file': open(tmp, 'rb'),
                  'response_format': 'verbose_json'}
        if language and language != 'auto':
            params['language'] = (language or '').split('-')[0].lower()[:2]
        r = _openai().audio.transcriptions.create(**params)
        return {
            'text': (r.text or '').strip(),
            'lang': _norm_lang(getattr(r, 'language', language or '')),
            'duration': float(getattr(r, 'duration', 0) or 0),
        }
    except Exception as e:
        log.error('[cascade] whisper: %s', str(e)[:160])
        return {'text': '', 'lang': '', 'duration': 0.0}
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


# ─────────────────────── MT — Local Opus/CTranslate2 → MyMemory ──────────
# Motor LOCAL (el mismo de LF: CTranslate2 + SentencePiece + modelos Opus/Argos).
# 100% en el servidor, gratis, rápido (int8 CPU), sin nube. Baja el modelo de
# cada par una vez y lo cachea en MT_DIR (conviene un Volume en Fly para que
# sobreviva a los deploys). Respaldo gratis: MyMemory.

_local_mt = None


def _local():
    global _local_mt
    if _local_mt is None:
        try:
            from app.services.local_translation import LocalTranslationService
            _local_mt = LocalTranslationService()
        except Exception as e:
            log.warning('[cascade] local mt init: %s', str(e)[:120])
            _local_mt = False
    return _local_mt


def _mymemory(text, src, tgt):
    try:
        import requests
        import urllib.parse
        src = (src or 'auto').split('-')[0].lower()
        tgt = (tgt or 'en').split('-')[0].lower()
        pair = (('autodetect' if src in ('', 'auto') else src) + '|' + tgt)
        q = urllib.parse.urlencode({'q': text, 'langpair': pair})
        r = requests.get('https://api.mymemory.translated.net/get?' + q, timeout=8)
        j = r.json()
        return ((j.get('responseData') or {}).get('translatedText') or '').strip()
    except Exception as e:
        log.error('[cascade] mymemory: %s', str(e)[:120])
        return ''


def translate(text, src, tgt):
    text = (text or '').strip()
    if not text:
        return ''
    ps = (src or '').split('-')[0].lower()
    pt = (tgt or '').split('-')[0].lower()
    if ps and pt and ps == pt:
        return text
    # Motor LOCAL (Opus/CTranslate2) primario — gratis, rápido, sin nube.
    lm = _local()
    if lm:
        try:
            r = lm.translate(text, ps or 'auto', pt)
            if r is not None and r.strip():
                return r.strip()
        except Exception as e:
            log.warning('[cascade] local mt → mymemory: %s', str(e)[:120])
    # Respaldo gratis: MyMemory (par no disponible en local, o falló).
    return _mymemory(text, ps or 'auto', pt) or text


# ───────────────────────── TTS — Edge → Google → Gemini ─────────────────
_EDGE_VOICE = {
    'en': 'en-US-AriaNeural', 'es': 'es-MX-DaliaNeural', 'fr': 'fr-FR-DeniseNeural',
    'it': 'it-IT-ElsaNeural', 'pt': 'pt-BR-FranciscaNeural', 'de': 'de-DE-KatjaNeural',
    'nl': 'nl-NL-ColetteNeural', 'ru': 'ru-RU-SvetlanaNeural', 'zh': 'zh-CN-XiaoxiaoNeural',
    'ja': 'ja-JP-NanamiNeural', 'ko': 'ko-KR-SunHiNeural', 'ar': 'ar-SA-ZariyahNeural',
    'hi': 'hi-IN-SwaraNeural', 'tr': 'tr-TR-EmelNeural', 'pl': 'pl-PL-ZofiaNeural',
    'uk': 'uk-UA-PolinaNeural', 'el': 'el-GR-AthinaNeural', 'sv': 'sv-SE-SofieNeural',
    'da': 'da-DK-ChristelNeural', 'no': 'nb-NO-PernilleNeural', 'fi': 'fi-FI-NooraNeural',
    'cs': 'cs-CZ-VlastaNeural', 'ro': 'ro-RO-AlinaNeural', 'hu': 'hu-HU-NoemiNeural',
    'th': 'th-TH-PremwadeeNeural', 'vi': 'vi-VN-HoaiMyNeural', 'id': 'id-ID-GadisNeural',
    'he': 'he-IL-HilaNeural', 'bg': 'bg-BG-KalinaNeural',
}

# Voces MASCULINAS (paralelo a _EDGE_VOICE, que son femeninas).
_EDGE_VOICE_M = {
    'en': 'en-US-GuyNeural', 'es': 'es-MX-JorgeNeural', 'fr': 'fr-FR-HenriNeural',
    'it': 'it-IT-DiegoNeural', 'pt': 'pt-BR-AntonioNeural', 'de': 'de-DE-ConradNeural',
    'nl': 'nl-NL-MaartenNeural', 'ru': 'ru-RU-DmitryNeural', 'zh': 'zh-CN-YunxiNeural',
    'ja': 'ja-JP-KeitaNeural', 'ko': 'ko-KR-InJoonNeural', 'ar': 'ar-SA-HamedNeural',
    'hi': 'hi-IN-MadhurNeural', 'tr': 'tr-TR-AhmetNeural', 'pl': 'pl-PL-MarekNeural',
    'uk': 'uk-UA-OstapNeural', 'el': 'el-GR-NestorasNeural', 'sv': 'sv-SE-MattiasNeural',
    'da': 'da-DK-JeppeNeural', 'no': 'nb-NO-FinnNeural', 'fi': 'fi-FI-HarriNeural',
    'cs': 'cs-CZ-AntoninNeural', 'ro': 'ro-RO-EmilNeural', 'hu': 'hu-HU-TamasNeural',
    'th': 'th-TH-NiwatNeural', 'vi': 'vi-VN-NamMinhNeural', 'id': 'id-ID-ArdiNeural',
    'he': 'he-IL-AvriNeural', 'bg': 'bg-BG-BorislavNeural',
}


def _edge_tts(text, lang, gender='female'):
    """Voz neural (Microsoft Edge TTS, GRATIS) via la API de Python — sin subprocess,
    sin depender del PATH del contenedor. Devuelve base64 MP3 o ''. edge-tts entrega
    directamente MP3, así que el formato ya empata con el cliente (audio/mpeg)."""
    try:
        import asyncio
        import edge_tts
        pm = (lang or 'en').split('-')[0].lower()
        voice = (_EDGE_VOICE_M.get(pm) if gender == 'male' else _EDGE_VOICE.get(pm)) or _EDGE_VOICE.get(pm)
        text = (text or '').strip()
        if not voice or not text:
            return ''

        async def _gen():
            buf = b''
            communicate = edge_tts.Communicate(text, voice)
            async for chunk in communicate.stream():
                if chunk.get('type') == 'audio' and chunk.get('data'):
                    buf += chunk['data']
            return buf

        # Event loop propio por request → seguro en los threads de gunicorn (gthread).
        # Timeout de 6s: si Edge no responde (red de Fly a Microsoft), corta y cae a Google
        # en vez de colgarse. Edge sano responde en ~1s.
        loop = asyncio.new_event_loop()
        try:
            audio = loop.run_until_complete(asyncio.wait_for(_gen(), timeout=6))
        finally:
            try:
                loop.close()
            except Exception:
                pass
        return base64.b64encode(audio).decode('ascii') if audio else ''
    except Exception as e:
        log.warning('[cascade] edge_tts: %s', str(e)[:140])
        return ''


def _free_tts(text, lang):
    """Voz gratis (Google Translate TTS, no oficial). base64 MP3 o ''."""
    try:
        import urllib.request
        import urllib.parse
        lang = (lang or 'en').split('-')[0].lower()
        text = (text or '').strip()
        if not text:
            return ''
        chunks, rest = [], text
        while rest:
            if len(rest) <= 190:
                chunks.append(rest)
                break
            cut = rest.rfind(' ', 0, 190)
            if cut <= 0:
                cut = 190
            chunks.append(rest[:cut])
            rest = rest[cut:].lstrip()
        audio = b''
        for i, ch in enumerate(chunks):
            q = urllib.parse.urlencode({'ie': 'UTF-8', 'tl': lang, 'client': 'tw-ob',
                                        'q': ch, 'total': len(chunks), 'idx': i, 'textlen': len(ch)})
            req = urllib.request.Request('https://translate.googleapis.com/translate_tts?' + q,
                                         headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=8) as r:
                audio += r.read()
        return base64.b64encode(audio).decode('ascii') if audio else ''
    except Exception as e:
        log.warning('[cascade] free_tts: %s', str(e)[:120])
        return ''


def _gemini_tts(text, lang, gender='female'):
    """Capa 3 — voz de nube para la cola rara (~99 idiomas). base64 MP3 o ''."""
    try:
        import requests
        from pydub import AudioSegment
        key = os.getenv('GEMINI_API_KEY', '')
        text = (text or '').strip()
        if not key or not text:
            return ''
        if len(text) > 900:
            text = text[:900]
        model = os.getenv('GEMINI_TTS_MODEL', 'gemini-2.5-flash-preview-tts')
        voice = 'Puck' if gender == 'male' else 'Kore'   # Puck = masculina, Kore = femenina
        url = 'https://generativelanguage.googleapis.com/v1beta/models/' + model + ':generateContent'
        body = {'contents': [{'parts': [{'text': text}]}],
                'generationConfig': {'responseModalities': ['AUDIO'],
                                     'speechConfig': {'voiceConfig': {'prebuiltVoiceConfig': {'voiceName': voice}}}}}
        r = requests.post(url, headers={'x-goog-api-key': key, 'Content-Type': 'application/json'},
                          json=body, timeout=15)
        if r.status_code != 200:
            log.warning('[cascade] gemini_tts HTTP %s %s', r.status_code, r.text[:140])
            return ''
        part = r.json()['candidates'][0]['content']['parts'][0]
        inline = part.get('inlineData') or part.get('inline_data') or {}
        b64pcm = inline.get('data')
        if not b64pcm:
            return ''
        mt = inline.get('mimeType') or inline.get('mime_type') or ''
        m = re.search(r'rate=(\d+)', mt)
        rate = int(m.group(1)) if m else 24000
        seg = AudioSegment(data=base64.b64decode(b64pcm), sample_width=2, frame_rate=rate, channels=1)
        buf = io.BytesIO()
        seg.export(buf, format='mp3', bitrate='64k')
        out = buf.getvalue()
        return base64.b64encode(out).decode('ascii') if out else ''
    except Exception as e:
        log.warning('[cascade] gemini_tts: %s', str(e)[:140])
        return ''


def synthesize(text, lang, gender='female'):
    """Voz en cascada: Edge (neural) → Google → Gemini (cola rara). base64 MP3.
    Género aplica a Edge y Gemini; Google TTS tiene voz única por idioma."""
    return _edge_tts(text, lang, gender) or _free_tts(text, lang) or _gemini_tts(text, lang, gender) or ''


# ───────────────────────── Orquestador ──────────────────────────────────
def _finish(heard, src_lang, target_lang, duration, t0, gender='female'):
    heard = (heard or '').strip()
    if not heard:
        return {'heard': '', 'translated': '', 'audio': '',
                'src': src_lang, 'target': target_lang,
                'duration': duration, 'ms': int((time.time() - t0) * 1000)}
    translated = translate(heard, src_lang, target_lang)
    audio = synthesize(translated, target_lang, gender)
    return {
        'heard': heard, 'translated': translated, 'audio': audio,
        'src': src_lang, 'target': target_lang,
        'duration': duration, 'ms': int((time.time() - t0) * 1000),
    }


def run_from_text(text, src_lang, target_lang, gender='female'):
    """El navegador YA transcribió (SpeechRecognition, gratis). Solo traducir + voz."""
    return _finish(text, src_lang, target_lang, 0.0, time.time(), gender)


def run_from_audio(audio_bytes, src_lang, target_lang, gender='female'):
    """RESPALDO: sin SpeechRecognition en el navegador → Whisper transcribe, luego traducir + voz."""
    t0 = time.time()
    stt = transcribe(audio_bytes, src_lang)
    return _finish(stt['text'], src_lang, target_lang, stt['duration'], t0, gender)


# compat
run = run_from_audio


# ───────────────────────── Pre-descarga de modelos (arranque) ────────────
def prewarm(pairs=None):
    """Baja EN SEGUNDO PLANO los modelos Opus de los pares comunes, para que el
    primer huésped no espere la descarga. Solo descarga a disco (no los carga en
    RAM), así no infla la memoria. Configurable por env:
        PREWARM_STAFF_LANG   (default 'es')
        PREWARM_TOURIST_LANGS (default 'en,fr,de,it,pt,zh,ja')
    """
    import threading

    if pairs is None:
        staff = (os.getenv('PREWARM_STAFF_LANG', 'es') or 'es').split('-')[0].lower()
        tourists = (os.getenv('PREWARM_TOURIST_LANGS', 'en,fr,de,it,pt,zh,ja') or '').split(',')
        pairs = []
        for t in tourists:
            t = (t or '').strip().split('-')[0].lower()
            if t and t != staff:
                pairs.append((staff, t))
                pairs.append((t, staff))

    def _run():
        try:
            lm = _local()
            if not lm:
                return
            for (a, b) in pairs:
                try:
                    lm._ensure_dir(a, b)   # descarga + extrae a disco si falta (sin cargar en RAM)
                    log.info('[cascade] prewarm listo: %s->%s', a, b)
                except Exception as e:
                    log.warning('[cascade] prewarm %s->%s: %s', a, b, str(e)[:100])
            log.info('[cascade] prewarm terminado (%d pares)', len(pairs))
        except Exception as e:
            log.warning('[cascade] prewarm: %s', str(e)[:120])

    threading.Thread(target=_run, daemon=True, name='mt-prewarm').start()
