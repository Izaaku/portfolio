#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empaqueta Moonshine-ES (base) para transformers.js — RÁPIDO y en ESPAÑOL.

El problema que resuelve:
  - UsefulSensors/moonshine-es ya trae un decoder cuantizado (rápido), PERO le falta
    poner el `scale` del embed_tokens DENTRO de las ramas del nodo `If`, así que
    onnxruntime lo rechaza ("Missing required scale"). Este script copia esos escalares
    al lugar correcto (el peso NO se toca), y el archivo pasa de roto -> carga en cualquier motor.
  - Además fuerza is_multilingual:false para que NO salga en inglés.

Resultado: encoder_model_quantized.onnx (~21MB) + decoder_model_merged_quantized.onnx (~43MB)
           + metadata.  Transcribe español en ~250-800ms en el navegador.

Requisitos:  pip install onnx huggingface_hub
             (onnxruntime es OPCIONAL, solo para la verificación local)
Uso:         edita HF_REPO y HF_TOKEN abajo, luego:  python empaquetar_moonshine_es.py
"""
import os, sys, json, shutil, tempfile

# ======= EDITA ESTO =======
HF_REPO  = "izaaku16/moonshine-base-es-ONNX"   # tu repo (se sobrescribe)
HF_TOKEN = "PON_AQUI_TU_TOKEN_WRITE"           # token con permiso WRITE
SUBIR    = True                                # False = solo arma la carpeta, no sube
# ==========================

WORK = os.path.join(tempfile.gettempdir(), "moonshine_es_pkg")
SRC_REPO = "UsefulSensors/moonshine-es"
META_REPO = "onnx-community/moonshine-base-ONNX"
EMB = "model.decoder.embed_tokens.weight_merged_0"

def log(m): print(f"\n>>> {m}", flush=True)

def main():
    try:
        import onnx
        from huggingface_hub import hf_hub_download, HfApi
    except ImportError as e:
        print("Falta una librería. Corre:  pip install onnx huggingface_hub")
        print("Detalle:", e); sys.exit(1)

    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(os.path.join(WORK, "onnx"), exist_ok=True)
    dl = os.path.join(WORK, "_src")

    # 1) descargar fuente (ya cuantizada)
    log("1/4 Descargando fuente cuantizada (UsefulSensors/moonshine-es)...")
    enc_q = hf_hub_download(SRC_REPO, "onnx/merged/base/quantized/encoder_model.onnx", local_dir=dl)
    dec_q = hf_hub_download(SRC_REPO, "onnx/merged/base/quantized/decoder_model_merged.onnx", local_dir=dl)
    shutil.copy(enc_q, os.path.join(WORK, "onnx", "encoder_model_quantized.onnx"))

    # 2) ARREGLAR el decoder: copiar scale+zero_point del embed_tokens dentro de las ramas del If
    log("2/4 Arreglando el decoder (poniendo el scale del embed_tokens dentro del If)...")
    m = onnx.load(dec_q)
    g = m.graph
    top = {i.name: i for i in g.initializer}
    need = [f"{EMB}_scale", f"{EMB}_zero_point"]
    copies = [top[n] for n in need if n in top]
    if len(copies) != len(need):
        print("   [!] No encontré los escalares esperados; el modelo fuente cambió. Abortando.")
        print("   initializers embed_tokens:", [n for n in top if "embed_tokens" in n]); sys.exit(1)
    If = [n for n in g.node if n.op_type == "If"][0]
    for a in If.attribute:
        if a.type == onnx.AttributeProto.GRAPH:
            sub = a.g
            subn = {i.name for i in sub.initializer}
            used = set()
            for nd in sub.node: used |= set(nd.input)
            for c in copies:
                if c.name in used and c.name not in subn:
                    ni = sub.initializer.add(); ni.CopyFrom(c)
    onnx.save(m, os.path.join(WORK, "onnx", "decoder_model_merged_quantized.onnx"))

    # 3) metadata + is_multilingual:false
    log("3/4 Bajando metadata y forzando is_multilingual:false...")
    for f in ["config.json","generation_config.json","preprocessor_config.json",
              "tokenizer.json","tokenizer_config.json","special_tokens_map.json"]:
        p = hf_hub_download(META_REPO, f, local_dir=os.path.join(WORK, "_meta"))
        shutil.copy(p, os.path.join(WORK, f))
    for f in ["config.json","generation_config.json"]:
        d = json.load(open(os.path.join(WORK, f), encoding="utf-8"))
        d["is_multilingual"] = False
        json.dump(d, open(os.path.join(WORK, f), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # verificación local opcional
    try:
        import onnxruntime as ort
        ort.InferenceSession(os.path.join(WORK,"onnx","decoder_model_merged_quantized.onnx"),
                             providers=["CPUExecutionProvider"])
        print("   verificación: el decoder arreglado CARGA en onnxruntime ✓")
    except ImportError:
        print("   (onnxruntime no instalado; me salto la verificación local, no pasa nada)")
    except Exception as e:
        print("   [!] el decoder NO cargó:", str(e)[:160]); sys.exit(1)

    for t in [dl, os.path.join(WORK,"_meta")]:
        shutil.rmtree(t, ignore_errors=True)
    print("\nCarpeta lista en:", WORK)
    for r,_,fs in os.walk(WORK):
        for f in fs:
            print(f"   {os.path.getsize(os.path.join(r,f))/1e6:7.2f} MB  {os.path.relpath(os.path.join(r,f),WORK)}")

    # 4) subir
    if not SUBIR:
        log("SUBIR=False -> no se sube. Revisa la carpeta de arriba."); return
    if "PON_AQUI" in HF_TOKEN:
        print("\n[!] Edita HF_TOKEN con tu token WRITE antes de subir."); return
    log(f"4/4 Subiendo a {HF_REPO} ...")
    api = HfApi(token=HF_TOKEN)
    print("   autenticado como:", api.whoami()["name"])
    api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
    # borra archivos viejos del repo para que no queden los float que pesaban
    try:
        api.delete_folder(path_in_repo="onnx", repo_id=HF_REPO, repo_type="model",
                          commit_message="limpiar onnx viejo")
    except Exception:
        pass
    api.upload_folder(folder_path=WORK, repo_id=HF_REPO, repo_type="model",
                      commit_message="Moonshine-ES base q8 (merged arreglado, español) para transformers.js")
    print("\n✅ LISTO. Subido a https://huggingface.co/" + HF_REPO)

if __name__ == "__main__":
    main()
