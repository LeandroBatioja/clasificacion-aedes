"""
============================================================
 FASE 6 - EVALUACION SOBRE EL CONJUNTO DE PRUEBA Y
          VALIDACION CONTEXTUAL (CONJUNTO ECUATORIANO)
============================================================
 Comparacion de arquitecturas de aprendizaje profundo para
 la clasificacion de mosquitos vectores del dengue
 Leandro Steven Batioja Mena - PUCE Esmeraldas 2026-2027

 Aplica, sin cambiar nada:
   - plan, 3.7 (Fase 6) y 3.8
   - fase2/protocolo_evaluacion.md v1.0
   - fase2/protocolo_evaluacion_anexo_fase6.md v1.0
 El conjunto de prueba se usa aqui por primera vez.

 PASOS
   1. Predicciones de los 20 modelos (5 semillas x 4 arquitecturas)
      sobre la prueba (589) y el conjunto ecuatoriano (127).
      Se guardan y no se repiten si el programa se vuelve a lanzar.
   2. Nivel 1 (estabilidad): metricas de cada corrida en prueba y en
      el control; media, DE, minimo y maximo.
   3. Nivel 2 (corrida mediana): metricas por subconjunto con IC por
      bootstrap de grupos; Delta_control y Delta_campo; Q de Cochran,
      W y McNemar con Bonferroni.
   4. Eficiencia: cada arquitectura en un proceso aparte.
   5. Jerarquia de seleccion en prueba y en control; senales de atajo
      por plataforma.
   6. Grad-CAM sobre las 64 imagenes de prueba con sus cajas.
   7. Validacion contextual (conjunto ecuatoriano).
   8. Figuras.

 USO (desde ~/Imagenes/tesis)
   python fase6/evaluar.py --solo-verificar     (comprueba entradas)
   python -u fase6/evaluar.py > fase6/log_evaluacion.txt 2>&1
============================================================
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

RAIZ_SCRIPT = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ_SCRIPT.parent / "fase4"))

NOMBRE = {"efficientnet_b0": "EfficientNet-B0", "mobilenetv3_large_100": "MobileNetV3-Large",
          "resnet50": "ResNet50", "vit_small_patch16_224": "ViT-S/16"}
SEMILLAS = (42, 43, 44, 45, 46)
CORTO = {"aedes_aegypti": "ae", "aedes_albopictus": "al",
         "dryad": "D", "ong": "O", "inaturalist": "I", "mosquito_alert": "M"}
CLASE_ECU = {"Aedes_aegypti": "aedes_aegypti", "Aedes_albopictus": "aedes_albopictus"}
OBS_PARCIAL = "156329843"          # anexo, 6.1: 32 vistas parciales de un mismo ejemplar
UMBRAL_DELTA = 0.05                 # protocolo v1.0, 3
UMBRAL_EMPATE = 0.02                # plan 3.8 y anexo, 4
UMBRAL_PICOS = 25.0                 # protocolo v1.0, 5.5
MIN_INFORMATIVAS = 8
PROTOCOLO_SHA = "0a88a99c6235d9f68432fd908e25f3cb3c7ee927954e7cfda98f98618376cb31"


# ------------------------------------------------------------
# utilidades
def sha256(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def leer_csv(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def escribir_csv(p, filas, campos=None):
    p.parent.mkdir(parents=True, exist_ok=True)
    campos = campos or (list(filas[0]) if filas else ["vacio"])
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)


def guardar_json(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2, ensure_ascii=False, default=float)


def r4(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


def titulo(t):
    print("\n" + "=" * 70 + f"\n  {t}\n" + "=" * 70, flush=True)


# ------------------------------------------------------------
# metricas rapidas (numpy). Clase 0 = aegypti, 1 = albopictus
def conteos(y, p):
    tp1 = int(np.sum((y == 1) & (p == 1)))
    tn1 = int(np.sum((y == 0) & (p == 0)))
    fp1 = int(np.sum((y == 0) & (p == 1)))
    fn1 = int(np.sum((y == 1) & (p == 0)))
    return tn1, fp1, fn1, tp1     # matriz [[tn1, fp1], [fn1, tp1]] con filas = real


def f1_de(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def resumen_metricas(y, p):
    """Todas las metricas de la seccion 3.8 (NaN si no definidas)."""
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    tn, fp, fn, tp = conteos(y, p)          # desde la clase albopictus
    nae, nal = int(np.sum(y == 0)), int(np.sum(y == 1))
    sens_ae = tn / nae if nae else float("nan")
    sens_al = tp / nal if nal else float("nan")
    prec_ae = tn / (tn + fn) if (tn + fn) else float("nan")
    prec_al = tp / (tp + fp) if (tp + fp) else float("nan")
    f1_ae = f1_de(tn, fn, fp) if nae else float("nan")
    f1_al = f1_de(tp, fp, fn) if nal else float("nan")
    ambas = nae > 0 and nal > 0
    return {"n": n, "n_aegypti": nae, "n_albopictus": nal,
            "exactitud": (tn + tp) / n if n else float("nan"),
            "exactitud_balanceada": (sens_ae + sens_al) / 2 if ambas else float("nan"),
            "macro_f1": (f1_ae + f1_al) / 2 if ambas else float("nan"),
            "sens_aegypti": sens_ae, "sens_albopictus": sens_al,
            "espec_aegypti": sens_al, "espec_albopictus": sens_ae,
            "prec_aegypti": prec_ae, "prec_albopictus": prec_al,
            "f1_aegypti": f1_ae, "f1_albopictus": f1_al,
            "matriz_confusion": [[tn, fp], [fn, tp]]}


def macro_f1_rapido(y, p):
    tn, fp, fn, tp = conteos(y, p)
    if not (np.any(y == 0) and np.any(y == 1)):
        return float("nan")
    return (f1_de(tn, fn, fp) + f1_de(tp, fp, fn)) / 2


def ic(valores):
    v = np.asarray([x for x in valores if not math.isnan(x)])
    if len(v) == 0:
        return None, None
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# ------------------------------------------------------------
# bootstrap por grupos, estratificado
def replicas_bootstrap(claves_grupo, estratos, n_rep, semilla):
    """claves_grupo[i] y estratos[i] por imagen. Devuelve lista de arrays de indices de imagen."""
    rnd = np.random.default_rng(semilla)
    por_grupo = defaultdict(list)
    estrato_de = {}
    for i, (g, e) in enumerate(zip(claves_grupo, estratos)):
        por_grupo[g].append(i)
        estrato_de[g] = e
    grupos_estrato = defaultdict(list)
    for g in sorted(por_grupo):
        grupos_estrato[estrato_de[g]].append(g)
    indices = {g: np.array(v) for g, v in por_grupo.items()}
    reps = []
    for _ in range(n_rep):
        partes = []
        for e in sorted(grupos_estrato):
            gs = grupos_estrato[e]
            for k in rnd.integers(0, len(gs), len(gs)):
                partes.append(indices[gs[k]])
        reps.append(np.concatenate(partes))
    return reps


# ------------------------------------------------------------
# modelos e imagenes
def preparar_cache_ecuador(filas, carpeta, cache, lado):
    from PIL import Image, ImageOps
    hechas = 0
    for f in filas:
        dst = cache / f["clase"] / (Path(f["filename"]).stem + ".jpg")
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(carpeta / f["clase"] / f["filename"]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((lado, lado), Image.LANCZOS)
            im.save(dst, quality=95)
        hechas += 1
    return hechas


def cargar_modelo(arq, ckpt):
    import timm
    import torch
    m = timm.create_model(arq, pretrained=False, num_classes=2)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    m.eval()
    return m, m.pretrained_cfg["mean"], m.pretrained_cfg["std"]


def transformacion(mean, std):
    from torchvision import transforms as T
    from entrenar import TAM
    return T.Compose([T.Resize((TAM, TAM), interpolation=T.InterpolationMode.BICUBIC),
                      T.ToTensor(), T.Normalize(mean, std)])


def predecir(modelo, rutas, tf, lote=64):
    import torch
    from PIL import Image
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(rutas), lote):
            x = torch.stack([tf(Image.open(r).convert("RGB")) for r in rutas[i:i + lote]])
            o = torch.softmax(modelo(x), 1)
            preds += o.argmax(1).tolist()
            probs += o[:, 1].tolist()
    return preds, probs


# ------------------------------------------------------------
# eficiencia en un proceso aparte (anexo, 5)
def rss_actual_mb():
    """RAM residente actual del proceso (Linux); None si no se puede leer."""
    try:
        for l in open("/proc/self/status", encoding="utf-8"):
            if l.startswith("VmRSS:"):
                return int(l.split()[1]) / 1024
    except OSError:
        return None


def medir_eficiencia(arq, ckpt, lista_rutas, hilos):
    import torch
    torch.set_num_threads(hilos)
    from PIL import Image
    base = rss_actual_mb()                      # con torch y timm ya cargados, sin modelo
    modelo, mean, std = cargar_modelo(arq, ckpt)
    tf = transformacion(mean, std)
    rutas = [l for l in Path(lista_rutas).read_text(encoding="utf-8").splitlines() if l][:110]
    xs = [tf(Image.open(r).convert("RGB")).unsqueeze(0) for r in rutas]
    tiempos = []
    with torch.no_grad():
        for k, x in enumerate(xs):
            t0 = time.perf_counter()
            modelo(x)
            if k >= 10:
                tiempos.append(1000 * (time.perf_counter() - t0))
        lote = torch.cat(xs[:64])
        modelo(lote)                                   # calentamiento
        t0 = time.perf_counter()
        for _ in range(3):
            modelo(lote)
        ms_lote = 1000 * (time.perf_counter() - t0) / (3 * len(lote))
    despues = rss_actual_mb()
    pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps({
        "arquitectura": arq, "checkpoint": str(ckpt),
        "parametros": sum(p.numel() for p in modelo.parameters()),
        "tamano_checkpoint_mb": round(Path(ckpt).stat().st_size / 2 ** 20, 2),
        "ms_por_imagen_lote1_mediana": round(float(np.median(tiempos)), 2),
        "ms_por_imagen_lote1_p10_p90": [round(float(np.percentile(tiempos, 10)), 2),
                                        round(float(np.percentile(tiempos, 90)), 2)],
        "imagenes_medidas": len(tiempos),
        "ms_por_imagen_lote64": round(ms_lote, 2),
        "ram_base_mb": round(base) if base else None, "ram_tras_inferencia_mb": round(despues) if despues else None,
        "ram_pico_proceso_mb": round(pico),
        "ram_modelo_e_inferencia_mb": round(despues - base) if base and despues else None,
        "hilos": hilos}))


# ------------------------------------------------------------
# Grad-CAM (protocolo v1.0, 5)
def gradcam_modelo(arq, modelo, mean, std, filas, cajas, ruta_de):
    from PIL import Image
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from entrenar import TAM, CLASES, capa_gradcam, reshape_vit
    for p in modelo.parameters():
        p.requires_grad_(True)
    modelo.eval()
    cam = GradCAM(model=modelo, target_layers=[capa_gradcam(arq, modelo)],
                  reshape_transform=reshape_vit if arq.startswith("vit") else None)
    tf = transformacion(mean, std)
    res, mapas = [], []
    for f in filas:
        im = Image.open(ruta_de(f)).convert("RGB")
        x = tf(im).unsqueeze(0)
        pred = int(modelo(x).argmax(1))
        mapa = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]
        c = cajas[(f["especie"], f["nombre_destino"])]
        W, H = float(c["ancho"]), float(c["alto"])
        x0, x1 = int(float(c["x0"]) * TAM / W), int(math.ceil(float(c["x1"]) * TAM / W))
        y0, y1 = int(float(c["y0"]) * TAM / H), int(math.ceil(float(c["y1"]) * TAM / H))
        dentro = np.zeros_like(mapa, dtype=bool)
        dentro[y0:y1, x0:x1] = True
        py, px = np.unravel_index(np.argmax(mapa), mapa.shape)
        area = dentro.mean()
        masa = mapa[dentro].sum() / mapa.sum() if mapa.sum() > 0 else 0.0
        res.append({"arquitectura": arq, "estrato": f["estrato"], "nombre_destino": f["nombre_destino"],
                    "especie": f["especie"], "grupo": f["grupo"], "prediccion": CLASES[pred],
                    "acierto": int(CLASES[pred] == f["especie"]), "informativa": c["informativa"],
                    "pico_dentro": int(dentro[py, px]),
                    "concentracion": round(float(masa / area), 3) if area > 0 else "",
                    "fraccion_area": round(float(area), 3)})
        mapas.append((mapa, (x0, y0, x1, y1), (px, py), CLASES[pred] == f["especie"], CLASES[pred]))
    return res, mapas


def resumen_gradcam(filas):
    inf = [r for r in filas if r["informativa"] == "si"]
    fuera = sum(1 - r["pico_dentro"] for r in inf)
    conc = [r["concentracion"] for r in inf if r["concentracion"] != ""]
    pct = 100 * fuera / len(inf) if inf else None
    med = float(np.median(conc)) if conc else None
    return {"imagenes": len(filas), "informativas": len(inf), "picos_fuera": fuera,
            "pct_picos_fuera": r4(pct), "mediana_concentracion": r4(med),
            "alerta_fondo": pct is not None and pct > UMBRAL_PICOS and len(inf) >= MIN_INFORMATIVAS,
            "alerta_no_localizada": med is not None and med <= 1.0,
            "evaluable_por_estrato": len(inf) >= MIN_INFORMATIVAS}


def laminas_gradcam(muestra, mapas_por_arq, ruta_de, salida, arqs):
    from PIL import Image, ImageDraw
    from entrenar import TAM
    salida.mkdir(parents=True, exist_ok=True)
    por_estrato = defaultdict(list)
    for i, f in enumerate(muestra):
        por_estrato[f["estrato"]].append(i)
    cab = 18
    for est, idx in sorted(por_estrato.items()):
        hoja = Image.new("RGB", ((1 + len(arqs)) * TAM, cab + len(idx) * TAM), "white")
        d = ImageDraw.Draw(hoja)
        for c, t in enumerate(["Imagen"] + [NOMBRE[a] for a in arqs]):
            d.text((c * TAM + 4, 3), t, fill=(0, 0, 0))
        for fila, i in enumerate(idx):
            im = Image.open(ruta_de(muestra[i])).convert("RGB").resize((TAM, TAM))
            y = cab + fila * TAM
            hoja.paste(im, (0, y))
            rgb = np.asarray(im).astype(np.float32) / 255
            for c, a in enumerate(arqs, start=1):
                mapa, (x0, y0, x1, y1), (px, py), ok, pred = mapas_por_arq[a][i]
                calor = np.stack([mapa, mapa * 0.4, 1 - mapa], -1)
                v = Image.fromarray((255 * (0.55 * rgb + 0.45 * calor)).clip(0, 255).astype(np.uint8))
                dv = ImageDraw.Draw(v)
                dv.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(0, 255, 0), width=2)
                dv.ellipse([px - 4, py - 4, px + 4, py + 4], outline=(255, 255, 255), width=2)
                dv.text((3, 3), f"{'OK' if ok else 'X'} {pred[6:10]}", fill=(255, 255, 0))
                hoja.paste(v, (c * TAM, y))
        hoja.save(salida / f"{est}.jpg", quality=90)


# ------------------------------------------------------------
# jerarquia de seleccion (plan 3.8; anexo, 4)
def jerarquia(datos):
    """datos[arq] = {macro_f1, sens_aegypti, de_macro_f1, ms_inferencia, tamano_mb}"""
    traza = []
    cand = list(datos)
    mejor = max(datos[a]["macro_f1"] for a in cand)
    cand = [a for a in cand if mejor - datos[a]["macro_f1"] < UMBRAL_EMPATE]
    traza.append(f"1. Macro F1: mejor {mejor:.4f}; empatadas (dif. < {UMBRAL_EMPATE}): "
                 + ", ".join(f"{NOMBRE[a]} {datos[a]['macro_f1']:.4f}" for a in cand))
    if len(cand) > 1:
        m = max(datos[a]["sens_aegypti"] for a in cand)
        cand = [a for a in cand if m - datos[a]["sens_aegypti"] < UMBRAL_EMPATE]
        traza.append(f"2. Sensibilidad Ae. aegypti: mejor {m:.4f}; empatadas: "
                     + ", ".join(f"{NOMBRE[a]} {datos[a]['sens_aegypti']:.4f}" for a in cand))
    if len(cand) > 1:
        m = min(round(datos[a]["de_macro_f1"], 4) for a in cand)
        cand = [a for a in cand if round(datos[a]["de_macro_f1"], 4) == m]
        traza.append(f"3. Menor DE del macro F1 (5 corridas): {m:.4f}: " + ", ".join(NOMBRE[a] for a in cand))
    if len(cand) > 1 and all(datos[a].get("ms_inferencia") is not None for a in cand):
        cand = sorted(cand, key=lambda a: (datos[a]["ms_inferencia"], datos[a]["tamano_mb"]))[:1]
        traza.append(f"4. Eficiencia (ms/imagen, tamano): {NOMBRE[cand[0]]}")
    elif len(cand) > 1:
        traza.append("4. Eficiencia no medida (--sin-eficiencia); empate sin resolver")
    return {"seleccionada": cand[0] if len(cand) == 1 else None, "candidatas_finales": cand, "traza": traza}


# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particion", default="fase3/particion")
    ap.add_argument("--depurado", default="fase3/conjunto_depurado_v2")
    ap.add_argument("--cache", default="fase4/cache_512")
    ap.add_argument("--definitivas", default="fase5/definitivas")
    ap.add_argument("--ecuador", default="fase6/Dataset_Dengue_Aedes_Ecuador")
    ap.add_argument("--procedencia-inat", default="fase3/inaturalist/procedencia_inaturalist.csv")
    ap.add_argument("--protocolo", default="fase2/protocolo_evaluacion.md")
    ap.add_argument("--anexo", default="fase2/protocolo_evaluacion_anexo_fase6.md")
    ap.add_argument("--salida", default="fase6")
    ap.add_argument("--replicas", type=int, default=2000)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--hilos", type=int, default=6)
    ap.add_argument("--sin-gradcam", action="store_true")
    ap.add_argument("--sin-eficiencia", action="store_true")
    ap.add_argument("--sin-figuras", action="store_true")
    ap.add_argument("--solo-verificar", action="store_true")
    ap.add_argument("--medir", nargs=3, metavar=("ARQ", "CKPT", "LISTA"), help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.medir:
        medir_eficiencia(*args.medir, args.hilos)
        return

    import torch
    torch.set_num_threads(args.hilos)
    from entrenar import ARQUITECTURAS, CLASES, CACHE_LADO, ruta_cache, preparar_cache

    t_inicio = time.time()
    sal = Path(args.salida)
    sal.mkdir(parents=True, exist_ok=True)
    part, dfin = Path(args.particion), Path(args.definitivas)

    # ---------- entradas ----------
    titulo("FASE 6 - EVALUACION (plan 3.7-3.8; protocolo v1.0 y anexo)")
    sha_p, sha_a = sha256(args.protocolo), sha256(args.anexo)
    print(f"  Protocolo v1.0: {sha_p} {'(coincide con la Entrada 007 bis)' if sha_p == PROTOCOLO_SHA else '(NO COINCIDE)'}")
    print(f"  Anexo Fase 6:   {sha_a or 'NO ENCONTRADO'}")
    particion = leer_csv(part / "particion.csv")
    prueba = [f for f in particion if f["subconjunto"] == "prueba"]
    for f in prueba:
        f["estrato"] = f"{CORTO[f['origen']]}-{CORTO[f['especie']]}"
    resumen = json.load(open(dfin / "resumen_definitivas.json", encoding="utf-8"))
    medianas = {a: int(resumen["estadisticos"][a]["semilla_corrida_mediana"]) for a in ARQUITECTURAS}
    muestra_gc = leer_csv(part / "gradcam_muestra.csv")
    cajas = {(r["especie"], r["nombre_destino"]): r for r in leer_csv(part / "gradcam_cajas.csv")}
    ecu_dir = Path(args.ecuador)
    ecuador = leer_csv(ecu_dir / "metadata.csv") if (ecu_dir / "metadata.csv").exists() else []

    problemas = []
    if sha_p != PROTOCOLO_SHA:
        problemas.append("la huella del protocolo v1.0 no coincide")
    if len(prueba) != 589:
        problemas.append(f"la prueba tiene {len(prueba)} imagenes (se esperaban 589)")
    faltan_ck = [f"{a}/semilla_{s}" for a in ARQUITECTURAS for s in SEMILLAS
                 if not (dfin / a / f"semilla_{s}" / "modelo.pt").exists()]
    if faltan_ck:
        problemas.append(f"faltan checkpoints: {faltan_ck}")
    faltan_img = [f["nombre_destino"] for f in prueba
                  if not (Path(args.depurado) / f["especie"] / f["nombre_destino"]).exists()
                  and not ruta_cache(Path(args.cache), f).exists()]
    if faltan_img:
        problemas.append(f"faltan {len(faltan_img)} imagenes de prueba, p. ej. {faltan_img[:3]}")
    sin_caja = [f["nombre_destino"] for f in muestra_gc if (f["especie"], f["nombre_destino"]) not in cajas]
    if sin_caja:
        problemas.append(f"{len(sin_caja)} imagenes de Grad-CAM sin caja")
    if not ecuador:
        problemas.append(f"no se encontro {ecu_dir / 'metadata.csv'}")
    else:
        faltan_e = [f["filename"] for f in ecuador if not (ecu_dir / f["clase"] / f["filename"]).exists()]
        if faltan_e:
            problemas.append(f"faltan {len(faltan_e)} imagenes ecuatorianas")
    c_est = Counter(f["estrato"] for f in prueba)
    print(f"  Prueba: {len(prueba)} imagenes, {len({(f['especie'], f['grupo']) for f in prueba})} grupos; "
          f"estratos {dict(sorted(c_est.items()))}")
    print(f"  Corridas medianas: " + ", ".join(f"{NOMBRE[a]} {s}" for a, s in medianas.items()))
    print(f"  Grad-CAM: {len(muestra_gc)} imagenes, {len(cajas)} cajas")
    c_ecu = Counter((f["clase"], f["quality_grade"]) for f in ecuador)
    print(f"  Ecuador: {len(ecuador)} imagenes {dict(c_ecu)}")

    # fuga: observaciones o fotos del conjunto ecuatoriano en el entrenamiento (anexo, 6.4)
    fuga = []
    pi = Path(args.procedencia_inat)
    if pi.exists():
        proc = leer_csv(pi)
        obs_tr = {str(r.get("observacion", "")) for r in proc}
        fot_tr = {str(r.get("foto", "")) for r in proc}
        fuga = [f["filename"] for f in ecuador if f["observation_id"] in obs_tr or f["photo_id"] in fot_tr]
        print(f"  Fuga con iNaturalist de entrenamiento ({len(proc)} fotos): {len(fuga)} coincidencias")
    else:
        problemas.append(f"no se encontro {pi} para comprobar la fuga")
    if problemas:
        print("\n  PROBLEMAS:\n   - " + "\n   - ".join(problemas))
        if args.solo_verificar or any("checkpoints" in p or "prueba tiene" in p or "imagenes de prueba" in p
                                      for p in problemas):
            sys.exit(1)
    if args.solo_verificar:
        print("\n  Verificacion correcta: se puede ejecutar la evaluacion.")
        return
    ecuador = [f for f in ecuador if f["filename"] not in set(fuga)]

    # ---------- 1. predicciones ----------
    titulo("1. PREDICCIONES (20 modelos)")
    cache = Path(args.cache)
    n = preparar_cache(prueba, Path(args.depurado), cache)
    cache_ecu = sal / "cache_ecuador_512"
    n_e = preparar_cache_ecuador(ecuador, ecu_dir, cache_ecu, CACHE_LADO)
    print(f"  Copias reducidas nuevas: prueba {n}, Ecuador {n_e}")
    rutas_p = [ruta_cache(cache, f) for f in prueba]
    rutas_e = [cache_ecu / f["clase"] / (Path(f["filename"]).stem + ".jpg") for f in ecuador]
    y = np.array([CLASES.index(f["especie"]) for f in prueba])
    y_e = np.array([CLASES.index(CLASE_ECU[f["clase"]]) for f in ecuador])
    P, PE, PROB, PROBE = {}, {}, {}, {}
    for a in ARQUITECTURAS:
        for s in SEMILLAS:
            dp = sal / "predicciones" / a / f"semilla_{s}.csv"
            if dp.exists():
                filas = leer_csv(dp)
                # el nombre de archivo se repite entre especies (p. ej. ong__7.jpg): clave = especie + nombre
                fp = {(r["especie"], r["id"]): r for r in filas if r["conjunto"] == "prueba"}
                fe = {r["id"]: r for r in filas if r["conjunto"] == "ecuador"}
                if len(fp) == len(prueba) and all((f["especie"], f["nombre_destino"]) in fp for f in prueba) \
                        and all(f["filename"] in fe for f in ecuador):
                    P[a, s] = np.array([CLASES.index(fp[f["especie"], f["nombre_destino"]]["prediccion"]) for f in prueba])
                    PROB[a, s] = [float(fp[f["especie"], f["nombre_destino"]]["prob_albopictus"]) for f in prueba]
                    PE[a, s] = np.array([CLASES.index(fe[f["filename"]]["prediccion"]) for f in ecuador])
                    PROBE[a, s] = [float(fe[f["filename"]]["prob_albopictus"]) for f in ecuador]
                    print(f"  {NOMBRE[a]:<18} semilla {s}: ya calculada")
                    continue
            t0 = time.time()
            modelo, mean, std = cargar_modelo(a, dfin / a / f"semilla_{s}" / "modelo.pt")
            tf = transformacion(mean, std)
            pp, pr = predecir(modelo, rutas_p, tf)
            pe, pre = predecir(modelo, rutas_e, tf)
            filas = [{"conjunto": "prueba", "id": f["nombre_destino"], "especie": f["especie"],
                      "origen": f["origen"], "grupo": f["grupo"], "estrato": f["estrato"],
                      "prediccion": CLASES[k], "prob_albopictus": round(q, 5)}
                     for f, k, q in zip(prueba, pp, pr)]
            filas += [{"conjunto": "ecuador", "id": f["filename"], "especie": CLASE_ECU[f["clase"]],
                       "origen": f"inaturalist_ecuador_{f['quality_grade']}", "grupo": f["observation_id"],
                       "estrato": "ecuador", "prediccion": CLASES[k], "prob_albopictus": round(q, 5)}
                      for f, k, q in zip(ecuador, pe, pre)]
            escribir_csv(dp, filas)
            P[a, s], PROB[a, s], PE[a, s], PROBE[a, s] = np.array(pp), pr, np.array(pe), pre
            print(f"  {NOMBRE[a]:<18} semilla {s}: macro F1 prueba {macro_f1_rapido(y, P[a, s]):.4f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
            del modelo

    # ---------- 2. nivel 1: estabilidad ----------
    titulo("2. NIVEL 1 - ESTABILIDAD (5 corridas, prueba y control)")
    control = np.array([f["parte"] == "control" for f in prueba])
    METR = ["exactitud", "exactitud_balanceada", "macro_f1", "sens_aegypti", "sens_albopictus",
            "espec_aegypti", "espec_albopictus", "prec_aegypti", "prec_albopictus", "f1_aegypti", "f1_albopictus"]
    filas_est, estab = [], {}
    for a in ARQUITECTURAS:
        estab[a] = {}
        for nombre_sub, m in (("prueba", np.ones(len(y), bool)), ("control", control)):
            vals = defaultdict(list)
            for s in SEMILLAS:
                r = resumen_metricas(y[m], P[a, s][m])
                filas_est.append({"arquitectura": a, "subconjunto": nombre_sub, "semilla": s,
                                  "corrida_mediana": int(s == medianas[a]), **{k: r4(r[k]) for k in METR}})
                for k in METR:
                    vals[k].append(r[k])
            estab[a][nombre_sub] = {k: {"media": r4(np.mean(v)), "de": r4(np.std(v, ddof=1)),
                                        "min": r4(min(v)), "max": r4(max(v))} for k, v in vals.items()}
    escribir_csv(sal / "estabilidad_5_corridas.csv", filas_est)
    print(f"  {'Arquitectura':<20}{'Macro F1 prueba (media +- DE)':>32}{'Rango':>18}{'Control (media +- DE)':>26}")
    for a in ARQUITECTURAS:
        e, c = estab[a]["prueba"]["macro_f1"], estab[a]["control"]["macro_f1"]
        print(f"  {NOMBRE[a]:<20}{e['media']:>22.4f} +- {e['de']:.4f}{e['min']:>10.4f}-{e['max']:.4f}"
              f"{c['media']:>16.4f} +- {c['de']:.4f}")

    # ---------- 3. nivel 2: corrida mediana ----------
    titulo("3. NIVEL 2 - CORRIDA MEDIANA: METRICAS POR SUBCONJUNTO (IC 95 %, bootstrap por grupos)")
    Pm = {a: P[a, medianas[a]] for a in ARQUITECTURAS}
    claves = [(f["especie"], f["grupo"]) for f in prueba]
    reps = replicas_bootstrap(claves, [f["estrato"] for f in prueba], args.replicas, args.semilla)
    est_arr = np.array([f["estrato"] for f in prueba])
    origen = np.array([f["origen"] for f in prueba])
    SUBS = {"prueba": np.ones(len(y), bool), "control": control, "dryad": origen == "dryad",
            "ong": origen == "ong", "campo": ~control, "I-ae": est_arr == "I-ae",
            "M-al": est_arr == "M-al", "M-ae": est_arr == "M-ae"}

    def boot(a, mascara, fn):
        out = []
        for idx in reps:
            idx = idx[mascara[idx]]
            out.append(fn(y[idx], Pm[a][idx]) if len(idx) else float("nan"))
        return out

    fn_m = {"macro_f1": macro_f1_rapido,
            "exactitud": lambda yy, pp: float(np.mean(yy == pp)) if len(yy) else float("nan"),
            "exactitud_balanceada": lambda yy, pp: (float(np.mean(pp[yy == 0] == 0)) + float(np.mean(pp[yy == 1] == 1))) / 2
            if (np.any(yy == 0) and np.any(yy == 1)) else float("nan"),
            "sens_aegypti": lambda yy, pp: float(np.mean(pp[yy == 0] == 0)) if np.any(yy == 0) else float("nan"),
            "sens_albopictus": lambda yy, pp: float(np.mean(pp[yy == 1] == 1)) if np.any(yy == 1) else float("nan")}
    filas_m, metricas_sub = [], defaultdict(dict)
    for a in ARQUITECTURAS:
        for sub, m in SUBS.items():
            r = resumen_metricas(y[m], Pm[a][m])
            metricas_sub[a][sub] = {k: (v if isinstance(v, (int, list)) else r4(v)) for k, v in r.items()}
            n_grupos = len({claves[i] for i in np.where(m)[0]})
            for k in METR:
                if r[k] is None or (isinstance(r[k], float) and math.isnan(r[k])):
                    continue
                lo, hi = ic(boot(a, m, fn_m[k])) if (k in fn_m and sub != "M-ae") else (None, None)
                filas_m.append({"arquitectura": a, "subconjunto": sub, "n_imagenes": int(m.sum()),
                                "n_grupos": n_grupos, "metrica": k, "valor": r4(r[k]),
                                "ic95_inf": r4(lo), "ic95_sup": r4(hi)})
                if lo is not None:
                    metricas_sub[a][sub][f"ic_{k}"] = [r4(lo), r4(hi)]
    escribir_csv(sal / "metricas_por_estrato.csv", filas_m)
    guardar_json(sal / "metricas_prueba.json", {"corrida_mediana": medianas, "metricas": metricas_sub})
    print(f"  {'Arquitectura':<20}{'Prueba':>22}{'Control':>22}{'Campo':>22}{'Sens I-ae':>11}{'Sens M-al':>11}{'M-ae':>7}")
    for a in ARQUITECTURAS:
        def fmt(sub):
            d = metricas_sub[a][sub]
            lo, hi = d.get("ic_macro_f1", [None, None])
            return f"{d['macro_f1']:.3f} [{lo:.3f}-{hi:.3f}]"
        mae = metricas_sub[a]["M-ae"]
        print(f"  {NOMBRE[a]:<20}{fmt('prueba'):>22}{fmt('control'):>22}{fmt('campo'):>22}"
              f"{metricas_sub[a]['I-ae']['sens_aegypti']:>11.3f}{metricas_sub[a]['M-al']['sens_albopictus']:>11.3f}"
              f"{(round(mae['sens_aegypti'] * mae['n']) if mae['sens_aegypti'] is not None else 0):>4}/{mae['n']}")
    for a in ARQUITECTURAS:
        print(f"  {NOMBRE[a]:<20} matriz (filas real ae/al, columnas pred ae/al): "
              f"{metricas_sub[a]['prueba']['matriz_confusion']}")

    # Delta_control y Delta_campo (protocolo v1.0, 3)
    titulo("3b. SUBCONJUNTO DE CONTROL (protocolo v1.0, 3)")
    ctrl_res = {}
    for a in ARQUITECTURAS:
        dc, dk = [], []
        for idx in reps:
            fa = macro_f1_rapido(y[idx], Pm[a][idx])
            ic_ = idx[control[idx]]
            ik_ = idx[~control[idx]]
            fc = macro_f1_rapido(y[ic_], Pm[a][ic_])
            fk = macro_f1_rapido(y[ik_], Pm[a][ik_])
            dc.append(fa - fc)
            dk.append(fk - fc)
        d_c = metricas_sub[a]["prueba"]["macro_f1"] - metricas_sub[a]["control"]["macro_f1"]
        d_k = metricas_sub[a]["campo"]["macro_f1"] - metricas_sub[a]["control"]["macro_f1"]
        lo, hi = ic(dc)
        lo2, hi2 = ic(dk)
        senal = d_c >= UMBRAL_DELTA and lo > 0
        ctrl_res[a] = {"delta_control": r4(d_c), "ic95": [r4(lo), r4(hi)], "senal_dependencia": senal,
                       "delta_campo": r4(d_k), "ic95_campo": [r4(lo2), r4(hi2)]}
        print(f"  {NOMBRE[a]:<20} Delta_control {d_c:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"{'SENAL' if senal else 'sin senal':<10} Delta_campo {d_k:+.4f} [{lo2:+.4f}, {hi2:+.4f}]")
    orden_p = sorted(ARQUITECTURAS, key=lambda a: -metricas_sub[a]["prueba"]["macro_f1"])
    orden_c = sorted(ARQUITECTURAS, key=lambda a: -metricas_sub[a]["control"]["macro_f1"])
    print(f"  Orden en prueba:  {' > '.join(NOMBRE[a] for a in orden_p)}")
    print(f"  Orden en control: {' > '.join(NOMBRE[a] for a in orden_c)}   "
          f"({'coincide' if orden_p == orden_c else 'DIFIERE'})")

    # ---------- pruebas estadisticas (plan 3.8; anexo, 2) ----------
    titulo("3c. PRUEBAS ESTADISTICAS (corrida mediana, 589 imagenes)")
    from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
    C = np.column_stack([(Pm[a] == y).astype(int) for a in ARQUITECTURAS])
    nimg, k = C.shape
    q = cochrans_q(C)
    Q, pq = float(q.statistic), float(q.pvalue)
    W = Q / (nimg * (k - 1))
    print(f"  Q de Cochran = {Q:.3f}, gl = {k - 1}, p = {pq:.4g}   W = {W:.4f}   "
          f"({'significativa' if pq < 0.05 else 'no significativa'})")
    pares = []
    for i, j in combinations(range(k), 2):
        ai, aj = ARQUITECTURAS[i], ARQUITECTURAS[j]
        n11 = int(np.sum((C[:, i] == 1) & (C[:, j] == 1)))
        b = int(np.sum((C[:, i] == 1) & (C[:, j] == 0)))
        c = int(np.sum((C[:, i] == 0) & (C[:, j] == 1)))
        n00 = int(np.sum((C[:, i] == 0) & (C[:, j] == 0)))
        r = mcnemar([[n11, b], [c, n00]], exact=True)
        dif = (b - c) / nimg
        se = math.sqrt(max(b + c - (b - c) ** 2 / nimg, 0)) / nimg
        db = [float(np.mean(Pm[ai][idx] == y[idx]) - np.mean(Pm[aj][idx] == y[idx])) for idx in reps]
        lo, hi = ic(db)
        pb = min(1.0, float(r.pvalue) * 6)
        pares.append({"par": f"{NOMBRE[ai]} vs {NOMBRE[aj]}", "a": ai, "b": aj, "ambos_aciertan": n11,
                      "solo_a": b, "solo_b": c, "ambos_fallan": n00, "p_exacta": r4(float(r.pvalue)),
                      "p_bonferroni": r4(pb), "significativa": pb < 0.05 and pq < 0.05,
                      "dif_exactitud": r4(dif), "ic95_wald": [r4(dif - 1.96 * se), r4(dif + 1.96 * se)],
                      "ic95_bootstrap_grupos": [r4(lo), r4(hi)]})
        print(f"  {pares[-1]['par']:<38} b={b:>3} c={c:>3}  p={r.pvalue:.4f}  p_Bonf={pb:.4f}  "
              f"dif {dif:+.4f} Wald [{dif - 1.96 * se:+.4f}, {dif + 1.96 * se:+.4f}] "
              f"bootstrap [{lo:+.4f}, {hi:+.4f}]")
    if pq >= 0.05:
        print("  La Q de Cochran no es significativa: los pares se informan, pero no se interpretan (anexo, 2).")
    guardar_json(sal / "pruebas_estadisticas.json", {
        "n_imagenes": nimg, "arquitecturas": ARQUITECTURAS,
        "exactitud": {a: r4(np.mean(Pm[a] == y)) for a in ARQUITECTURAS},
        "cochran": {"Q": r4(Q), "gl": k - 1, "p": float(pq), "W": r4(W), "significativa": pq < 0.05},
        "mcnemar": pares, "correccion": "Bonferroni (x6)", "alfa": 0.05})

    # ---------- 4. eficiencia (la jerarquia la usa en el criterio 4) ----------
    efic = {}
    if not args.sin_eficiencia:
        titulo("4. EFICIENCIA (un proceso por arquitectura; anexo, 5)")
        lista = sal / "_rutas_eficiencia.txt"
        lista.write_text("\n".join(str(r) for r in rutas_p[:110]), encoding="utf-8")
        for a in ARQUITECTURAS:
            ck = dfin / a / f"semilla_{medianas[a]}" / "modelo.pt"
            out = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--medir", a, str(ck), str(lista),
                                  "--hilos", str(args.hilos)], capture_output=True, text=True)
            linea = [l for l in out.stdout.splitlines() if l.startswith("{")]
            if out.returncode or not linea:
                print(f"  {NOMBRE[a]}: ERROR\n{out.stderr[-800:]}")
                continue
            efic[a] = json.loads(linea[-1])
            e = efic[a]
            print(f"  {NOMBRE[a]:<20} {e['ms_por_imagen_lote1_mediana']:>7.2f} ms/img (lote 1)  "
                  f"{e['ms_por_imagen_lote64']:>6.2f} ms/img (lote 64)  RAM {e['ram_tras_inferencia_mb']} MB "
                  f"(base {e['ram_base_mb']}, modelo e inferencia {e['ram_modelo_e_inferencia_mb']}, "
                  f"pico {e['ram_pico_proceso_mb']})  {e['tamano_checkpoint_mb']} MB  {e['parametros'] / 1e6:.2f} M",
                  flush=True)
        lista.unlink(missing_ok=True)
        guardar_json(sal / "eficiencia.json", efic)

    # ---------- jerarquia de seleccion ----------
    titulo("5. SELECCION DEL MEJOR MODELO (plan 3.8; anexo, 4)")
    sel = {}
    for sub in ("prueba", "control"):
        datos = {a: {"macro_f1": metricas_sub[a][sub]["macro_f1"], "sens_aegypti": metricas_sub[a][sub]["sens_aegypti"],
                     "de_macro_f1": estab[a][sub]["macro_f1"]["de"],
                     "ms_inferencia": efic.get(a, {}).get("ms_por_imagen_lote1_mediana"),
                     "tamano_mb": efic.get(a, {}).get("tamano_checkpoint_mb")} for a in ARQUITECTURAS}
        sel[sub] = {"datos": datos, **jerarquia(datos)}
        print(f"  Sobre {sub}:")
        for t in sel[sub]["traza"]:
            print(f"    {t}")
        print(f"    -> {NOMBRE.get(sel[sub]['seleccionada'], 'empate sin resolver')}")
    sel["coincide"] = sel["prueba"]["seleccionada"] == sel["control"]["seleccionada"]
    sel["orden_prueba"], sel["orden_control"] = orden_p, orden_c
    guardar_json(sal / "seleccion_mejor_modelo.json", sel)
    guardar_json(sal / "control_dryad_ong.json", {"umbral": UMBRAL_DELTA, "por_arquitectura": ctrl_res,
                                                  "orden_prueba": orden_p, "orden_control": orden_c,
                                                  "orden_coincide": orden_p == orden_c,
                                                  "seleccion_coincide": sel["coincide"]})

    # ---------- atajo por plataforma (protocolo v1.0, 4) ----------
    atajo = []
    for i, f in enumerate(prueba):
        if f["estrato"] == "M-ae":
            atajo.append({"caso": "M-ae (prueba)", "imagen": f["nombre_destino"], "grupo": f["grupo"],
                          "calidad": "", **{NOMBRE[a]: CLASES[Pm[a][i]][6:] for a in ARQUITECTURAS}})
    for i, f in enumerate(ecuador):
        if f["clase"] == "Aedes_albopictus":
            atajo.append({"caso": "albopictus Ecuador", "imagen": f["filename"], "grupo": f["observation_id"],
                          "calidad": f["quality_grade"],
                          **{NOMBRE[a]: CLASES[PE[a, medianas[a]][i]][6:] for a in ARQUITECTURAS}})
    escribir_csv(sal / "atajo_plataforma.csv", atajo,
                 ["caso", "imagen", "grupo", "calidad"] + [NOMBRE[a] for a in ARQUITECTURAS])

    # ---------- 6. Grad-CAM ----------
    gc_res = {}
    if not args.sin_gradcam:
        titulo("6. GRAD-CAM (64 imagenes de prueba; protocolo v1.0, 5)")
        idx_de = {(f["especie"], f["nombre_destino"]): f for f in prueba}
        muestra = []
        for g in muestra_gc:
            f = dict(idx_de[g["especie"], g["nombre_destino"]])
            f["estrato"] = g["estrato"]
            muestra.append(f)
        ruta_de = lambda f: ruta_cache(cache, f)
        filas_gc, mapas = [], {}
        for a in ARQUITECTURAS:
            t0 = time.time()
            modelo, mean, std = cargar_modelo(a, dfin / a / f"semilla_{medianas[a]}" / "modelo.pt")
            res, mapas[a] = gradcam_modelo(a, modelo, mean, std, muestra, cajas, ruta_de)
            filas_gc += res
            gc_res[a] = {"total": resumen_gradcam(res),
                         "por_estrato": {e: resumen_gradcam([r for r in res if r["estrato"] == e])
                                         for e in sorted({r["estrato"] for r in res})}}
            t = gc_res[a]["total"]
            alertas = [e for e, v in gc_res[a]["por_estrato"].items() if v["alerta_fondo"]]
            print(f"  {NOMBRE[a]:<20} picos fuera {t['picos_fuera']}/{t['informativas']} ({t['pct_picos_fuera']} %)  "
                  f"mediana conc. {t['mediana_concentracion']}  alerta total: "
                  f"{'SI' if t['alerta_fondo'] or t['alerta_no_localizada'] else 'no'}  "
                  f"estratos con alerta: {alertas or 'ninguno'}  ({time.time() - t0:.0f} s)", flush=True)
            del modelo
        escribir_csv(sal / "gradcam_indicadores.csv", filas_gc)
        guardar_json(sal / "gradcam_resumen.json", gc_res)
        laminas_gradcam(muestra, mapas, ruta_de, sal / "gradcam_laminas", ARQUITECTURAS)
        print(f"  Laminas: {sal / 'gradcam_laminas'}/")

    # ---------- 7. Ecuador ----------
    titulo("7. VALIDACION CONTEXTUAL - CONJUNTO ECUATORIANO (plan 3.8; anexo, 6)")
    q_ = np.array([f["quality_grade"] for f in ecuador])
    obs = np.array([f["observation_id"] for f in ecuador])
    R = q_ == "research"
    Rae, Ral = R & (y_e == 0), R & (y_e == 1)
    idx_r = np.where(R)[0]                     # bootstrap por observacion, solo research, estratificado por especie
    reps_e = [idx_r[r] for r in replicas_bootstrap([(ecuador[i]["clase"], obs[i]) for i in idx_r],
                                                   [ecuador[i]["clase"] for i in idx_r], args.replicas, args.semilla)]
    ecu, casos = {"fuga_excluidas": fuga, "por_arquitectura": {}}, []
    print(f"  Research: {int(Rae.sum())} aegypti ({len(set(obs[Rae]))} obs.) y {int(Ral.sum())} albopictus "
          f"({len(set(obs[Ral]))} obs.)")
    print(f"  {'Arquitectura':<20}{'Sens ae (71)':>22}{'Exact. (75)':>13}{'F1 ae (75)':>12}{'Por obs.':>10}"
          f"{'Sin 156329843':>15}{'Sens 119':>10}{'Prueba I-ae':>13}{'Prueba ae':>11}")
    for a in ARQUITECTURAS:
        pe = PE[a, medianas[a]]
        sens = float(np.mean(pe[Rae] == 0))
        r75 = resumen_metricas(y_e[R], pe[R])
        b_s, b_e, b_f = [], [], []
        for idx in reps_e:
            ya, pa = y_e[idx], pe[idx]
            if np.any(ya == 0):
                b_s.append(float(np.mean(pa[ya == 0] == 0)))
                b_e.append(float(np.mean(ya == pa)))
                tn, fp, fn, tp = conteos(ya, pa)
                b_f.append(f1_de(tn, fn, fp))
        por_obs = defaultdict(list)
        for i in np.where(Rae)[0]:
            por_obs[obs[i]].append(int(pe[i] == 0))
        media_obs = float(np.mean([np.mean(v) for v in por_obs.values()]))
        mayoria = sum(1 for v in por_obs.values() if np.mean(v) > 0.5)
        sin_p = Rae & (obs != OBS_PARCIAL)
        todas_ae = y_e == 0
        sens5 = [float(np.mean(PE[a, s][Rae] == 0)) for s in SEMILLAS]
        d = {"sens_aegypti_71": r4(sens), "ic95_sens": [r4(x) for x in ic(b_s)],
             "exactitud_75": r4(r75["exactitud"]), "ic95_exactitud": [r4(x) for x in ic(b_e)],
             "f1_aegypti_75": r4(r75["f1_aegypti"]), "ic95_f1": [r4(x) for x in ic(b_f)],
             "matriz_75": r75["matriz_confusion"],
             "por_observacion_media": r4(media_obs), "observaciones_con_mayoria_de_aciertos": mayoria,
             "observaciones": len(por_obs),
             "aciertos_por_observacion": {o: f"{sum(v)}/{len(v)}" for o, v in sorted(por_obs.items())},
             "sens_sin_obs_parcial": r4(np.mean(pe[sin_p] == 0)), "n_sin_obs_parcial": int(sin_p.sum()),
             "sens_obs_parcial": r4(np.mean(pe[Rae & (obs == OBS_PARCIAL)] == 0)),
             "sens_aegypti_119": r4(np.mean(pe[todas_ae] == 0)), "n_aegypti_todas": int(todas_ae.sum()),
             "metricas_127": {k: (v if isinstance(v, (int, list)) else r4(v))
                              for k, v in resumen_metricas(y_e, pe).items()},
             "albopictus_research_aciertos": f"{int(np.sum(pe[Ral] == 1))}/{int(Ral.sum())}",
             "albopictus_todas_aciertos": f"{int(np.sum(pe[y_e == 1] == 1))}/{int(np.sum(y_e == 1))}",
             "sens_71_cinco_corridas": {"valores": [r4(v) for v in sens5], "media": r4(np.mean(sens5)),
                                        "de": r4(np.std(sens5, ddof=1))},
             "comparacion_prueba": {"sens_I_ae": metricas_sub[a]["I-ae"]["sens_aegypti"],
                                    "sens_aegypti_global": metricas_sub[a]["prueba"]["sens_aegypti"],
                                    "f1_aegypti_global": metricas_sub[a]["prueba"]["f1_aegypti"]}}
        ecu["por_arquitectura"][a] = d
        print(f"  {NOMBRE[a]:<20}{sens:>8.3f} [{d['ic95_sens'][0]:.3f}-{d['ic95_sens'][1]:.3f}]"
              f"{d['exactitud_75']:>13.3f}{d['f1_aegypti_75']:>12.3f}{media_obs:>10.3f}"
              f"{d['sens_sin_obs_parcial']:>15.3f}{d['sens_aegypti_119']:>10.3f}"
              f"{d['comparacion_prueba']['sens_I_ae']:>13.3f}{d['comparacion_prueba']['sens_aegypti_global']:>11.3f}")
    print("  Albopictus ecuatorianos (aciertos): " + ", ".join(
        f"{NOMBRE[a]} {ecu['por_arquitectura'][a]['albopictus_research_aciertos']} research, "
        f"{ecu['por_arquitectura'][a]['albopictus_todas_aciertos']} todas" for a in ARQUITECTURAS))
    for i, f in enumerate(ecuador):
        casos.append({"archivo": f["filename"], "clase": f["clase"], "calidad": f["quality_grade"],
                      "observacion": f["observation_id"], "lugar": f["place_guess"], "fecha": f["observed_on"],
                      **{NOMBRE[a]: CLASES[PE[a, medianas[a]][i]][6:] for a in ARQUITECTURAS},
                      **{f"prob_albopictus_{a}": round(PROBE[a, medianas[a]][i], 4) for a in ARQUITECTURAS}})
    escribir_csv(sal / "ecuador_casos.csv", casos)
    guardar_json(sal / "ecuador.json", ecu)

    # ---------- 8. figuras ----------
    if not args.sin_figuras:
        titulo("8. FIGURAS")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig_dir = sal / "figuras"
            fig_dir.mkdir(parents=True, exist_ok=True)
            for a in ARQUITECTURAS:
                fig, ax = plt.subplots(1, 2, figsize=(11, 4))
                for s in SEMILLAS:
                    pc = dfin / a / f"semilla_{s}" / "curvas.csv"
                    if not pc.exists():
                        continue
                    cv = leer_csv(pc)
                    ep = [int(r["epoca"]) for r in cv]
                    estilo = "-" if s == medianas[a] else ":"
                    l, = ax[0].plot(ep, [float(r["perdida_entrenamiento"]) for r in cv], estilo, label=f"semilla {s}")
                    ax[0].plot(ep, [float(r["perdida_validacion"]) for r in cv], estilo, color=l.get_color(), alpha=0.45)
                    f1v = [float(r["macro_f1_validacion"]) for r in cv]
                    ax[1].plot(ep, f1v, estilo, color=l.get_color(), label=f"semilla {s}")
                    b = int(np.argmax(f1v))
                    ax[1].plot(ep[b], f1v[b], "o", color=l.get_color())
                ax[0].set(title=f"{NOMBRE[a]}: perdida (oscura: entrenamiento; clara: validacion)",
                          xlabel="Epoca", ylabel="Entropia cruzada")
                ax[1].set(title=f"{NOMBRE[a]}: macro F1 en validacion (punto: checkpoint)",
                          xlabel="Epoca", ylabel="Macro F1")
                ax[1].legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(fig_dir / f"curvas_{a}.png", dpi=150)
                plt.close(fig)
            fig, ax = plt.subplots(figsize=(8, 4))
            xs = np.arange(len(ARQUITECTURAS))
            for k_, (sub, col) in enumerate((("prueba", "#4C72B0"), ("control", "#DD8452"), ("campo", "#55A868"))):
                v = [metricas_sub[a][sub]["macro_f1"] for a in ARQUITECTURAS]
                lo = [v[i] - metricas_sub[a][sub]["ic_macro_f1"][0] for i, a in enumerate(ARQUITECTURAS)]
                hi = [metricas_sub[a][sub]["ic_macro_f1"][1] - v[i] for i, a in enumerate(ARQUITECTURAS)]
                ax.bar(xs + (k_ - 1) * 0.27, v, 0.27, yerr=[lo, hi], capsize=3, label=sub, color=col)
            ax.set_xticks(xs, [NOMBRE[a] for a in ARQUITECTURAS])
            ax.set(ylabel="Macro F1 (IC 95 %)", ylim=(0.5, 1.0), title="Macro F1 en prueba por subconjunto (corrida mediana)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / "macro_f1_subconjuntos.png", dpi=150)
            plt.close(fig)
            fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
            for ax, a in zip(axs, ARQUITECTURAS):
                mc = np.array(metricas_sub[a]["prueba"]["matriz_confusion"])
                ax.imshow(mc, cmap="Blues")
                for (i, j), v in np.ndenumerate(mc):
                    ax.text(j, i, str(v), ha="center", va="center", color="white" if v > mc.max() / 2 else "black")
                ax.set_xticks([0, 1], ["aegypti", "albopictus"])
                ax.set_yticks([0, 1], ["aegypti", "albopictus"])
                ax.set(title=NOMBRE[a], xlabel="Prediccion", ylabel="Real")
            fig.tight_layout()
            fig.savefig(fig_dir / "matrices_confusion.png", dpi=150)
            plt.close(fig)
            print(f"  Figuras en {fig_dir}/")
        except Exception as e:
            print(f"  Figuras no generadas: {type(e).__name__}: {e}")

    # ---------- reporte ----------
    guardar_json(sal / "reporte_fase6.json", {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duracion_min": round((time.time() - t_inicio) / 60, 1),
        "entorno": {"python": platform.python_version(), "torch": torch.__version__, "hilos": args.hilos},
        "protocolo_sha256": sha_p, "anexo_sha256": sha_a, "replicas": args.replicas, "semilla": args.semilla,
        "corrida_mediana": medianas,
        "estabilidad": {a: {s: estab[a][s]["macro_f1"] for s in ("prueba", "control")} for a in ARQUITECTURAS},
        "metricas_prueba": {a: metricas_sub[a]["prueba"] for a in ARQUITECTURAS},
        "control": ctrl_res, "orden_prueba": orden_p, "orden_control": orden_c,
        "cochran": {"Q": r4(Q), "p": float(pq), "W": r4(W)},
        "seleccion": {"prueba": sel["prueba"]["seleccionada"], "control": sel["control"]["seleccionada"]},
        "gradcam": {a: v["total"] for a, v in gc_res.items()},
        "ecuador": {a: {k: v[k] for k in ("sens_aegypti_71", "ic95_sens", "albopictus_research_aciertos")}
                    for a, v in ecu["por_arquitectura"].items()},
        "eficiencia": efic})
    print(f"\n  Reporte: {sal / 'reporte_fase6.json'}   ({(time.time() - t_inicio) / 60:.1f} min)")


if __name__ == "__main__":
    main()
