"""
============================================================
 FASES 4-5 - ENTRENAMIENTO Y EVALUACION (PRUEBA PILOTO)
============================================================
 Comparacion de arquitecturas de aprendizaje profundo para
 la clasificacion de mosquitos vectores del dengue
 Leandro Steven Batioja Mena - PUCE Esmeraldas 2026-2027

 PROPOSITO
 ---------
 Un solo programa para el piloto (Fase 4) y las corridas
 definitivas (Fase 5), de modo que las cuatro arquitecturas
 pasen exactamente por el mismo codigo.

 Prueba piloto (plan 3.7, Fase 4):
   - 400 imagenes de entrenamiento (fase3/particion/piloto_entrenamiento.csv)
   - 5 epocas, una configuracion y una semilla por arquitectura
   - metricas sobre el conjunto de VALIDACION completo
   - Grad-CAM sobre 20 imagenes de validacion con sus cajas
     (protocolo de evaluacion, apartado 6)
   - criterios de aceptacion y extrapolacion del tiempo por
     corrida completa: (t / imagenes_piloto / 5) x imagenes x 15

 DECISIONES DE IMPLEMENTACION (registrar en la ficha tecnica)
 ------------------------------------------------------------
   Carga        PIL + ImageOps.exif_transpose (igual que las cajas)
   Cache        copia de cada imagen con lado mayor <= 512 px
                (fase4/cache_512), para no decodificar fotografias
                de hasta 108 MP en cada epoca
   Entrada      redimensionado directo a 224 x 224 (bicubico) y
                normalizacion de cada modelo segun timm
   Aumento      x2: cada epoca recorre cada imagen original sin
                transformar y dos versiones aumentadas (volteo
                horizontal, rotacion +-15 grados, escala 0,9-1,1,
                brillo y contraste +-0,2). Solo en entrenamiento
   Ajuste fino  parcial: se congela la red y se entrenan los
                ultimos bloques y el clasificador (ver BLOQUES).
                Las capas congeladas mantienen BatchNorm en modo
                evaluacion
   Optimizador  AdamW (weight decay 0,01, valor por defecto de PyTorch),
                perdida de entropia cruzada (ficha tecnica, 5)
   Checkpoint   epoca con mayor macro F1 en validacion
   Detencion    temprana tras 5 epocas sin mejora (Fase 5)

 FASE 5 (plan 3.2 y 3.7; ficha tecnica, 5)
   --modo busqueda    3 configuraciones al azar (semilla 42) del espacio
                      lr {1e-3, 5e-4, 1e-4} x batch {16, 32}, las mismas
                      para las cuatro arquitecturas; 30 % del
                      entrenamiento por grupos; maximo 10 epocas.
                      Elige la de mayor macro F1 en validacion.
   --modo definitiva  5 corridas por arquitectura (semillas 42-46) con
                      la configuracion elegida; entrenamiento completo,
                      maximo 15 epocas, detencion tras 5 sin mejora;
                      checkpoint de mayor macro F1 en validacion.
   Ambos modos se pueden interrumpir y retomar: las corridas
   terminadas no se repiten. El conjunto de prueba NO se usa.

 USO (desde ~/Imagenes/tesis)
 ---
   python fase4/entrenar.py --modo piloto
   python fase4/entrenar.py --modo piloto --arquitecturas resnet50
   python fase4/entrenar.py --modo piloto --solo-gradcam   (repite Grad-CAM)

 Configuracion unica del piloto: lr 5e-4 (valor central del espacio de
 busqueda de la ficha) y batch 32. La busqueda de hiperparametros es de
 la Fase 5.

 SALIDAS (fase4/piloto/)
 -------
   <arq>/curvas.csv           perdida y metricas por epoca
   <arq>/metricas_val.json    metricas del mejor checkpoint
   <arq>/predicciones_val.csv prediccion por imagen
   <arq>/modelo.pt            mejor checkpoint
   <arq>/gradcam.csv          pico dentro y concentracion
   <arq>/gradcam.jpg          lamina con mapas y cajas
   reporte_piloto.json        criterios de aceptacion y
                              extrapolacion para las cuatro
============================================================
"""

import argparse
import csv
import json
import math
import os
import platform
import random
import resource
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

Image.MAX_IMAGE_PIXELS = None

ARQUITECTURAS = ["efficientnet_b0", "mobilenetv3_large_100", "resnet50", "vit_small_patch16_224"]
CLASES = ["aedes_aegypti", "aedes_albopictus"]
TAM = 224
CACHE_LADO = 512


# ------------------------------------------------------------
# Ajuste fino parcial: modulos que se entrenan en cada arquitectura.
# Criterio comun: la ultima etapa de la red (aprox. el ultimo cuarto de
# la profundidad) mas la cabeza de clasificacion.
def bloques_entrenables(nombre, m):
    if nombre == "efficientnet_b0":            # 7 etapas: se entrenan la 6 y la 7
        return [m.blocks[5], m.blocks[6], m.conv_head, m.bn2, m.classifier]
    if nombre == "mobilenetv3_large_100":      # 7 etapas: se entrenan la 6 y la 7
        return [m.blocks[5], m.blocks[6], m.conv_head, m.norm_head, m.classifier]
    if nombre == "resnet50":                   # 4 etapas: se entrena layer4
        return [m.layer4, m.fc]
    if nombre == "vit_small_patch16_224":      # 12 bloques: se entrenan los 3 ultimos
        return [m.blocks[9], m.blocks[10], m.blocks[11], m.norm, m.fc_norm, m.head]
    raise ValueError(nombre)


def capa_gradcam(nombre, m):
    if nombre == "efficientnet_b0":
        return m.bn2                           # tras la ultima convolucion, antes del pooling
    if nombre == "mobilenetv3_large_100":
        return m.blocks[-1]                    # en timm, conv_head va despues del pooling
    if nombre == "resnet50":
        return m.layer4[-1]
    if nombre == "vit_small_patch16_224":
        return m.blocks[-1].norm1
    raise ValueError(nombre)


def reshape_vit(t, h=14, w=14):
    r = t[:, 1:, :].reshape(t.size(0), h, w, t.size(2))
    return r.permute(0, 3, 1, 2)


# ------------------------------------------------------------
def fijar_semilla(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)


def ruta_cache(cache, fila):
    return cache / fila["especie"] / (Path(fila["nombre_destino"]).stem + ".jpg")


def preparar_cache(filas, depurado, cache):
    hechas = 0
    for f in filas:
        dst = ruta_cache(cache, f)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(depurado / f["especie"] / f["nombre_destino"]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((CACHE_LADO, CACHE_LADO), Image.LANCZOS)
            im.save(dst, quality=95)
        hechas += 1
    return hechas


class Conjunto(Dataset):
    def __init__(self, filas, cache, mean, std, factor_aumento=0):
        self.filas, self.cache, self.k = filas, cache, factor_aumento
        base = [T.Resize((TAM, TAM), interpolation=T.InterpolationMode.BICUBIC)]
        fin = [T.ToTensor(), T.Normalize(mean, std)]
        self.t_orig = T.Compose(base + fin)
        self.t_aum = T.Compose(base + [
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=15, scale=(0.9, 1.1), fill=0),
            T.ColorJitter(brightness=0.2, contrast=0.2)] + fin)

    def __len__(self):
        return len(self.filas) * (1 + self.k)

    def __getitem__(self, i):
        f = self.filas[i % len(self.filas)]
        im = Image.open(ruta_cache(self.cache, f)).convert("RGB")
        t = self.t_orig if i < len(self.filas) else self.t_aum
        return t(im), CLASES.index(f["especie"]), i % len(self.filas)


def ram_pico_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def metricas(y, p):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                 f1_score, precision_recall_fscore_support)
    cm = confusion_matrix(y, p, labels=[0, 1])
    pr, rc, f1, sop = precision_recall_fscore_support(y, p, labels=[0, 1], zero_division=0)
    res = {"exactitud": accuracy_score(y, p), "exactitud_balanceada": balanced_accuracy_score(y, p),
           "macro_f1": f1_score(y, p, average="macro", zero_division=0),
           "matriz_confusion": cm.tolist(), "por_especie": {}}
    for c, nom in enumerate(CLASES):
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fp = cm[:, c].sum() - cm[c, c]
        res["por_especie"][nom] = {"precision": pr[c], "sensibilidad": rc[c], "f1": f1[c],
                                   "especificidad": tn / (tn + fp) if (tn + fp) else 0.0,
                                   "soporte": int(sop[c])}
    return json.loads(json.dumps(res, default=float))


def evaluar(modelo, cargador):
    modelo.eval()
    ys, ps, idx, prob, perd = [], [], [], [], 0.0
    crit = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y, i in cargador:
            o = modelo(x)
            perd += crit(o, y).item()
            ys += y.tolist()
            ps += o.argmax(1).tolist()
            idx += i.tolist()
            prob += torch.softmax(o, 1)[:, 1].tolist()
    return ys, ps, idx, prob, perd / max(1, len(ys))


def modo_entrenamiento(modelo, entrenables):
    """Train en los modulos que se ajustan; eval (BatchNorm fijo) en el resto."""
    modelo.eval()
    for mod in entrenables:
        mod.train()


# ------------------------------------------------------------
def gradcam_indicadores(nombre, modelo, filas, cajas, cache, mean, std, salida):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    for p in modelo.parameters():
        p.requires_grad_(True)
    modelo.eval()
    cam = GradCAM(model=modelo, target_layers=[capa_gradcam(nombre, modelo)],
                  reshape_transform=reshape_vit if nombre.startswith("vit") else None)
    t = T.Compose([T.Resize((TAM, TAM), interpolation=T.InterpolationMode.BICUBIC),
                   T.ToTensor(), T.Normalize(mean, std)])
    res, vistas = [], []
    for f in filas:
        im = Image.open(ruta_cache(cache, f)).convert("RGB")
        x = t(im).unsqueeze(0)
        pred = int(modelo(x).argmax(1))
        mapa = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]   # 224 x 224, en [0, 1]
        c = cajas[(f["especie"], f["nombre_destino"])]
        W, H = float(c["ancho"]), float(c["alto"])
        x0, x1 = int(float(c["x0"]) * TAM / W), int(math.ceil(float(c["x1"]) * TAM / W))
        y0, y1 = int(float(c["y0"]) * TAM / H), int(math.ceil(float(c["y1"]) * TAM / H))
        dentro = np.zeros_like(mapa, dtype=bool)
        dentro[y0:y1, x0:x1] = True
        py, px = np.unravel_index(np.argmax(mapa), mapa.shape)
        area = dentro.mean()
        masa = mapa[dentro].sum() / mapa.sum() if mapa.sum() > 0 else 0.0
        res.append({"nombre_destino": f["nombre_destino"], "especie": f["especie"],
                    "prediccion": CLASES[pred], "acierto": int(CLASES[pred] == f["especie"]),
                    "informativa": c["informativa"], "pico_dentro": int(dentro[py, px]),
                    "concentracion": round(float(masa / area), 3) if area > 0 else "",
                    "fraccion_area": round(float(area), 3)})
        rgb = np.asarray(im.resize((TAM, TAM))).astype(np.float32) / 255
        calor = np.stack([mapa, mapa * 0.4, 1 - mapa], -1)
        mezcla = Image.fromarray((255 * (0.55 * rgb + 0.45 * calor)).clip(0, 255).astype(np.uint8))
        d = ImageDraw.Draw(mezcla)
        d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(0, 255, 0), width=2)
        d.ellipse([px - 4, py - 4, px + 4, py + 4], outline=(255, 255, 255), width=2)
        d.text((3, 3), f"{'OK' if res[-1]['acierto'] else 'X'} {CLASES[pred][6:10]}", fill=(255, 255, 0))
        vistas.append(mezcla)
    cols = 5
    hoja = Image.new("RGB", (cols * TAM, math.ceil(len(vistas) / cols) * TAM), "white")
    for k, v in enumerate(vistas):
        hoja.paste(v, ((k % cols) * TAM, (k // cols) * TAM))
    hoja.save(salida / "gradcam.jpg", quality=90)
    with open(salida / "gradcam.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    inf = [r for r in res if r["informativa"] == "si"]
    fuera = sum(1 - r["pico_dentro"] for r in inf)
    conc = sorted(r["concentracion"] for r in inf if r["concentracion"] != "")
    return {"imagenes": len(res), "informativas": len(inf),
            "picos_fuera": fuera, "pct_picos_fuera": round(100 * fuera / len(inf), 1) if inf else None,
            "mediana_concentracion": conc[len(conc) // 2] if conc else None}


# ------------------------------------------------------------
def entrenar_una(nombre, args, entren, val, cache, salida, pesos):
    import timm
    fijar_semilla(args.semilla)
    salida.mkdir(parents=True, exist_ok=True)
    modelo = timm.create_model(nombre, pretrained=pesos, num_classes=2)
    cfg = modelo.pretrained_cfg
    mean, std = cfg["mean"], cfg["std"]
    for p in modelo.parameters():
        p.requires_grad_(False)
    entrenables = bloques_entrenables(nombre, modelo)
    for mod in entrenables:
        for p in mod.parameters():
            p.requires_grad_(True)
    n_total = sum(p.numel() for p in modelo.parameters())
    n_entr = sum(p.numel() for p in modelo.parameters() if p.requires_grad)

    g = torch.Generator().manual_seed(args.semilla)
    ds_e = Conjunto(entren, cache, mean, std, args.factor_aumento)
    ds_v = Conjunto(val, cache, mean, std, 0)
    dl_e = DataLoader(ds_e, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                      generator=g, worker_init_fn=lambda w: np.random.seed(args.semilla + w))
    dl_v = DataLoader(ds_v, batch_size=64, shuffle=False, num_workers=args.workers)
    opt = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    curvas, mejor, sin_mejora = [], -1.0, 0
    t_entreno = 0.0
    print(f"\n  >>> {nombre}: parametros {n_total / 1e6:.2f} M, entrenables {n_entr / 1e6:.2f} M "
          f"({100 * n_entr / n_total:.0f} %), {len(ds_e)} imagenes por epoca")
    for ep in range(1, args.epocas + 1):
        modo_entrenamiento(modelo, entrenables)
        t0 = time.time()
        perd, aciertos, n = 0.0, 0, 0
        for x, y, _ in dl_e:
            opt.zero_grad()
            o = modelo(x)
            l = crit(o, y)
            l.backward()
            opt.step()
            perd += l.item() * len(y)
            aciertos += (o.argmax(1) == y).sum().item()
            n += len(y)
        te = time.time() - t0
        t_entreno += te
        t1 = time.time()
        ys, ps, idx, prob, perd_v = evaluar(modelo, dl_v)
        tv = time.time() - t1
        m = metricas(ys, ps)
        fila = {"epoca": ep, "perdida_entrenamiento": round(perd / n, 5),
                "exactitud_entrenamiento": round(aciertos / n, 4), "perdida_validacion": round(perd_v, 5),
                "exactitud_validacion": round(m["exactitud"], 4), "macro_f1_validacion": round(m["macro_f1"], 4),
                "segundos_entrenamiento": round(te, 1), "segundos_validacion": round(tv, 1),
                "img_s_entrenamiento": round(n / te, 1), "ram_pico_mb": round(ram_pico_mb())}
        curvas.append(fila)
        print(f"      epoca {ep}: perdida {fila['perdida_entrenamiento']:.4f}  val macroF1 "
              f"{fila['macro_f1_validacion']:.4f}  ({te:.0f} s, {fila['img_s_entrenamiento']} img/s)")
        if m["macro_f1"] > mejor:
            mejor, sin_mejora = m["macro_f1"], 0
            torch.save(modelo.state_dict(), salida / "modelo.pt")
            mejores = {"epoca": ep, "metricas": m, "ys": ys, "ps": ps, "idx": idx, "prob": prob}
        else:
            sin_mejora += 1
            if args.paciencia and sin_mejora >= args.paciencia:
                print(f"      detencion temprana en la epoca {ep}")
                break

    with open(salida / "curvas.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curvas[0]))
        w.writeheader()
        w.writerows(curvas)
    with open(salida / "predicciones_val.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["nombre_destino", "especie", "origen", "grupo", "prediccion", "prob_albopictus"])
        for i, p, pr in zip(mejores["idx"], mejores["ps"], mejores["prob"]):
            f = val[i]
            w.writerow([f["nombre_destino"], f["especie"], f["origen"], f["grupo"], CLASES[p], round(pr, 4)])
    json.dump({"arquitectura": nombre, "mejor_epoca": mejores["epoca"], **mejores["metricas"]},
              open(salida / "metricas_val.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    modelo.load_state_dict(torch.load(salida / "modelo.pt", weights_only=True))
    return modelo, mean, std, curvas, {"total": n_total, "entrenables": n_entr}, t_entreno, mejores


# ------------------------------------------------------------
FRACCIONES = (0.25, 0.50, 0.75, 1.00)
UMBRAL_ESTABLE = 0.02


def subconjuntos_anidados(filas, semilla, fracciones=FRACCIONES):
    """Fracciones anidadas (25 c 50 c 75 c 100 %) por grupos completos,
    estratificadas por especie x repositorio."""
    rnd = random.Random(semilla)
    estratos = {}
    for f in filas:
        estratos.setdefault((f["especie"], f["origen"]), {}).setdefault(f["grupo"], []).append(f)
    orden = {}
    for k in sorted(estratos):
        grupos = sorted(estratos[k])
        rnd.shuffle(grupos)
        orden[k] = [estratos[k][g] for g in grupos]
    salida = {}
    for fr in fracciones:
        sel = []
        for k, grupos in orden.items():
            total = sum(len(g) for g in grupos)
            tomadas = []
            for g in grupos:
                if len(tomadas) >= fr * total - 1e-9:
                    break
                tomadas.extend(g)
            sel.extend(tomadas)
        salida[fr] = sel
    return salida


def curva_aprendizaje(args, particion, val, entorno):
    """Plan, 3.4: verificacion empirica del tamano del conjunto mediante la
    curva de aprendizaje. Criterio fijado antes de ejecutar: el rendimiento
    se considera estabilizado si el macro F1 en validacion mejora menos de
    0,02 entre el 75 % y el 100 % del entrenamiento."""
    if args.arquitecturas == ARQUITECTURAS:
        args.arquitecturas = ["mobilenetv3_large_100"]      # la mas rapida (piloto: 0,7 h/corrida)
    if args.salida == "fase4/piloto":
        args.salida = "fase4/curva"
    entren_total = [f for f in particion if f["subconjunto"] == "entrenamiento"]
    subs = subconjuntos_anidados(entren_total, args.semilla)
    print("=" * 66)
    print("  CURVA DE APRENDIZAJE (FASE 4, plan 3.4)")
    print("=" * 66)
    print(f"  Arquitectura: {args.arquitecturas}   epocas: {args.epocas}   lr {args.lr}   batch {args.batch}")
    for fr, sel in subs.items():
        c = {e: sum(1 for f in sel if f["especie"] == e) for e in CLASES}
        print(f"    {int(fr * 100):>3} %: {len(sel)} imagenes {c}")
    cache = Path(args.cache)
    t0 = time.time()
    n = preparar_cache(entren_total + val, Path(args.depurado), cache)
    print(f"  Cache: {n} imagenes nuevas en {time.time() - t0:.0f} s")
    sal = Path(args.salida)
    sal.mkdir(parents=True, exist_ok=True)
    rep = {"fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "entorno": entorno,
           "configuracion": dict(vars(args)), "criterio": f"mejora 75->100 % < {UMBRAL_ESTABLE}",
           "resultados": {}}
    for nombre in args.arquitecturas:
        res = {}
        for fr, sel in subs.items():
            etiqueta = f"p{int(fr * 100):03d}"
            _, _, _, curvas, _, t_ent, mejores = entrenar_una(
                nombre, args, sel, val, cache, sal / nombre / etiqueta, not args.sin_pesos)
            res[etiqueta] = {"imagenes": len(sel),
                             "por_especie": {e: sum(1 for f in sel if f["especie"] == e) for e in CLASES},
                             "macro_f1_val": round(mejores["metricas"]["macro_f1"], 4),
                             "exactitud_val": round(mejores["metricas"]["exactitud"], 4),
                             "mejor_epoca": mejores["epoca"], "segundos": round(t_ent, 1)}
            rep["resultados"][nombre] = res
            json.dump(rep, open(sal / "reporte_curva.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        mejora = res["p100"]["macro_f1_val"] - res["p075"]["macro_f1_val"]
        res["mejora_75_100"] = round(mejora, 4)
        res["estabilizada"] = mejora < UMBRAL_ESTABLE
    print("\n  " + f"{'Arquitectura':<24}" + "".join(f"{int(f * 100):>9} %" for f in FRACCIONES)
          + f"{'Mejora 75-100':>15}{'Estable':>9}")
    for nombre, res in rep["resultados"].items():
        print(f"  {nombre:<24}" + "".join(f"{res[f'p{int(f * 100):03d}']['macro_f1_val']:>11.4f}" for f in FRACCIONES)
              + f"{res['mejora_75_100']:>15.4f}{('si' if res['estabilizada'] else 'NO'):>9}")
    json.dump(rep, open(sal / "reporte_curva.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n  Reporte: {sal / 'reporte_curva.json'}")


# ------------------------------------------------------------
# FASE 5 (plan 3.2 y 3.7; ficha tecnica, 5)
ESPACIO = [(lr, b) for lr in (1e-3, 5e-4, 1e-4) for b in (16, 32)]
N_CONFIG = 3
FRACCION_BUSQUEDA = 0.30
SEMILLAS_DEFINITIVAS = (42, 43, 44, 45, 46)


def configuraciones_busqueda(semilla):
    """Tres configuraciones al azar del espacio comun, las mismas para las cuatro
    arquitecturas (mismo presupuesto y mismas opciones para todas)."""
    rnd = random.Random(semilla)
    return sorted(rnd.sample(ESPACIO, N_CONFIG), key=lambda c: (-c[0], c[1]))


def copia_args(args, **cambios):
    a = argparse.Namespace(**vars(args))
    for k, v in cambios.items():
        setattr(a, k, v)
    return a


def busqueda(args, particion, val, entorno):
    if args.salida == "fase4/piloto":
        args.salida = "fase5/busqueda"
    sal = Path(args.salida)
    sal.mkdir(parents=True, exist_ok=True)
    entren_total = [f for f in particion if f["subconjunto"] == "entrenamiento"]
    sub = subconjuntos_anidados(entren_total, args.semilla, (FRACCION_BUSQUEDA,))[FRACCION_BUSQUEDA]
    configs = configuraciones_busqueda(args.semilla)
    cache = Path(args.cache)
    n = preparar_cache(entren_total + val, Path(args.depurado), cache)
    print("=" * 66)
    print("  BUSQUEDA DE HIPERPARAMETROS (FASE 5, plan 3.2)")
    print("=" * 66)
    print(f"  Submuestra: {len(sub)} imagenes ({int(FRACCION_BUSQUEDA * 100)} % del entrenamiento, "
          f"por grupos y estratificada)   cache nuevas: {n}")
    print(f"  Configuraciones (comunes a las cuatro): {configs}")
    print(f"  Maximo {args.epocas_busqueda} epocas, detencion tras {args.paciencia_def} sin mejora")
    rep_path = sal / "reporte_busqueda.json"
    rep = json.load(open(rep_path, encoding="utf-8")) if rep_path.exists() else {
        "entorno": entorno, "submuestra": len(sub), "configuraciones": configs, "resultados": {}}
    for nombre in args.arquitecturas:
        res = rep["resultados"].setdefault(nombre, {})
        for lr, b in configs:
            etiqueta = f"lr{lr:g}_b{b}"
            if etiqueta in res:
                print(f"  {nombre} {etiqueta}: ya hecha (macro F1 {res[etiqueta]['macro_f1_val']:.4f})")
                continue
            a = copia_args(args, lr=lr, batch=b, epocas=args.epocas_busqueda, paciencia=args.paciencia_def)
            t0 = time.time()
            _, _, _, curvas, _, t_ent, mejores = entrenar_una(
                nombre, a, sub, val, cache, sal / nombre / etiqueta, not args.sin_pesos)
            res[etiqueta] = {"lr": lr, "batch": b, "macro_f1_val": round(mejores["metricas"]["macro_f1"], 4),
                             "mejor_epoca": mejores["epoca"], "epocas": len(curvas),
                             "segundos": round(time.time() - t0, 1)}
            (sal / nombre / etiqueta / "modelo.pt").unlink(missing_ok=True)    # no se reutiliza
            rep["fecha"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            json.dump(rep, open(rep_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    # seleccion: mayor macro F1 en validacion
    sel = {}
    print(f"\n  {'Arquitectura':<24}" + "".join(f"{f'lr{lr:g} b{b}':>14}" for lr, b in configs) + "   Seleccionada")
    for nombre, res in rep["resultados"].items():
        if len(res) < len(configs):
            continue
        mejor = max(res.values(), key=lambda r: r["macro_f1_val"])
        sel[nombre] = {"lr": mejor["lr"], "batch": mejor["batch"], "macro_f1_val": mejor["macro_f1_val"]}
        print(f"  {nombre:<24}" + "".join(f"{res[f'lr{lr:g}_b{b}']['macro_f1_val']:>14.4f}" for lr, b in configs)
              + f"   lr {mejor['lr']:g}, batch {mejor['batch']}")
    rep["seleccion"] = sel
    json.dump(rep, open(rep_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(sel, open(sal / "configuracion_seleccionada.json", "w", encoding="utf-8"), indent=2)
    print(f"\n  Seleccion: {sal / 'configuracion_seleccionada.json'}")


def definitivas(args, particion, val, entorno):
    if args.salida == "fase4/piloto":
        args.salida = "fase5/definitivas"
    sal = Path(args.salida)
    sal.mkdir(parents=True, exist_ok=True)
    sel_path = Path(args.configuracion)
    if not sel_path.exists():
        sys.exit(f"ERROR: falta {sel_path}. Ejecute antes --modo busqueda")
    sel = json.load(open(sel_path, encoding="utf-8"))
    entren = [f for f in particion if f["subconjunto"] == "entrenamiento"]
    cache = Path(args.cache)
    n = preparar_cache(entren + val, Path(args.depurado), cache)
    print("=" * 66)
    print("  CORRIDAS DEFINITIVAS (FASE 5)")
    print("=" * 66)
    print(f"  Entrenamiento: {len(entren)} (x{1 + args.factor_aumento} por epoca)   validacion: {len(val)}   "
          f"cache nuevas: {n}")
    print(f"  Semillas: {SEMILLAS_DEFINITIVAS}   maximo {args.epocas_def} epocas, "
          f"detencion tras {args.paciencia_def} sin mejora")
    res_path = sal / "resumen_definitivas.json"
    resumen = json.load(open(res_path, encoding="utf-8")) if res_path.exists() else {
        "entorno": entorno, "semillas": SEMILLAS_DEFINITIVAS, "corridas": {}}
    for nombre in args.arquitecturas:
        if nombre not in sel:
            print(f"  {nombre}: sin configuracion seleccionada; se omite")
            continue
        cfg = sel[nombre]
        corr = resumen["corridas"].setdefault(nombre, {})
        for s in SEMILLAS_DEFINITIVAS:
            d = sal / nombre / f"semilla_{s}"
            if str(s) in corr and (d / "metricas_val.json").exists():
                print(f"  {nombre} semilla {s}: ya hecha (macro F1 {corr[str(s)]['macro_f1_val']:.4f})")
                continue
            a = copia_args(args, lr=cfg["lr"], batch=cfg["batch"], semilla=s,
                           epocas=args.epocas_def, paciencia=args.paciencia_def)
            inicio = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            t0 = time.time()
            modelo, mean, std, curvas, params, t_ent, mejores = entrenar_una(
                nombre, a, entren, val, cache, d, not args.sin_pesos)
            dur = time.time() - t0
            # tiempo de inferencia por imagen (validacion de la mejor epoca)
            sv = curvas[mejores["epoca"] - 1]["segundos_validacion"]
            corr[str(s)] = {
                "inicio": inicio, "fin": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "lr": cfg["lr"], "batch": cfg["batch"], "semilla": s,
                "epocas_ejecutadas": len(curvas), "mejor_epoca": mejores["epoca"],
                "detencion_temprana": len(curvas) < args.epocas_def,
                "macro_f1_val": round(mejores["metricas"]["macro_f1"], 4),
                "exactitud_val": round(mejores["metricas"]["exactitud"], 4),
                "horas": round(dur / 3600, 3), "segundos_entrenamiento": round(t_ent, 1),
                "ms_inferencia_por_imagen": round(1000 * sv / len(val), 2),
                "ram_pico_proceso_mb": max(c["ram_pico_mb"] for c in curvas),
                "tamano_modelo_mb": round((d / "modelo.pt").stat().st_size / 2 ** 20, 2),
                "parametros": params}
            resumen["fecha"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            json.dump(resumen, open(res_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
            del modelo
    # resumen por arquitectura y corrida mediana (plan 3.8)
    print(f"\n  {'Arquitectura':<24}{'Corridas':>9}{'Media F1':>10}{'DE':>8}{'Min':>8}{'Max':>8}"
          f"{'Mediana':>9}{'Horas':>8}")
    for nombre, corr in resumen["corridas"].items():
        v = sorted((c["macro_f1_val"], int(k)) for k, c in corr.items())
        if not v:
            continue
        f1 = [x for x, _ in v]
        mediana = v[(len(v) - 1) // 2][1]
        de = float(np.std(f1, ddof=1)) if len(f1) > 1 else 0.0
        resumen.setdefault("estadisticos", {})[nombre] = {
            "corridas": len(f1), "media_macro_f1_val": round(float(np.mean(f1)), 4),
            "de_macro_f1_val": round(de, 4), "min": min(f1), "max": max(f1),
            "semilla_corrida_mediana": mediana,
            "horas_totales": round(sum(c["horas"] for c in corr.values()), 2)}
        e = resumen["estadisticos"][nombre]
        print(f"  {nombre:<24}{len(f1):>9}{e['media_macro_f1_val']:>10.4f}{de:>8.4f}{min(f1):>8.4f}"
              f"{max(f1):>8.4f}{mediana:>9}{e['horas_totales']:>8.2f}")
    json.dump(resumen, open(res_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n  Resumen: {res_path}")


def tendencia(valores):
    x = np.arange(len(valores))
    return float(np.polyfit(x, valores, 1)[0]) if len(valores) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["piloto", "curva", "busqueda", "definitiva"], default="piloto")
    ap.add_argument("--arquitecturas", nargs="+", default=ARQUITECTURAS)
    ap.add_argument("--particion", default="fase3/particion")
    ap.add_argument("--depurado", default="fase3/conjunto_depurado_v2")
    ap.add_argument("--cache", default="fase4/cache_512")
    ap.add_argument("--salida", default="fase4/piloto")
    ap.add_argument("--epocas", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--weight-decay", type=float, default=0.01,
                    help="valor por defecto de AdamW en PyTorch; la ficha no fija otro")
    ap.add_argument("--factor-aumento", type=int, default=2)
    ap.add_argument("--paciencia", type=int, default=0, help="0 = sin detencion temprana (piloto)")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--hilos", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--horas-max", type=float, default=6.0, help="tiempo maximo por corrida completa")
    ap.add_argument("--sin-pesos", action="store_true", help="solo para pruebas: sin pesos de ImageNet")
    ap.add_argument("--epocas-busqueda", type=int, default=10)
    ap.add_argument("--epocas-def", type=int, default=15)
    ap.add_argument("--paciencia-def", type=int, default=5)
    ap.add_argument("--configuracion", default="fase5/busqueda/configuracion_seleccionada.json")
    ap.add_argument("--solo-gradcam", action="store_true",
                    help="repite solo Grad-CAM con los modelos ya guardados del piloto")
    args = ap.parse_args()

    torch.set_num_threads(args.hilos)
    part = Path(args.particion)
    particion = list(csv.DictReader(open(part / "particion.csv", encoding="utf-8")))
    entren = list(csv.DictReader(open(part / "piloto_entrenamiento.csv", encoding="utf-8")))
    val = [f for f in particion if f["subconjunto"] == "validacion"]
    gc_filas = list(csv.DictReader(open(part / "piloto_gradcam.csv", encoding="utf-8")))
    cajas = {(r["especie"], r["nombre_destino"]): r
             for r in csv.DictReader(open(part / "piloto_cajas.csv", encoding="utf-8"))}
    faltan = [f["nombre_destino"] for f in gc_filas if (f["especie"], f["nombre_destino"]) not in cajas]
    if faltan:
        sys.exit(f"ERROR: faltan cajas del piloto: {faltan[:5]}")
    n_entren_real = sum(1 for f in particion if f["subconjunto"] == "entrenamiento")

    import timm
    import sklearn
    try:
        import pytorch_grad_cam
        v_gc = getattr(pytorch_grad_cam, "__version__", "instalado")
    except ImportError:
        v_gc = None
    entorno = {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__,
               "scikit-learn": sklearn.__version__, "grad-cam": v_gc, "hilos": args.hilos,
               "cpu": platform.processor() or platform.machine()}

    if args.modo == "curva":
        curva_aprendizaje(args, particion, val, entorno)
        return
    if args.modo == "busqueda":
        busqueda(args, particion, val, entorno)
        return
    if args.modo == "definitiva":
        definitivas(args, particion, val, entorno)
        return

    print("=" * 66)
    print("  PRUEBA PILOTO (FASE 4)")
    print("=" * 66)
    print(f"  Entorno: {entorno}")
    print(f"  Entrenamiento piloto: {len(entren)}   validacion: {len(val)}   Grad-CAM: {len(gc_filas)}")
    print(f"  Configuracion: lr {args.lr}, batch {args.batch}, {args.epocas} epocas, "
          f"aumento x{args.factor_aumento}, AdamW wd {args.weight_decay}, semilla {args.semilla}")

    cache = Path(args.cache)
    t0 = time.time()
    n = preparar_cache(entren + val + gc_filas, Path(args.depurado), cache)
    print(f"  Cache: {n} imagenes nuevas en {time.time() - t0:.0f} s ({cache})")

    sal = Path(args.salida)
    sal.mkdir(parents=True, exist_ok=True)
    reporte = {"fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "entorno": entorno,
               "configuracion": {k: v for k, v in vars(args).items()},
               "imagenes": {"entrenamiento_piloto": len(entren), "validacion": len(val),
                            "entrenamiento_real": n_entren_real}, "arquitecturas": {}}

    if args.solo_gradcam:
        previo = json.load(open(sal / "reporte_piloto.json", encoding="utf-8"))
        for nombre in args.arquitecturas:
            r = previo["arquitecturas"].get(nombre, {})
            try:
                modelo = timm.create_model(nombre, pretrained=False, num_classes=2)
                modelo.load_state_dict(torch.load(sal / nombre / "modelo.pt", weights_only=True))
                cfg = modelo.pretrained_cfg
                r["gradcam"] = gradcam_indicadores(nombre, modelo, gc_filas, cajas, cache,
                                                   cfg["mean"], cfg["std"], sal / nombre)
                r["gradcam_ok"] = True
                r.pop("gradcam_error", None)
                print(f"  {nombre}: Grad-CAM {r['gradcam']}")
            except Exception as e:
                r["gradcam_ok"] = False
                r["gradcam_error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            previo["arquitecturas"][nombre] = r
        reporte = previo
        args.arquitecturas = []

    for nombre in args.arquitecturas:
        r = {"completo": False}
        try:
            modelo, mean, std, curvas, params, t_ent, mejores = entrenar_una(
                nombre, args, entren, val, cache, sal / nombre, not args.sin_pesos)
            perdidas = [c["perdida_entrenamiento"] for c in curvas]
            img_epoca = len(entren) * (1 + args.factor_aumento)
            img_epoca_real = n_entren_real * (1 + args.factor_aumento)
            seg_val_epoca = sum(c["segundos_validacion"] for c in curvas) / len(curvas)
            # Regla del plan: (t / imagenes piloto / epocas) x imagenes reales x 15
            horas = (t_ent / img_epoca / len(curvas)) * img_epoca_real * 15 / 3600 + seg_val_epoca * 15 / 3600
            r.update({
                "completo": True, "parametros": params, "epocas": len(curvas),
                "segundos_entrenamiento_piloto": round(t_ent, 1),
                "img_s_entrenamiento": round(img_epoca * len(curvas) / t_ent, 1),
                "ram_pico_mb": max(c["ram_pico_mb"] for c in curvas),
                "perdida_inicial": perdidas[0], "perdida_final": perdidas[-1],
                "pendiente_perdida": round(tendencia(perdidas), 5),
                "mejor_epoca": mejores["epoca"], "macro_f1_val": round(mejores["metricas"]["macro_f1"], 4),
                "exactitud_val": round(mejores["metricas"]["exactitud"], 4),
                "horas_estimadas_corrida_completa": round(horas, 2)})
            try:
                r["gradcam"] = gradcam_indicadores(nombre, modelo, gc_filas, cajas, cache, mean, std,
                                                   sal / nombre)
                r["gradcam_ok"] = True
            except Exception as e:
                r["gradcam_ok"] = False
                r["gradcam_error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        reporte["arquitecturas"][nombre] = r
        json.dump(reporte, open(sal / "reporte_piloto.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # ---------- criterios de aceptacion ----------
    print("\n" + "=" * 66)
    print("  CRITERIOS DE ACEPTACION (plan 3.7, Fase 4; protocolo, 6)")
    print("=" * 66)
    print(f"  {'Arquitectura':<24}{'Sin error':>10}{'Perdida':>10}{'Metricas':>10}{'Grad-CAM':>10}"
          f"{'h/corrida':>11}{'Picos fuera':>13}")
    total_h = 0.0
    for nombre, r in reporte["arquitecturas"].items():
        ok_err = r.get("completo", False)
        ok_perd = ok_err and r["pendiente_perdida"] < 0 and r["perdida_final"] < r["perdida_inicial"]
        ok_met = ok_err and "macro_f1_val" in r
        ok_gc = r.get("gradcam_ok", False)
        h = r.get("horas_estimadas_corrida_completa")
        ok_h = h is not None and h <= args.horas_max
        pf = (r.get("gradcam") or {}).get("pct_picos_fuera")
        r["criterios"] = {"sin_errores": ok_err, "perdida_decreciente": ok_perd, "metricas_calculadas": ok_met,
                          "gradcam_sin_fallos": ok_gc, "tiempo_corrida_admisible": ok_h,
                          "gradcam_alerta_piloto": pf is not None and pf > 50}
        total_h += h or 0
        si = lambda b: "si" if b else "NO"
        print(f"  {nombre:<24}{si(ok_err):>10}{si(ok_perd):>10}{si(ok_met):>10}{si(ok_gc):>10}"
              f"{(f'{h:.2f}' if h is not None else '-'):>11}{(f'{pf:.0f} %' if pf is not None else '-'):>13}")
    # presupuesto de la Fase 5: 5 corridas por arquitectura + 3 busquedas al 30 % y 10 epocas
    busq = sum(3 * (r.get("horas_estimadas_corrida_completa") or 0) * 0.30 * 10 / 15
               for r in reporte["arquitecturas"].values())
    fase5 = 5 * total_h + busq
    reporte["presupuesto_fase5_horas"] = {"corridas_definitivas": round(5 * total_h, 1),
                                          "busqueda_hiperparametros": round(busq, 1),
                                          "total": round(fase5, 1), "disponible": 240.0}
    print(f"\n  Presupuesto estimado de la Fase 5: {5 * total_h:.1f} h de corridas definitivas + "
          f"{busq:.1f} h de busqueda = {fase5:.1f} h de 240 h disponibles")
    print("  Alerta de Grad-CAM en el piloto (> 50 % de picos fuera): "
          + (", ".join(n for n, r in reporte["arquitecturas"].items() if r["criterios"]["gradcam_alerta_piloto"])
             or "ninguna"))
    json.dump(reporte, open(sal / "reporte_piloto.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n  Reporte: {sal / 'reporte_piloto.json'}")


if __name__ == "__main__":
    main()
