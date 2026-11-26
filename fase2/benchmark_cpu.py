"""
============================================================
 FASE 2 - BENCHMARK DE RENDIMIENTO EN CPU
============================================================
 Comparacion de arquitecturas de aprendizaje profundo para
 la clasificacion de mosquitos vectores del dengue
 Leandro Steven Batioja Mena - PUCE Esmeraldas 2026-2027

 PROPOSITO
 ---------
 La Tabla 2 del plan estima el presupuesto de computo a partir
 de un rendimiento supuesto de 12 img/s (arquitecturas ligeras)
 y 5 img/s (arquitecturas de mayor tamano). Este script MIDE el
 rendimiento real del equipo y contrasta la estimacion con datos
 empiricos, cumpliendo lo declarado en la Fase 4 del cronograma.

 QUE MIDE
 --------
   - Tiempo de inferencia por imagen (modo evaluacion)
   - Tiempo de entrenamiento por imagen (adelante + atras)
   - Memoria RAM maxima utilizada
   - Extrapolacion a horas por corrida completa
   - Comparacion frente al presupuesto declarado (240 h)

 Uso:
   python benchmark_cpu.py
   python benchmark_cpu.py --lotes 5 --batch 16
============================================================
"""

import argparse
import json
import time
import gc
from datetime import datetime

import torch
import timm
import numpy as np

# ------------------------------------------------------------
# PARAMETROS DEL PLAN (secciones 3.2, 3.3 y Tabla 2)
# ------------------------------------------------------------
ARQUITECTURAS = {
    "EfficientNet-B0":   ("efficientnet_b0",          "ligera"),
    "MobileNetV3-Large": ("mobilenetv3_large_100",    "ligera"),
    "ResNet50":          ("resnet50",                 "mayor"),
    "ViT-S/16":          ("vit_small_patch16_224",    "mayor"),
}

NUM_CLASES       = 2       # Aedes aegypti / Aedes albopictus
RESOLUCION       = 224     # pixeles (seccion 3.3)
EPOCAS_DEF       = 15      # epocas del entrenamiento definitivo (3.2)
EPOCAS_BUSQUEDA  = 10      # epocas de la busqueda de hiperparametros
IMGS_ENTRENAMIENTO = 10800 # conjunto tras aumento x2
FRACCION_BUSQUEDA  = 0.30  # submuestra para busqueda (3.2)
CORRIDAS_POR_ARQ   = 5     # corridas definitivas (3.2)
CONFIGS_POR_ARQ    = 3     # configuraciones de hiperparametros (3.2)
HORAS_DISPONIBLES  = 240   # presupuesto declarado en Fase 5

# Estimacion teorica de la Tabla 2, para contrastar
ESTIMADO_TABLA2 = {"ligera": 12.0, "mayor": 5.0}   # img/s


def memoria_mb():
    """Memoria residente del proceso, en MB."""
    try:
        with open("/proc/self/status") as f:
            for l in f:
                if l.startswith("VmRSS"):
                    return int(l.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


def medir(nombre, tid, batch, lotes, calentamiento=1):
    """Mide inferencia y entrenamiento de una arquitectura."""
    print(f"\n  {nombre}")
    print("  " + "-" * 58)

    modelo = timm.create_model(tid, pretrained=True, num_classes=NUM_CLASES)

    # Ajuste fino parcial: congelar todo salvo el clasificador (3.2)
    for p in modelo.parameters():
        p.requires_grad = False
    for p in modelo.get_classifier().parameters():
        p.requires_grad = True

    lote = torch.randn(batch, 3, RESOLUCION, RESOLUCION)
    etiquetas = torch.randint(0, NUM_CLASES, (batch,))
    criterio = torch.nn.CrossEntropyLoss()
    params = [p for p in modelo.parameters() if p.requires_grad]
    optimizador = torch.optim.AdamW(params, lr=1e-3)

    mem_inicial = memoria_mb()

    # ---------- INFERENCIA ----------
    modelo.eval()
    with torch.no_grad():
        for _ in range(calentamiento):
            modelo(lote)
    tiempos = []
    with torch.no_grad():
        for _ in range(lotes):
            t0 = time.perf_counter()
            modelo(lote)
            tiempos.append(time.perf_counter() - t0)
    t_inf = float(np.median(tiempos))
    ips_inf = batch / t_inf
    ms_img = 1000 * t_inf / batch
    print(f"    Inferencia      : {ips_inf:>7.1f} img/s   ({ms_img:.1f} ms/img)")

    # ---------- ENTRENAMIENTO ----------
    modelo.train()
    for _ in range(calentamiento):
        optimizador.zero_grad()
        criterio(modelo(lote), etiquetas).backward()
        optimizador.step()
    tiempos = []
    for _ in range(lotes):
        t0 = time.perf_counter()
        optimizador.zero_grad()
        criterio(modelo(lote), etiquetas).backward()
        optimizador.step()
        tiempos.append(time.perf_counter() - t0)
    t_ent = float(np.median(tiempos))
    ips_ent = batch / t_ent
    print(f"    Entrenamiento   : {ips_ent:>7.1f} img/s   ({1000*t_ent/batch:.1f} ms/img)")

    mem_pico = memoria_mb()
    print(f"    Memoria RAM     : {mem_pico:>7.0f} MB    (delta {mem_pico-mem_inicial:+.0f} MB)")

    n_params = sum(p.numel() for p in modelo.parameters())

    del modelo, optimizador
    gc.collect()

    return {
        "id_timm": tid,
        "parametros_millones": round(n_params / 1e6, 2),
        "img_s_inferencia": round(ips_inf, 2),
        "ms_por_imagen_inferencia": round(ms_img, 2),
        "img_s_entrenamiento": round(ips_ent, 2),
        "memoria_pico_mb": round(mem_pico, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16, help="tamano de lote (default 16)")
    ap.add_argument("--lotes", type=int, default=5, help="lotes a medir por arquitectura")
    args = ap.parse_args()

    print("=" * 62)
    print("  BENCHMARK DE RENDIMIENTO EN CPU")
    print("  Fase 2 / Verificacion para la Fase 4 (prueba piloto)")
    print("=" * 62)

    # CPU y RAM del equipo
    cpu = "no detectado"
    try:
        with open("/proc/cpuinfo") as f:
            for l in f:
                if "model name" in l:
                    cpu = l.split(":")[1].strip()
                    break
    except Exception:
        pass

    print(f"  CPU              : {cpu}")
    print(f"  Hilos de torch   : {torch.get_num_threads()}")
    print(f"  Resolucion       : {RESOLUCION}x{RESOLUCION}")
    print(f"  Tamano de lote   : {args.batch}")
    print(f"  Lotes medidos    : {args.lotes} (se reporta la mediana)")
    print(f"  Ajuste fino      : parcial (solo capa clasificadora)")

    print("\n" + "=" * 62)
    print("  MEDICIONES POR ARQUITECTURA")
    print("=" * 62)

    med = {}
    for nombre, (tid, clase) in ARQUITECTURAS.items():
        try:
            r = medir(nombre, tid, args.batch, args.lotes)
            r["clase"] = clase
            med[nombre] = r
        except Exception as e:
            print(f"    ERROR: {str(e)[:70]}")

    if not med:
        print("\n  No se completo ninguna medicion.")
        return

    # ---------- EXTRAPOLACION ----------
    print("\n" + "=" * 62)
    print("  EXTRAPOLACION AL PRESUPUESTO DE COMPUTO")
    print("=" * 62)
    print(f"  Conjunto de entrenamiento : {IMGS_ENTRENAMIENTO:,} imagenes")
    print(f"  Epocas definitivas        : {EPOCAS_DEF}")
    print(f"  Corridas por arquitectura : {CORRIDAS_POR_ARQ}")
    print(f"  Configuraciones busqueda  : {CONFIGS_POR_ARQ} "
          f"({int(FRACCION_BUSQUEDA*100)}% del conjunto, {EPOCAS_BUSQUEDA} epocas)")
    print()
    print(f"  {'Arquitectura':<20}{'h/corrida':>11}{'h definit.':>12}"
          f"{'h busqueda':>12}{'h total':>10}")
    print("  " + "-" * 65)

    total_general = 0.0
    imgs_busqueda = IMGS_ENTRENAMIENTO * FRACCION_BUSQUEDA

    for nombre, r in med.items():
        ips = r["img_s_entrenamiento"]
        h_corrida  = (IMGS_ENTRENAMIENTO * EPOCAS_DEF) / ips / 3600
        h_definit  = h_corrida * CORRIDAS_POR_ARQ
        h_busqueda = (imgs_busqueda * EPOCAS_BUSQUEDA) / ips / 3600 * CONFIGS_POR_ARQ
        h_total    = h_definit + h_busqueda
        total_general += h_total

        r["horas_por_corrida"]   = round(h_corrida, 2)
        r["horas_definitivas"]   = round(h_definit, 2)
        r["horas_busqueda"]      = round(h_busqueda, 2)
        r["horas_total"]         = round(h_total, 2)

        print(f"  {nombre:<20}{h_corrida:>11.2f}{h_definit:>12.1f}"
              f"{h_busqueda:>12.1f}{h_total:>10.1f}")

    print("  " + "-" * 65)
    print(f"  {'TOTAL REQUERIDO':<20}{'':>11}{'':>12}{'':>12}{total_general:>10.1f}")
    print(f"  {'DISPONIBLE (Fase 5)':<20}{'':>11}{'':>12}{'':>12}{HORAS_DISPONIBLES:>10.1f}")

    margen = HORAS_DISPONIBLES - total_general
    pct = 100 * margen / HORAS_DISPONIBLES
    print(f"  {'MARGEN':<20}{'':>11}{'':>12}{'':>12}{margen:>10.1f}  ({pct:+.0f}%)")

    # ---------- CONTRASTE CON LA TABLA 2 ----------
    print("\n" + "=" * 62)
    print("  CONTRASTE CON LA ESTIMACION DE LA TABLA 2")
    print("=" * 62)
    print(f"  {'Arquitectura':<20}{'estimado':>11}{'medido':>10}{'desviacion':>13}")
    print("  " + "-" * 56)
    for nombre, r in med.items():
        est = ESTIMADO_TABLA2[r["clase"]]
        real = r["img_s_entrenamiento"]
        desv = 100 * (real - est) / est
        print(f"  {nombre:<20}{est:>9.1f} i/s{real:>8.1f} i/s{desv:>+12.0f}%")

    # ---------- VEREDICTO ----------
    print("\n" + "=" * 62)
    print("  VEREDICTO")
    print("=" * 62)
    if total_general <= HORAS_DISPONIBLES * 0.85:
        print("  El presupuesto CIERRA con margen suficiente.")
        print("  La planificacion de la Tabla 2 se sostiene empiricamente.")
    elif total_general <= HORAS_DISPONIBLES:
        print("  El presupuesto CIERRA pero con margen ajustado (<15%).")
        print("  Considere reducir epocas o el factor de aumento de datos.")
    else:
        print("  El presupuesto NO CIERRA con los parametros actuales.")
        print(f"  Exceso: {total_general - HORAS_DISPONIBLES:.0f} horas.")
        print("  Medidas correctivas previstas en la Fase 4:")
        factor = HORAS_DISPONIBLES / total_general
        print(f"    - Reducir epocas de {EPOCAS_DEF} a "
              f"{max(5, int(EPOCAS_DEF * factor))}")
        print(f"    - O reducir la resolucion de {RESOLUCION} a 160 px (~-49% de tiempo)")
        print(f"    - O reducir el factor de aumento de datos")

    # ---------- GUARDAR ----------
    reporte = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu": cpu,
        "hilos_torch": torch.get_num_threads(),
        "resolucion": RESOLUCION,
        "batch": args.batch,
        "lotes_medidos": args.lotes,
        "parametros_plan": {
            "imgs_entrenamiento": IMGS_ENTRENAMIENTO,
            "epocas_definitivas": EPOCAS_DEF,
            "epocas_busqueda": EPOCAS_BUSQUEDA,
            "corridas_por_arquitectura": CORRIDAS_POR_ARQ,
            "configuraciones_busqueda": CONFIGS_POR_ARQ,
            "horas_disponibles": HORAS_DISPONIBLES,
        },
        "mediciones": med,
        "horas_total_requeridas": round(total_general, 2),
        "margen_horas": round(margen, 2),
        "margen_porcentaje": round(pct, 1),
    }
    with open("reporte_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    print(f"\n  Reporte guardado en: reporte_benchmark.json")
    print("=" * 62)


if __name__ == "__main__":
    main()
