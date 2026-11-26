"""
============================================================
 ANALISIS DE NITIDEZ - DETERMINACION EMPIRICA DEL UMBRAL
============================================================
 Comparacion de arquitecturas de aprendizaje profundo para
 la clasificacion de mosquitos vectores del dengue
 Leandro Steven Batioja Mena - PUCE Esmeraldas 2026-2027

 PROPOSITO
 ---------
 El plan declara (seccion 3.6.2.1) que el umbral de desenfoque
 se determina a partir de la distribucion de nitidez del propio
 conjunto. Este script ejecuta los pasos 1 a 3 del procedimiento
 del protocolo de criterios de depuracion (seccion 3).

 DECISIONES METODOLOGICAS
 ------------------------
 1. Normalizacion de escala: la varianza del Laplaciano depende
    de la resolucion. Antes de medir, cada imagen se reduce a
    un lado mayor de 512 px (sin ampliar las menores), de modo
    que la metrica sea comparable entre repositorios.
 2. Analisis por repositorio: las fuentes difieren en condiciones
    de captura. Un umbral que elimine desproporcionadamente una
    fuente reintroduce el sesgo que controla la medida M1.
 3. Muestra visual estratificada por repositorio y percentil.

 Se registran ademas ancho, alto y brillo medio de cada imagen
 para reutilizarlos en los criterios E2 y E5.

 USO
 ---
   python analizar_nitidez.py --raiz conjunto_bruto
   python analizar_nitidez.py --raiz conjunto_bruto --procesos 6

 ESTRUCTURA ESPERADA (generada por consolidar_conjunto.py)
 ---------------------------------------------------------
   conjunto_bruto/
       aedes_aegypti/     origen__archivo.jpg
       aedes_albopictus/  origen__archivo.jpg
============================================================
"""

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
UMBRALES = [3, 5, 10, 15, 20, 30, 50, 75, 100, 150]
CONTENCION = 30.0          # % maximo de descarte por grupo (protocolo, sec. 3)
PERCENTILES_MUESTRA = [1, 5, 10, 25, 50]
POR_PERCENTIL = 2          # imagenes por percentil y repositorio
MAS_BAJAS = 5              # imagenes de menor varianza por repositorio

LADO = 512                 # se sobrescribe desde argumentos


def cargar(ruta):
    img = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
    if img is None:
        with Image.open(ruta) as im:
            img = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    return img


def medir(args):
    """Devuelve las metricas de una imagen o un error."""
    ruta, especie, lado = args
    nombre = Path(ruta).name
    origen = nombre.split("__", 1)[0] if "__" in nombre else "desconocido"
    base = {"archivo": nombre, "ruta": ruta, "especie": especie, "origen": origen}
    try:
        img = cargar(ruta)
        alto, ancho = img.shape[:2]
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brillo = float(gris.mean())
        if lado and max(alto, ancho) > lado:
            f = lado / max(alto, ancho)
            gris = cv2.resize(gris, (int(ancho * f), int(alto * f)),
                              interpolation=cv2.INTER_AREA)
        var = float(cv2.Laplacian(gris, cv2.CV_64F).var())
        base.update({"ancho": ancho, "alto": alto,
                     "brillo_medio": round(brillo, 2),
                     "varianza_laplaciano": round(var, 3), "error": ""})
    except Exception as e:
        base.update({"ancho": "", "alto": "", "brillo_medio": "",
                     "varianza_laplaciano": "", "error": str(e)[:120]})
    return base


def distribucion(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return {}
    d = {"n": int(v.size), "minimo": float(v.min())}
    for p in [1, 5, 10, 25, 50, 75]:
        d[f"p{p}"] = float(np.percentile(v, p))
    d["maximo"] = float(v.max())
    return {k: (round(x, 2) if isinstance(x, float) else x) for k, x in d.items()}


def pct_bajo(v, u):
    v = np.asarray(v, dtype=float)
    return 100.0 * (v < u).sum() / v.size if v.size else 0.0


def tabla_distribucion(titulo, grupos):
    print("\n" + "=" * 70)
    print(f"  {titulo}")
    print("=" * 70)
    print(f"  {'Grupo':<30}{'n':>7}{'min':>8}{'p5':>8}{'p25':>8}{'p50':>9}{'max':>9}")
    print("  " + "-" * 68)
    res = {}
    for g in sorted(grupos):
        d = distribucion(grupos[g])
        res[g] = d
        print(f"  {g:<30}{d['n']:>7}{d['minimo']:>8.1f}{d['p5']:>8.1f}"
              f"{d['p25']:>8.1f}{d['p50']:>9.1f}{d['maximo']:>9.1f}")
    return res


def tabla_simulacion(titulo, grupos):
    print("\n" + "=" * 70)
    print(f"  {titulo}")
    print(f"  (% descartado; * = supera la regla de contencion del {CONTENCION:.0f} %)")
    print("=" * 70)
    nombres = sorted(grupos)
    cab = f"  {'Umbral':>7}" + "".join(f"{n[:13]:>14}" for n in nombres)
    print(cab)
    print("  " + "-" * (len(cab) - 2))
    sim = {}
    for u in UMBRALES:
        fila = f"  {u:>7}"
        sim[str(u)] = {}
        for n in nombres:
            p = pct_bajo(grupos[n], u)
            sim[str(u)][n] = round(p, 1)
            marca = "*" if p > CONTENCION else " "
            fila += f"{p:>12.1f}%{marca}"
        print(fila)
    return sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", required=True, help="carpeta conjunto_bruto")
    ap.add_argument("--salida", default="analisis_nitidez")
    ap.add_argument("--lado", type=int, default=512,
                    help="lado mayor para normalizar escala (0 = sin normalizar)")
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    raiz = Path(args.raiz).expanduser()
    if not raiz.is_dir():
        print(f"ERROR: no existe {raiz}")
        return
    salida = Path(args.salida)
    dir_muestra = salida / "muestra_visual"
    if dir_muestra.exists():
        shutil.rmtree(dir_muestra)
    dir_muestra.mkdir(parents=True)

    tareas = []
    for dir_esp in sorted(p for p in raiz.iterdir() if p.is_dir()):
        for f in sorted(dir_esp.iterdir()):
            if f.is_file() and f.suffix.lower() in EXT:
                tareas.append((str(f), dir_esp.name, args.lado))

    print("=" * 70)
    print("  ANALISIS DE NITIDEZ DEL CONJUNTO BRUTO")
    print("=" * 70)
    print(f"  Conjunto        : {raiz}")
    print(f"  Imagenes        : {len(tareas)}")
    print(f"  Normalizacion   : lado mayor {args.lado} px" if args.lado else
          "  Normalizacion   : desactivada")
    print(f"  Procesos        : {args.procesos}")
    print("\n  Midiendo...")

    inicio = datetime.now()
    reg = []
    with Pool(args.procesos) as pool:
        for i, r in enumerate(pool.imap_unordered(medir, tareas, chunksize=32), 1):
            reg.append(r)
            if i % 1000 == 0:
                print(f"    {i}/{len(tareas)}")
    dur = (datetime.now() - inicio).total_seconds()

    ok = [r for r in reg if not r["error"]]
    malas = [r for r in reg if r["error"]]
    print(f"  Procesadas : {len(ok)}   Ilegibles (E1): {len(malas)}   "
          f"Tiempo: {dur/60:.1f} min")
    if not ok:
        print("  No se pudo procesar ninguna imagen.")
        return

    por_esp = defaultdict(list)
    por_org = defaultdict(list)
    por_cruz = defaultdict(list)
    for r in ok:
        v = r["varianza_laplaciano"]
        por_esp[r["especie"]].append(v)
        por_org[r["origen"]].append(v)
        por_cruz[f"{r['especie'].replace('aedes_', '')}/{r['origen']}"].append(v)

    dist = {
        "global": distribucion([r["varianza_laplaciano"] for r in ok]),
        "por_especie": tabla_distribucion("DISTRIBUCION POR ESPECIE", por_esp),
        "por_repositorio": tabla_distribucion("DISTRIBUCION POR REPOSITORIO", por_org),
        "por_especie_y_repositorio": tabla_distribucion(
            "DISTRIBUCION POR ESPECIE Y REPOSITORIO", por_cruz),
    }
    sim = {
        "por_especie": tabla_simulacion("SIMULACION DE UMBRALES POR ESPECIE", por_esp),
        "por_repositorio": tabla_simulacion("SIMULACION DE UMBRALES POR REPOSITORIO", por_org),
    }

    # Umbral maximo que respeta la contencion en todo grupo
    admisible = None
    for u in UMBRALES:
        grupos = list(por_esp.values()) + list(por_org.values())
        if all(pct_bajo(g, u) <= CONTENCION for g in grupos):
            admisible = u
    print("\n" + "=" * 70)
    print("  UMBRAL MAXIMO QUE RESPETA LA REGLA DE CONTENCION")
    print("=" * 70)
    if admisible is None:
        print("  Ningun umbral candidato respeta la contencion en todos los grupos.")
    else:
        print(f"  {admisible}  (ningun repositorio ni especie pierde mas del "
              f"{CONTENCION:.0f} %)")
    print("  Es un TECHO, no una recomendacion: el umbral definitivo surge de")
    print("  la inspeccion visual y puede ser menor.")

    # ---------- MUESTRA VISUAL ESTRATIFICADA ----------
    print("\n" + "=" * 70)
    print("  MUESTRA VISUAL ESTRATIFICADA")
    print("=" * 70)
    copiadas = 0
    for org in sorted(por_org):
        filas = sorted((r for r in ok if r["origen"] == org),
                       key=lambda r: r["varianza_laplaciano"])
        d = dir_muestra / org
        d.mkdir(parents=True, exist_ok=True)
        elegidas = [("min", r) for r in filas[:MAS_BAJAS]]
        vals = np.array([r["varianza_laplaciano"] for r in filas])
        for p in PERCENTILES_MUESTRA:
            obj = np.percentile(vals, p)
            idx = int(np.argmin(np.abs(vals - obj)))
            for r in filas[idx: idx + POR_PERCENTIL]:
                elegidas.append((f"p{p:02d}", r))
        vistos = set()
        for etiqueta, r in elegidas:
            if r["ruta"] in vistos:
                continue
            vistos.add(r["ruta"])
            dst = d / (f"var{r['varianza_laplaciano']:08.2f}_{etiqueta}_"
                       f"{r['especie'].replace('aedes_', '')}_{r['archivo']}")
            try:
                shutil.copy2(r["ruta"], dst)
                copiadas += 1
            except Exception:
                pass
        print(f"  {org:<18} {len(vistos):>3} imagenes -> {d}")
    print(f"  Total copiadas: {copiadas}  (copias reales, no enlaces: pueden abrirse y borrarse)")

    # ---------- GUARDAR ----------
    campos = ["archivo", "especie", "origen", "ancho", "alto", "brillo_medio",
              "varianza_laplaciano", "error", "ruta"]
    csv_path = salida / "metricas_por_imagen.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        for r in sorted(reg, key=lambda r: (r["especie"], r["origen"], r["archivo"])):
            w.writerow({k: r.get(k, "") for k in campos})

    reporte = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raiz": str(raiz),
        "normalizacion_lado_px": args.lado,
        "procesadas": len(ok),
        "ilegibles": len(malas),
        "archivos_ilegibles": [r["archivo"] for r in malas],
        "duracion_min": round(dur / 60, 2),
        "distribuciones": dist,
        "simulacion_umbrales": sim,
        "regla_contencion_pct": CONTENCION,
        "umbral_maximo_admisible": admisible,
    }
    json_path = salida / "reporte_nitidez.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(reporte, fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("  SIGUIENTE PASO (MANUAL)")
    print("=" * 70)
    print(f"  1. Abra {dir_muestra}/ ; hay una carpeta por repositorio.")
    print("     Cada nombre empieza por su varianza, asi quedan ordenadas.")
    print("  2. En cada repositorio identifique desde que varianza el")
    print("     desenfoque impide ver escudo toracico, patas o abdomen.")
    print("  3. Adopte como umbral el valor mas bajo que elimine desenfoque")
    print(f"     real sin superar el techo de contencion ({admisible}).")
    print("  4. Registre umbral y justificacion en la Entrada 006.")
    print(f"\n  Archivos: {csv_path}\n            {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
