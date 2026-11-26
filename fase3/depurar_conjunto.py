"""
============================================================
 FASE 3 - DEPURACION DEL CONJUNTO BRUTO
============================================================
 Comparacion de arquitecturas de aprendizaje profundo para
 la clasificacion de mosquitos vectores del dengue
 Leandro Steven Batioja Mena - PUCE Esmeraldas 2026-2027

 Aplica los criterios de exclusion E1 a E9 en el orden fijado
 por el protocolo de criterios de depuracion (version 2).

   E1  Archivo corrupto, ilegible o truncado (sin tolerancia)
   E2  Resolucion inferior a 100 x 100 px
   E3  Duplicado exacto (MD5)
   E4  Duplicado perceptual (pHash, similitud > 90 %)
   E5  Exposicion deficiente (rango dinamico p99 - p1 < 40)
   E6  Desenfoque (varianza del Laplaciano < 10 a 512 px)
   E7  Estadio no adulto          -> controlado en la recopilacion
   E8  Etiqueta ambigua o ausente -> duplicados entre especies
   E9  Licencia incompatible      -> controlado en la recopilacion

 DECISIONES
 ----------
 - Cada imagen se asigna al PRIMER criterio que la excluye, de
   modo que los conteos por criterio son mutuamente excluyentes.
 - Los duplicados se agrupan en componentes. En cada componente
   se conserva un representante: el que supera E5 y E6, con mayor
   resolucion y varianza; los demas se excluyen por E3 o E4.
 - Si un componente contiene imagenes de ambas especies, la
   etiqueta es indeterminable y se excluye completo por E8.
 - E4 en dos etapas: el pHash de 256 bits propone candidatos
   (similitud > 90 %, es decir distancia de Hamming <= 25) y una
   verificacion por correlacion de pixeles (>= 0.98 sobre
   miniaturas de 64x64 normalizadas) confirma el duplicado.
   La verificacion es necesaria porque el pHash agrupa imagenes
   de laboratorio que comparten fondo, encuadre e iluminacion
   aunque el especimen sea distinto.
 - E8 se declara unicamente ante duplicados EXACTOS (MD5) entre
   especies. Una similitud perceptual entre especies no basta
   para afirmar que la etiqueta sea ambigua.
 - E5 se evalua por el rango dinamico (percentil 99 menos
   percentil 1 de la luminancia) y no por el brillo medio. El
   promedio global mide sobre todo el fondo: una fotografia de
   microscopio correctamente expuesta sobre fondo blanco supera
   235 de media aunque el especimen conserve todo su detalle.
   Una imagen realmente sobreexpuesta o subexpuesta pierde el
   rango tonal, que es lo que este criterio detecta.
 - El conjunto depurado se construye con enlaces duros.

 USO
 ---
   python depurar_conjunto.py --bruto conjunto_bruto \
                              --destino conjunto_depurado --simular
   python depurar_conjunto.py --bruto conjunto_bruto \
                              --destino conjunto_depurado
============================================================
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict, Counter
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageFile

# E1 estricto: un archivo truncado debe fallar, no rellenarse con gris
ImageFile.LOAD_TRUNCATED_IMAGES = False

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------- umbrales del protocolo (version 2) ----------
RES_MIN = 100
HAMMING_MAX = 25          # sobre 256 bits: similitud > 90 %
PHASH_TAM = 16            # hash_size=16 -> 256 bits
CORRELACION_MIN = 0.98    # verificacion de duplicado perceptual
MINIATURA = 64
RANGO_MIN = 40.0          # rango dinamico minimo (p99 - p1)
LAPLACIANO_MIN = 10.0
LADO_NORMALIZACION = 512
UMBRAL_M1 = 70.0

CRITERIOS = {
    "E1": "Archivo corrupto, ilegible o truncado",
    "E2": f"Resolucion inferior a {RES_MIN} x {RES_MIN} px",
    "E3": "Duplicado exacto (MD5)",
    "E4": "Duplicado perceptual (pHash > 90 %)",
    "E5": f"Exposicion deficiente (rango dinamico < {RANGO_MIN:.0f})",
    "E6": f"Desenfoque (varianza del Laplaciano < {LAPLACIANO_MIN:.0f})",
    "E7": "Estadio no adulto",
    "E8": "Etiqueta ambigua o ausente",
    "E9": "Licencia incompatible",
}


def medir(args):
    """Verificacion de integridad y calculo de todas las metricas."""
    ruta, especie = args
    nombre = Path(ruta).name
    origen = nombre.split("__", 1)[0] if "__" in nombre else "desconocido"
    r = {"archivo": nombre, "ruta": ruta, "especie": especie, "origen": origen,
         "ancho": "", "alto": "", "md5": "", "phash": "", "brillo": "",
         "varianza": "", "rango": "", "mini": b"", "error": ""}

    # ---- E1: integridad estricta ----
    try:
        with Image.open(ruta) as im:
            im.load()
            rgb = im.convert("RGB")
    except Exception as e:
        r["error"] = f"E1: {str(e)[:100]}"
        return r

    ancho, alto = rgb.size
    r["ancho"], r["alto"] = ancho, alto

    # ---- MD5 sobre los bytes del archivo ----
    h = hashlib.md5()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    r["md5"] = h.hexdigest()

    # ---- pHash (256 bits) ----
    r["phash"] = str(imagehash.phash(rgb, hash_size=PHASH_TAM))

    # ---- Brillo y varianza del Laplaciano ----
    # Mismo procedimiento que analizar_nitidez.py (v2) para que el
    # umbral determinado alli se aplique sobre valores comparables.
    img = cv2.imread(ruta, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r["brillo"] = round(float(gris.mean()), 2)
    p1, p99 = np.percentile(gris, [1, 99])
    r["rango"] = round(float(p99 - p1), 2)
    a, b = gris.shape[:2]
    if max(a, b) > LADO_NORMALIZACION:
        f = LADO_NORMALIZACION / max(a, b)
        gris = cv2.resize(gris, (int(b * f), int(a * f)), interpolation=cv2.INTER_AREA)
    r["varianza"] = round(float(cv2.Laplacian(gris, cv2.CV_64F).var()), 3)

    # ---- Miniatura para la verificacion de duplicados ----
    mini = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                      (MINIATURA, MINIATURA), interpolation=cv2.INTER_AREA)
    r["mini"] = mini.astype(np.uint8).tobytes()
    return r


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def pares_similares(hashes, maximo, bloque=128):
    """Pares (i, j), i < j, con distancia de Hamming <= maximo (256 bits)."""
    n = len(hashes)
    palabras = len(hashes[0]) // 16          # 64 hex -> 4 palabras de 64 bits
    h = np.array([[int(x[k * 16:(k + 1) * 16], 16) for k in range(palabras)]
                  for x in hashes], dtype=np.uint64)
    tabla = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    pares = []
    for ini in range(0, n, bloque):
        fin = min(ini + bloque, n)
        x = h[ini:fin, None, :] ^ h[None, :, :]
        dist = tabla[x.view(np.uint8)].reshape(fin - ini, n, palabras * 8).sum(axis=2)
        ii, jj = np.nonzero(dist <= maximo)
        ii = ii + ini
        m = jj > ii
        pares.extend(zip(ii[m].tolist(), jj[m].tolist()))
    return pares


def correlacion(a, b):
    """Correlacion de Pearson entre dos miniaturas normalizadas."""
    x = np.frombuffer(a, dtype=np.uint8).astype(np.float64)
    y = np.frombuffer(b, dtype=np.uint8).astype(np.float64)
    x = (x - x.mean()) / (x.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)
    return float((x * y).mean())


def supera_calidad(r):
    return r["rango"] >= RANGO_MIN and r["varianza"] >= LAPLACIANO_MIN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bruto", required=True)
    ap.add_argument("--destino", required=True)
    ap.add_argument("--salida", default="depuracion")
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--simular", action="store_true", help="no escribe el conjunto depurado")
    args = ap.parse_args()

    bruto = Path(args.bruto).expanduser()
    destino = Path(args.destino).expanduser()
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    tareas = []
    for d in sorted(p for p in bruto.iterdir() if p.is_dir()):
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in EXT:
                tareas.append((str(f), d.name))

    print("=" * 70)
    print("  DEPURACION DEL CONJUNTO BRUTO")
    print("=" * 70)
    print(f"  Conjunto bruto : {bruto}")
    print(f"  Destino        : {destino}")
    print(f"  Modo           : {'SIMULACION' if args.simular else 'ENLACES DUROS'}")
    print(f"  Imagenes       : {len(tareas)}")
    print(f"  Procesos       : {args.procesos}")
    print("\n  [1/3] Verificacion de integridad y calculo de metricas...")

    t0 = datetime.now()
    reg = []
    with Pool(args.procesos) as pool:
        for i, r in enumerate(pool.imap_unordered(medir, tareas, chunksize=32), 1):
            reg.append(r)
            if i % 2000 == 0:
                print(f"        {i}/{len(tareas)}")
    reg.sort(key=lambda r: (r["especie"], r["archivo"]))

    estado = {}      # indice -> (criterio, detalle)

    # ---------- E1 y E2 ----------
    for i, r in enumerate(reg):
        if r["error"]:
            estado[i] = ("E1", r["error"])
        elif r["ancho"] < RES_MIN or r["alto"] < RES_MIN:
            estado[i] = ("E2", f"{r['ancho']}x{r['alto']}")

    vivos = [i for i in range(len(reg)) if i not in estado]

    # ---------- E3, E4 y E8: componentes de duplicados ----------
    print("  [2/3] Deteccion de duplicados (MD5 y pHash con verificacion)...")
    uf = UnionFind(len(vivos))
    por_md5 = defaultdict(list)
    for k, i in enumerate(vivos):
        por_md5[reg[i]["md5"]].append(k)
    exactos = set()
    for grupo in por_md5.values():
        for k in grupo[1:]:
            uf.union(grupo[0], k)
            exactos.add(k)
            exactos.add(grupo[0])

    candidatos = pares_similares([reg[i]["phash"] for i in vivos], HAMMING_MAX)
    confirmados = rechazados = 0
    for a, b in candidatos:
        ia, ib = vivos[a], vivos[b]
        if reg[ia]["md5"] == reg[ib]["md5"]:
            confirmados += 1
            continue
        if correlacion(reg[ia]["mini"], reg[ib]["mini"]) >= CORRELACION_MIN:
            uf.union(a, b)
            confirmados += 1
        else:
            rechazados += 1
    print(f"        candidatos pHash: {len(candidatos)}   "
          f"confirmados: {confirmados}   rechazados por correlacion: {rechazados}")

    componentes = defaultdict(list)
    for k, i in enumerate(vivos):
        componentes[uf.find(k)].append(i)

    n_componentes_dup = 0
    conflictos = []
    for miembros in componentes.values():
        if len(miembros) == 1:
            continue
        n_componentes_dup += 1
        cuenta_md5 = Counter(reg[i]["md5"] for i in miembros)

        # E8 solo ante duplicado EXACTO entre especies
        conflicto = False
        for md5, veces in cuenta_md5.items():
            if veces > 1:
                esp = {reg[i]["especie"] for i in miembros if reg[i]["md5"] == md5}
                if len(esp) > 1:
                    conflicto = True
                    iguales = [i for i in miembros if reg[i]["md5"] == md5]
                    nombres = ", ".join(reg[i]["archivo"] for i in iguales)
                    for i in iguales:
                        estado[i] = ("E8", f"duplicado exacto entre especies: {nombres}"[:300])
                    conflictos.append([reg[i]["archivo"] + " (" + reg[i]["especie"] + ")"
                                       for i in iguales])
        miembros = [i for i in miembros if i not in estado]
        if len(miembros) <= 1:
            continue

        # Representante: primero el que supera E5 y E6; entre ellos, el de
        # mayor resolucion (mas informacion), luego mayor varianza y, por
        # ultimo, el nombre mas corto, que suele ser el archivo original.
        rep = sorted(miembros, key=lambda i: (not supera_calidad(reg[i]),
                                              -(reg[i]["ancho"] * reg[i]["alto"]),
                                              -reg[i]["varianza"],
                                              len(reg[i]["archivo"]),
                                              reg[i]["archivo"]))[0]
        for i in miembros:
            if i == rep:
                continue
            crit = "E3" if cuenta_md5[reg[i]["md5"]] > 1 else "E4"
            estado[i] = (crit, f"conservada: {reg[rep]['archivo']}")

    # ---------- E5 y E6 ----------
    for i in range(len(reg)):
        if i in estado:
            continue
        r = reg[i]
        if r["rango"] < RANGO_MIN:
            estado[i] = ("E5", f"rango {r['rango']} (media {r['brillo']})")
        elif r["varianza"] < LAPLACIANO_MIN:
            estado[i] = ("E6", f"varianza {r['varianza']}")

    aceptadas = [i for i in range(len(reg)) if i not in estado]
    dur = (datetime.now() - t0).total_seconds()

    # ---------- RESUMEN ----------
    especies = sorted({r["especie"] for r in reg})
    conteo = {c: Counter() for c in CRITERIOS}
    for i, (c, _) in estado.items():
        conteo[c][reg[i]["especie"]] += 1
    total_esp = Counter(r["especie"] for r in reg)
    acept_esp = Counter(reg[i]["especie"] for i in aceptadas)

    print("  [3/3] Resultados")
    print("\n" + "=" * 70)
    print("  IMAGENES ELIMINADAS POR CRITERIO")
    print("=" * 70)
    print(f"  {'Criterio':<48}" + "".join(f"{e.replace('aedes_', ''):>11}" for e in especies))
    print("  " + "-" * (48 + 11 * len(especies)))
    for c, desc in CRITERIOS.items():
        fila = f"  {c} {desc[:44]:<45}"
        for e in especies:
            fila += f"{conteo[c][e]:>11}"
        print(fila)
    print("  " + "-" * (48 + 11 * len(especies)))
    fila = f"  {'Total eliminadas':<48}"
    for e in especies:
        fila += f"{total_esp[e] - acept_esp[e]:>11}"
    print(fila)
    fila = f"  {'TOTAL ACEPTADAS':<48}"
    for e in especies:
        fila += f"{acept_esp[e]:>11}"
    print(fila)
    fila = f"  {'Porcentaje aceptado':<48}"
    for e in especies:
        fila += f"{100 * acept_esp[e] / total_esp[e]:>10.1f}%"
    print(fila)

    print("\n  E7 y E9: controlados en la recopilacion (seleccion de conjuntos de")
    print("  ejemplares adultos y verificacion de licencias por repositorio).")
    print(f"\n  Componentes de duplicados : {n_componentes_dup}")
    print(f"  Conflictos entre especies : {len(conflictos)}")
    for c in conflictos[:10]:
        print(f"    - {' | '.join(c)[:100]}")

    # ---------- M1 sobre el conjunto depurado ----------
    cruz = defaultdict(Counter)
    for i in aceptadas:
        cruz[reg[i]["especie"]][reg[i]["origen"]] += 1
    origenes = sorted({reg[i]["origen"] for i in aceptadas})
    print("\n" + "=" * 70)
    print("  CONJUNTO DEPURADO: ESPECIE x REPOSITORIO")
    print("=" * 70)
    print(f"  {'Repositorio':<18}" + "".join(f"{e.replace('aedes_', ''):>18}" for e in especies))
    for o in origenes:
        fila = f"  {o:<18}"
        for e in especies:
            n = cruz[e][o]
            t = sum(cruz[e].values())
            fila += f"{f'{n} ({100 * n / t:.1f}%)' if t else '0':>18}"
        print(fila)
    print(f"  {'TOTAL':<18}" + "".join(f"{sum(cruz[e].values()):>18}" for e in especies))

    minima = min(acept_esp.values()) if acept_esp else 0
    print("\n" + "=" * 70)
    print("  UMBRAL MINIMO DE MUESTRA (1.000 por especie)")
    print("=" * 70)
    for e in especies:
        est = "CUMPLE" if acept_esp[e] >= 1000 else "NO CUMPLE"
        print(f"  {e:<20} {acept_esp[e]:>7}   {est}")
    print(f"\n  Tamano de balanceo previsto: {minima} por especie")
    print(f"  Duracion: {dur / 60:.1f} min")

    # ---------- ESCRITURA ----------
    if not args.simular:
        for e in especies:
            (destino / e).mkdir(parents=True, exist_ok=True)
        enl = cop = 0
        for i in aceptadas:
            src = Path(reg[i]["ruta"])
            dst = destino / reg[i]["especie"] / src.name
            if dst.exists():
                continue
            try:
                os.link(src, dst)
                enl += 1
            except OSError:
                shutil.copy2(src, dst)
                cop += 1
        print(f"\n  Conjunto depurado escrito: {enl} enlaces duros, {cop} copias")

    # ---------- REGISTROS ----------
    with open(salida / "registro_imagenes_eliminadas.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["archivo", "especie", "origen", "criterio", "descripcion", "detalle"])
        for i in sorted(estado, key=lambda i: (estado[i][0], reg[i]["especie"], reg[i]["archivo"])):
            c, det = estado[i]
            w.writerow([reg[i]["archivo"], reg[i]["especie"], reg[i]["origen"], c, CRITERIOS[c], det])

    campos = ["archivo", "especie", "origen", "ancho", "alto", "md5", "phash",
              "brillo", "rango", "varianza", "estado", "criterio", "error"]
    with open(salida / "metricas_depuracion.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        for i, r in enumerate(reg):
            fila = {k: r.get(k, "") for k in campos}
            fila["estado"] = "eliminada" if i in estado else "aceptada"
            fila["criterio"] = estado[i][0] if i in estado else ""
            w.writerow(fila)

    reporte = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modo": "simulacion" if args.simular else "enlaces_duros",
        "umbrales": {"resolucion_min": RES_MIN, "hamming_max_256": HAMMING_MAX,
                     "correlacion_min": CORRELACION_MIN,
                     "rango_dinamico_min": RANGO_MIN,
                     "laplaciano_min": LAPLACIANO_MIN,
                     "lado_normalizacion": LADO_NORMALIZACION},
        "total_bruto": dict(total_esp),
        "eliminadas_por_criterio": {c: dict(conteo[c]) for c in CRITERIOS},
        "aceptadas": dict(acept_esp),
        "componentes_duplicados": n_componentes_dup,
        "candidatos_phash": len(candidatos),
        "duplicados_confirmados": confirmados,
        "rechazados_por_correlacion": rechazados,
        "conflictos_entre_especies": conflictos,
        "depurado_especie_repositorio": {e: dict(cruz[e]) for e in especies},
        "tamano_balanceo_previsto": minima,
        "duracion_min": round(dur / 60, 2),
    }
    with open(salida / "reporte_depuracion.json", "w", encoding="utf-8") as fh:
        json.dump(reporte, fh, indent=2, ensure_ascii=False)

    print("\n  Registros generados:")
    print(f"    {salida}/registro_imagenes_eliminadas.csv   (instrumento 3.6.2.1)")
    print(f"    {salida}/metricas_depuracion.csv")
    print(f"    {salida}/reporte_depuracion.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
