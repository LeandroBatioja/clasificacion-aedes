"""
============================================================
 FASE 3 - CONJUNTO DE TRABAJO Y VERIFICACION MORFOLOGICA
============================================================
 Comparacion de arquitecturas de aprendizaje profundo para
 la clasificacion de mosquitos vectores del dengue
 Leandro Steven Batioja Mena - PUCE Esmeraldas 2026-2027

 Entrada 007 de la bitacora. Version 2: categoria "otro" (no es
 Aedes aegypti ni Ae. albopictus, p. ej. Culex) y laminas con
 acercamiento del mosquito a resolucion completa; --solo-laminas
 regenera las laminas sin cambiar la muestra.

 Tres modos:

 1) PREPARAR (por defecto)
    a. Fija el conjunto de trabajo de la opcion A (Entrada 006
       quinquies): Dryad + Ong + celular de campo en ambas
       especies. Mosquito Alert solo con reporte verificable. En
       Ae. albopictus, Mosquito Alert se submuestrea por reportes
       completos (semilla fija) hasta igualar el total de
       Ae. aegypti. Kaggle y Couret quedan fuera.
       -> fase3/conjunto_trabajo.csv
    b. Extrae una muestra aleatoria de al menos el 10 % de cada
       especie, estratificada por repositorio.
    c. Genera laminas para una revision A CIEGAS: cada imagen
       lleva solo un numero, las dos especies van mezcladas y la
       etiqueta original se guarda aparte (clave_verificacion.csv).
       El revisor anota la especie que observa en
       verificacion_morfologica.csv, sin consultar la clave.

 2) EVALUAR (--evaluar)
    Compara lo observado con la etiqueta, calcula la tasa de
    error de etiquetado sobre las imagenes determinables, por
    especie y por repositorio, y genera laminas de las
    discrepancias.

 4) EXPORTAR MUESTRA (--exportar-muestra)
    Copia las imagenes de la muestra a muestra_ciega/, renombradas
    solo con su numero (001.jpg ...) y reducidas a 1600 px de lado
    mayor, para revisarlas a mayor detalle sin conocer la etiqueta.

 3) SOLO LAMINAS (--solo-laminas)
    Regenera las laminas de una muestra ya extraida, leyendo la
    clave; no cambia la muestra ni las anotaciones.

 PROTOCOLO (criterios_depuracion.md, seccion 4)
   Clave de referencia: Gyawali et al. (2025), referencia [23]
   Rasgos: escudo toracico, bandas en las patas, coloracion abdominal
     Ae. aegypti     escudo con escamas plateadas en forma de lira
     Ae. albopictus  escudo con una sola franja plateada central
   Valores de especie_observada: aegypti, albopictus, otro (no es
   ninguna de las dos especies; cuenta como error de etiqueta) o
   indeterminable.
   Si los rasgos no se distinguen, se anota "indeterminable"; esos
   casos siguen el plan de validacion escalonado del protocolo.
   Regla de decision, por repositorio:
     <= 15 %   continuar y corregir las etiquetas discrepantes
     15-25 %   ampliar la verificacion al 20 % de ese repositorio
     >  25 %   evaluar la exclusion parcial o total del repositorio
   En la columna "rasgo" se anota el rasgo en que se basa la
   decision: escudo, patas o abdomen.

 USO (desde la raiz del proyecto)
   python fase3/verificacion_morfologica.py
   python fase3/verificacion_morfologica.py --evaluar

 SALIDAS (fase3/verificacion_morfologica/)
   laminas/hoja_XX.jpg           imagenes numeradas para revisar
   verificacion_morfologica.csv  para llenar: id, especie_observada,
                                 rasgo, nota
   clave_verificacion.csv        etiqueta, archivo y repositorio
                                 (no abrir antes de terminar)
   resultado_verificacion.json   (modo --evaluar)
   discrepancias/                (modo --evaluar)
============================================================
"""

import argparse
import numpy as np
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

AEGYPTI, ALBOPICTUS = "aedes_aegypti", "aedes_albopictus"
FUENTES_A = ("dryad", "ong", "inaturalist", "mosquito_alert")
SEMILLA = 42
TESELA = 360        # lado de cada imagen en las laminas
POR_HOJA = 12       # 4 x 3
VALIDAS = {"aegypti", "albopictus", "otro", "indeterminable"}
POR_HOJA_ZOOM = 6   # 2 x 3, cada imagen con su acercamiento
PANEL = 300         # foto completa
ZOOM = 560          # acercamiento


# ------------------------------------------------------------
def conjunto_trabajo(especimenes):
    filas = list(csv.DictReader(open(especimenes, encoding="utf-8")))
    base = [f for f in filas if f["origen"] in FUENTES_A
            and not (f["origen"] == "mosquito_alert" and f.get("reporte_verificable") != "si")]
    aeg = [f for f in base if f["especie"] == AEGYPTI]
    alb_fijo = [f for f in base if f["especie"] == ALBOPICTUS and f["origen"] != "mosquito_alert"]
    alb_ma = [f for f in base if f["especie"] == ALBOPICTUS and f["origen"] == "mosquito_alert"]
    objetivo = len(aeg) - len(alb_fijo)
    if objetivo <= 0 or objetivo > len(alb_ma):
        raise SystemExit(f"ERROR: objetivo de Mosquito Alert fuera de rango ({objetivo})")

    por_grupo = defaultdict(list)
    for f in alb_ma:
        por_grupo[f["grupo"]].append(f)
    grupos = sorted(por_grupo)
    random.Random(SEMILLA).shuffle(grupos)
    elegidas = []
    for g in grupos:
        miembros = sorted(por_grupo[g], key=lambda f: f["nombre_destino"])
        falta = objetivo - len(elegidas)
        if falta <= 0:
            break
        elegidas.extend(miembros[:falta])     # el ultimo reporte puede quedar recortado
    trabajo = aeg + alb_fijo + elegidas
    return trabajo, len({f["grupo"] for f in elegidas})


def caja_mosquito(im):
    """Localiza el mosquito (mancha oscura y compacta) y devuelve su caja en
    coordenadas de la imagen original, con margen; o None."""
    g = im.convert("L")
    w0, h0 = g.size
    g.thumbnail((256, 256))
    a = np.asarray(g, dtype=np.float32)
    D = np.asarray(g.filter(ImageFilter.GaussianBlur(10)), dtype=np.float32) - a
    mask = D >= max(float(np.percentile(D, 99.3)), 8.0)
    h, w = mask.shape
    pend = set(zip(*np.nonzero(mask)))
    mejor = None
    while pend:
        pila, comp = [pend.pop()], []
        while pila:
            y, x = pila.pop()
            comp.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (y + dy, x + dx)
                    if q in pend:
                        pend.remove(q)
                        pila.append(q)
        ys, xs = np.array(comp).T
        bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        if len(ys) < 6 or bh > 0.6 * h or bw > 0.6 * w:
            continue
        peso = float(D[ys, xs].sum())
        if mejor is None or peso > mejor[0]:
            mejor = (peso, ys, xs)
    if not mejor:
        return None
    _, ys, xs = mejor
    sx, sy = w0 / w, h0 / h
    cy, cx = (ys.min() + ys.max()) / 2 * sy, (xs.min() + xs.max()) / 2 * sx
    lado = max((ys.max() - ys.min() + 1) * sy, (xs.max() - xs.min() + 1) * sx) * 2.2
    lado = max(lado, 0.12 * min(w0, h0))
    x0, y0 = max(0, cx - lado / 2), max(0, cy - lado / 2)
    return (int(x0), int(y0), int(min(w0, x0 + lado)), int(min(h0, y0 + lado)))


def lamina_zoom(items, ruta_dep, titulo, destino):
    cols, cel_w, cel_h = 2, PANEL + ZOOM + 24, ZOOM + 34
    filas = (len(items) + cols - 1) // cols
    hoja = Image.new("RGB", (cols * cel_w, filas * cel_h + 30), "white")
    d = ImageDraw.Draw(hoja)
    d.text((8, 8), titulo, fill="black")
    for i, it in enumerate(items):
        x, y = (i % cols) * cel_w, 30 + (i // cols) * cel_h
        d.text((x + 8, y + 4), f"#{it['id']}", fill="black")
        try:
            with Image.open(ruta_dep / it["especie"] / it["nombre_destino"]) as im:
                im = im.convert("RGB")
                caja = caja_mosquito(im)
                comp = im.copy()
                comp.thumbnail((PANEL, PANEL))
                hoja.paste(comp, (x + 4, y + 20))
                if caja:
                    f = comp.size[0] / im.size[0]
                    d.rectangle([x + 4 + caja[0] * f, y + 20 + caja[1] * f,
                                 x + 4 + caja[2] * f, y + 20 + caja[3] * f], outline="red", width=2)
                    z = im.crop(caja)
                else:
                    z = im.copy()
                z.thumbnail((ZOOM, ZOOM), Image.LANCZOS)
                if max(z.size) < ZOOM:            # ampliar si el recorte es pequeno
                    k = ZOOM / max(z.size)
                    z = z.resize((int(z.size[0] * k), int(z.size[1] * k)), Image.LANCZOS)
                hoja.paste(z, (x + PANEL + 14, y + 20))
                d.text((x + 8, y + 24 + PANEL), f"{im.size[0]}x{im.size[1]} px", fill="gray")
        except Exception:
            d.text((x + 20, y + 120), "no legible", fill="red")
    hoja.save(destino, quality=92)


def generar_laminas(muestra, dep, sal):
    (sal / "laminas").mkdir(parents=True, exist_ok=True)
    for f in (sal / "laminas").glob("*.jpg"):
        f.unlink()
    n = 0
    for h, ini in enumerate(range(0, len(muestra), POR_HOJA_ZOOM), 1):
        bloque = muestra[ini:ini + POR_HOJA_ZOOM]
        lamina_zoom(bloque, dep,
                    f"Verificacion morfologica - hoja {h} - ids {bloque[0]['id']}-{bloque[-1]['id']}"
                    "  (izquierda: foto completa; derecha: acercamiento)",
                    sal / "laminas" / f"hoja_{h:02d}.jpg")
        n = h
    return n


def asignar(conteos, n):
    """Reparto proporcional por resto mayor."""
    tot = sum(conteos.values())
    base = {k: n * v // tot for k, v in conteos.items()}
    resto = sorted(conteos, key=lambda k: -(n * conteos[k] / tot - base[k]))
    for k in resto[: n - sum(base.values())]:
        base[k] += 1
    return base


def lamina(items, ruta_dep, titulo, destino, etiqueta):
    cols = 4
    filas = (len(items) + cols - 1) // cols
    hoja = Image.new("RGB", (cols * TESELA, filas * TESELA + 30), "white")
    d = ImageDraw.Draw(hoja)
    d.text((8, 8), titulo, fill="black")
    for i, it in enumerate(items):
        x, y = (i % cols) * TESELA, 30 + (i // cols) * TESELA
        try:
            with Image.open(ruta_dep / it["especie"] / it["nombre_destino"]) as im:
                im = im.convert("RGB")
                im.thumbnail((TESELA - 8, TESELA - 26))
                hoja.paste(im, (x + 4, y + 4))
        except Exception:
            d.text((x + 20, y + 120), "no legible", fill="red")
        d.rectangle([x + 4, y + TESELA - 22, x + 120, y + TESELA - 4], fill="white")
        d.text((x + 8, y + TESELA - 20), etiqueta(it), fill="black")
    hoja.save(destino, quality=90)


# ------------------------------------------------------------
def preparar(args):
    sal = Path(args.salida)
    if (sal / "verificacion_morfologica.csv").exists() and not args.forzar:
        raise SystemExit("ERROR: ya existe una muestra. Use --forzar para regenerarla "
                         "(se perderian las anotaciones).")
    (sal / "laminas").mkdir(parents=True, exist_ok=True)
    for f in (sal / "laminas").glob("*.jpg"):
        f.unlink()

    trabajo, reportes_ma = conjunto_trabajo(args.especimenes)
    with open(args.conjunto, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["especie", "nombre_destino", "origen", "grupo"])
        for f in sorted(trabajo, key=lambda f: (f["especie"], f["origen"], f["nombre_destino"])):
            w.writerow([f["especie"], f["nombre_destino"], f["origen"], f["grupo"]])

    comp = defaultdict(Counter)
    for f in trabajo:
        comp[f["especie"]][f["origen"]] += 1
    print("=" * 62)
    print("  CONJUNTO DE TRABAJO (opcion A)")
    print("=" * 62)
    print(f"  {'Repositorio':<16}{'aegypti':>10}{'albopictus':>12}")
    for o in FUENTES_A:
        print(f"  {o:<16}{comp[AEGYPTI][o]:>10}{comp[ALBOPICTUS][o]:>12}")
    print(f"  {'TOTAL':<16}{sum(comp[AEGYPTI].values()):>10}{sum(comp[ALBOPICTUS].values()):>12}")
    print(f"  Mosquito Alert albopictus: {reportes_ma} reportes completos o recortados")
    for e in (AEGYPTI, ALBOPICTUS):
        o_max, n_max = comp[e].most_common(1)[0]
        print(f"  M1 {e}: {o_max} {100 * n_max / sum(comp[e].values()):.1f} %")

    # muestra estratificada
    rnd = random.Random(SEMILLA)
    muestra = []
    asignacion = {}
    for e in (AEGYPTI, ALBOPICTUS):
        n = max(args.n, -(-sum(comp[e].values()) // 10))      # al menos el 10 %
        asignacion[e] = asignar(comp[e], n)
        for o, k in asignacion[e].items():
            pool = sorted((f for f in trabajo if f["especie"] == e and f["origen"] == o),
                          key=lambda f: f["nombre_destino"])
            muestra.extend(rnd.sample(pool, k))
    rnd.shuffle(muestra)
    for i, f in enumerate(muestra, 1):
        f["id"] = i

    with open(sal / "clave_verificacion.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "especie_etiqueta", "origen", "grupo", "nombre_destino"])
        for f in muestra:
            w.writerow([f["id"], f["especie"].replace("aedes_", ""), f["origen"], f["grupo"],
                        f["nombre_destino"]])
    with open(sal / "verificacion_morfologica.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "especie_observada", "rasgo", "nota"])
        for f in muestra:
            w.writerow([f["id"], "", "", ""])

    hojas = generar_laminas(muestra, Path(args.depurado), sal)

    print("\n  MUESTRA (estratificada por repositorio)")
    for e in (AEGYPTI, ALBOPICTUS):
        n = sum(asignacion[e].values())
        print(f"  {e}: {n} ({100 * n / sum(comp[e].values()):.1f} %)  {dict(asignacion[e])}")
    print(f"\n  Laminas: {sal / 'laminas'} ({hojas} hojas)")
    print(f"  Llenar:  {sal / 'verificacion_morfologica.csv'}")
    print(f"  Conjunto de trabajo: {args.conjunto}")

    json.dump({"fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "semilla": SEMILLA,
               "conjunto_trabajo": {e: dict(comp[e]) for e in comp},
               "reportes_ma_albopictus": reportes_ma,
               "muestra": {e: dict(asignacion[e]) for e in asignacion}},
              open(sal / "preparacion.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)


# ------------------------------------------------------------
def solo_laminas(args):
    sal = Path(args.salida)
    muestra = []
    for r in csv.DictReader(open(sal / "clave_verificacion.csv", encoding="utf-8")):
        muestra.append({"id": int(r["id"]), "especie": "aedes_" + r["especie_etiqueta"],
                        "nombre_destino": r["nombre_destino"]})
    muestra.sort(key=lambda f: f["id"])
    hojas = generar_laminas(muestra, Path(args.depurado), sal)
    print(f"  Laminas regeneradas: {hojas} hojas en {sal / 'laminas'} (muestra sin cambios)")


# ------------------------------------------------------------
def exportar_muestra(args):
    sal = Path(args.salida)
    dest = sal / "muestra_ciega"
    dest.mkdir(parents=True, exist_ok=True)
    for f in dest.glob("*.jpg"):
        f.unlink()
    dep = Path(args.depurado)
    n = 0
    for r in csv.DictReader(open(sal / "clave_verificacion.csv", encoding="utf-8")):
        src = dep / ("aedes_" + r["especie_etiqueta"]) / r["nombre_destino"]
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail((1600, 1600), Image.LANCZOS)
                im.save(dest / f"{int(r['id']):03d}.jpg", quality=90)
                n += 1
        except Exception as e:
            print(f"  aviso: #{r['id']} no legible ({type(e).__name__})")
    print(f"  Exportadas {n} imagenes a {dest} (nombre = numero; sin etiqueta)")


# ------------------------------------------------------------
def evaluar(args):
    sal = Path(args.salida)
    clave = {r["id"]: r for r in csv.DictReader(open(sal / "clave_verificacion.csv", encoding="utf-8"))}
    obs = {r["id"]: r for r in csv.DictReader(open(sal / "verificacion_morfologica.csv", encoding="utf-8"))}
    vacios = [i for i, r in obs.items() if not r["especie_observada"].strip()]
    invalidos = [i for i, r in obs.items()
                 if r["especie_observada"].strip() and r["especie_observada"].strip().lower() not in VALIDAS]
    if vacios or invalidos:
        print(f"  Sin anotar: {len(vacios)}   Valores no validos: {invalidos[:20]}")
        print(f"  Valores validos: {sorted(VALIDAS)}")
        return

    est = defaultdict(lambda: Counter())
    disc = []
    for i, c in clave.items():
        o = obs[i]["especie_observada"].strip().lower()
        for k in (c["especie_etiqueta"], f"{c['especie_etiqueta']}|{c['origen']}"):
            est[k]["revisadas"] += 1
            if o == "indeterminable":
                est[k]["indeterminables"] += 1
            elif o == c["especie_etiqueta"]:
                est[k]["coinciden"] += 1
            else:
                est[k]["discrepan"] += 1
        if o not in ("indeterminable", c["especie_etiqueta"]):
            disc.append({**c, "observada": o, "nota": obs[i]["nota"]})

    res = {}
    print("=" * 70)
    print("  VERIFICACION MORFOLOGICA DE ETIQUETAS")
    print("=" * 70)
    print(f"  {'Grupo':<28}{'Revis.':>8}{'Indet.':>8}{'Coinc.':>8}{'Discr.':>8}{'Error':>9}")
    for k in sorted(est, key=lambda k: ("|" in k, k)):
        s = est[k]
        det = s["coinciden"] + s["discrepan"]
        tasa = s["discrepan"] / det if det else None
        res[k] = {**s, "determinables": det, "tasa_error": round(tasa, 4) if tasa is not None else None}
        print(f"  {k:<28}{s['revisadas']:>8}{s['indeterminables']:>8}{s['coinciden']:>8}"
              f"{s['discrepan']:>8}{(f'{100 * tasa:.1f} %' if tasa is not None else '-'):>9}")
    print("\n  Regla del protocolo, por repositorio: <= 15 % continuar y corregir; "
          "15-25 % ampliar al 20 % del repositorio; > 25 % evaluar su exclusion")
    for k, v in res.items():
        t = v["tasa_error"]
        if "|" in k and t is not None and t > 0.15:
            print(f"  ATENCION {k}: {100 * t:.1f} % -> {'ampliar al 20 %' if t <= 0.25 else 'evaluar exclusion'}")

    dd = sal / "discrepancias"
    dd.mkdir(exist_ok=True)
    for f in dd.glob("*.jpg"):
        f.unlink()
    for ini in range(0, len(disc), POR_HOJA):
        bloque = [{**d, "especie": "aedes_" + d["especie_etiqueta"]} for d in disc[ini:ini + POR_HOJA]]
        lamina(bloque, Path(args.depurado), "Discrepancias: etiqueta / observada",
               dd / f"discrepancias_{ini // POR_HOJA + 1:02d}.jpg",
               lambda it: f"#{it['id']} {it['especie_etiqueta'][:4]}/{it['observada'][:4]}")
    json.dump({"fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "resultados": res,
               "discrepancias": disc},
              open(sal / "resultado_verificacion.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n  Discrepancias: {len(disc)}  -> {dd}")
    print(f"  Resultado: {sal / 'resultado_verificacion.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--especimenes", default="fase3/agrupacion/especimenes.csv")
    ap.add_argument("--depurado", default="fase3/conjunto_depurado_v2")
    ap.add_argument("--conjunto", default="fase3/conjunto_trabajo.csv")
    ap.add_argument("--salida", default="fase3/verificacion_morfologica")
    ap.add_argument("--n", type=int, default=200, help="imagenes por especie (minimo 10 %%)")
    ap.add_argument("--evaluar", action="store_true")
    ap.add_argument("--forzar", action="store_true")
    ap.add_argument("--solo-laminas", action="store_true")
    ap.add_argument("--exportar-muestra", action="store_true")
    args = ap.parse_args()
    if args.evaluar:
        evaluar(args)
    elif args.exportar_muestra:
        exportar_muestra(args)
    elif args.solo_laminas:
        solo_laminas(args)
    else:
        preparar(args)


if __name__ == "__main__":
    main()
