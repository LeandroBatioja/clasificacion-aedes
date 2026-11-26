"""
============================================================
 FASE 3 - CONSOLIDACION DEL CONJUNTO BRUTO
============================================================
 Comparacion de arquitecturas de aprendizaje profundo para
 la clasificacion de mosquitos vectores del dengue
 Leandro Steven Batioja Mena - PUCE Esmeraldas 2026-2027

 Version 2 (23/10/2026). Cambios respecto a la v1:
   - Sexto repositorio: 6-INATURALIST (fotografias de celular de
     Ae. aegypti, fuera de Ecuador), por observacion del tutor.
     Se procesa si la carpeta existe.
   - --sufijo: escribe el CSV de procedencia y el reporte con un
     sufijo (p. ej. _v2), para no sobrescribir los de la v1, que
     son evidencia de la Entrada 005.
   - Se detiene si el destino ya contiene imagenes: reconsolidar
     sobre un conjunto existente generaba nombres duplicados.
   - Dryad: se incorporan las fotografias de celular (Phone_) de
     DeviceDataset/body, clases AE y AL. En la v1 se excluyo todo
     DeviceDataset por suponer que sus clases eran dispositivos;
     en realidad body contiene carpetas por especie y los nombres
     llevan el codigo de especimen, el mismo de DepictionDataset
     (Phone_AEa04_... corresponde a Macro_AEa04_... y Micro_AEa04_...).
     Solo se toman las Phone_: las Macro_ y Micro_ de DeviceDataset
     repiten las de DepictionDataset.

 PROPOSITO
 ---------
 Unifica los repositorios en una estructura por especie,
 conservando la trazabilidad del origen de cada imagen y del
 especimen al que pertenece. Genera la tabla cruzada
 especie x repositorio exigida por la medida M1 del protocolo
 de criterios de depuracion.

 QUE HACE
 --------
   1. Recorre los repositorios con sus rutas especificas
   2. Filtra lo que NO corresponde (JA, KO, wing, DeviceDataset)
   3. Enlaza (enlace duro) con prefijo de origen a conjunto_bruto/
   4. Extrae el identificador de especimen donde existe
   5. Genera la tabla cruzada y verifica la medida M1
   6. Produce el CSV de procedencia (instrumento 3.6.1.2)

 USO
 ---
   python consolidar_conjunto.py --repos ../repositorios \
                                 --destino ../conjunto_bruto_v2 --sufijo _v2
   python consolidar_conjunto.py --repos ../repositorios \
                                 --destino ../conjunto_bruto_v2 --sufijo _v2 --simular
============================================================
"""

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

AEGYPTI = "aedes_aegypti"
ALBOPICTUS = "aedes_albopictus"

# Umbral de la medida M1: ninguna especie debe provenir en mas
# de este porcentaje de una unica fuente
UMBRAL_M1 = 70.0


def es_imagen(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in EXT


def especimen_dryad(nombre: str):
    """
    Extrae el identificador de especimen de Dryad.
    Macro_AEa04_2412_.png -> AEa04
    Los tres dispositivos (Macro/Micro/Phone) del mismo
    especimen comparten este identificador.
    """
    m = re.search(r"_((?:AE|AL|JA|KO)[a-z]?\d+)(?:_|\.)", nombre, re.I)
    return m.group(1) if m else None


def especimen_kaggle(nombre: str):
    """
    aegypti100a.jpg        -> aegypti100
    aegypti29b - Copy.jpg  -> aegypti29
    Las letras finales (a, b, c, d) son tomas del mismo especimen.
    Las copias se agrupan con su original y MD5 las eliminara.
    """
    m = re.match(r"(aegypti\d+)[a-z]?(?:\s*-\s*copy)?\.", nombre, re.I)
    return m.group(1) if m else None


def especimen_inaturalist(nombre: str):
    """
    inat_123456_987654.jpg -> obs123456
    Se descarga una sola foto por observacion, pero el identificador
    de la observacion es el del especimen.
    """
    m = re.match(r"inat_(\d+)_\d+\.", nombre, re.I)
    return f"obs{m.group(1)}" if m else None


def recolectar(repos: Path):
    """Devuelve la lista de registros con origen, especie y especimen."""
    reg = []

    # ---------- 1-KAGGLE ----------
    base = repos / "1-KAGGLE" / "aegypti"
    for sub in ["aegypti_0", "aegypti_0_testing"]:
        d = base / sub
        if not d.is_dir():
            print(f"    aviso: no existe {d}")
            continue
        for f in sorted(d.iterdir()):
            if es_imagen(f):
                reg.append({
                    "origen": "kaggle",
                    "especie": AEGYPTI,
                    "ruta": f,
                    "especimen": especimen_kaggle(f.name),
                    "detalle": sub,
                })

    # ---------- 2-MOSQUITO ALERT ----------
    base = repos / "2-MOSQUITO ALERT"
    for sub, esp in [("aedes_aegypti", AEGYPTI), ("aedes_albopictus", ALBOPICTUS)]:
        d = base / sub
        if not d.is_dir():
            print(f"    aviso: no existe {d}")
            continue
        for f in sorted(d.iterdir()):
            if es_imagen(f):
                # el agrupamiento por reporte se hace en agrupar_especimenes.py
                reg.append({
                    "origen": "mosquito_alert",
                    "especie": esp,
                    "ruta": f,
                    "especimen": f.stem,
                    "detalle": sub,
                })

    # ---------- 3-DRYAD ----------
    # DepictionDataset / body / {train,val,test} / {AE,AL}  (Macro y Micro)
    # DeviceDataset / body / ... / {AE,AL} / Phone_*         (celular)
    # Se descartan: wing, Macro/Micro de DeviceDataset (repetidas), JA y KO
    base = repos / "3-DRIAD"
    dep = None
    for cand in base.rglob("DepictionDataset"):
        if cand.is_dir():
            dep = cand
            break
    if dep is None:
        print("    aviso: no se encontro DepictionDataset en 3-DRIAD")
    else:
        for split in ["train", "val", "test"]:
            for clase, esp in [("AE", AEGYPTI), ("AL", ALBOPICTUS)]:
                d = dep / "body" / split / clase
                if not d.is_dir():
                    continue
                for f in sorted(d.iterdir()):
                    if es_imagen(f):
                        reg.append({
                            "origen": "dryad",
                            "especie": esp,
                            "ruta": f,
                            "especimen": especimen_dryad(f.name),
                            "detalle": f"body/{split}/{clase}",
                        })
        dev = dep.parent / "DeviceDataset" / "body"
        if not dev.is_dir():
            print("    aviso: no se encontro DeviceDataset/body en 3-DRIAD")
        else:
            for f in sorted(dev.rglob("Phone_*")):
                if not es_imagen(f) or f.parent.name not in ("AE", "AL"):
                    continue
                reg.append({
                    "origen": "dryad",
                    "especie": AEGYPTI if f.parent.name == "AE" else ALBOPICTUS,
                    "ruta": f,
                    "especimen": especimen_dryad(f.name),
                    "detalle": "DeviceDataset/" + str(f.parent.relative_to(dev.parent)),
                })

    # ---------- 4-COURET ----------
    d = repos / "4-COURET"
    if not d.is_dir():
        print(f"    aviso: no existe {d}")
    else:
        for f in sorted(d.rglob("*")):
            if es_imagen(f):
                reg.append({
                    "origen": "couret",
                    "especie": AEGYPTI,
                    "ruta": f,
                    "especimen": f.stem,
                    "detalle": "NEWORLEANS",
                })

    # ---------- 5-ONG ----------
    base = repos / "5-ONG"
    if not base.is_dir():
        print(f"    aviso: no existe {base}")
    else:
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            low = d.name.lower()
            if "aegypti" in low:
                esp = AEGYPTI
            elif "albopictus" in low:
                esp = ALBOPICTUS
            else:
                print(f"    aviso: carpeta ignorada en 5-ONG -> {d.name}")
                continue
            for f in sorted(d.iterdir()):
                if es_imagen(f):
                    reg.append({
                        "origen": "ong",
                        "especie": esp,
                        "ruta": f,
                        "especimen": None,
                        "detalle": d.name.strip(),
                    })

    # ---------- 6-INATURALIST ----------
    d = repos / "6-INATURALIST" / "aedes_aegypti"
    if not d.is_dir():
        print(f"    aviso: no existe {d} (se omite iNaturalist)")
    else:
        for f in sorted(d.iterdir()):
            if es_imagen(f):
                reg.append({
                    "origen": "inaturalist",
                    "especie": AEGYPTI,
                    "ruta": f,
                    "especimen": especimen_inaturalist(f.name),
                    "detalle": "research_grade_fuera_de_Ecuador",
                })

    return reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", required=True, help="carpeta con los repositorios")
    ap.add_argument("--destino", required=True, help="carpeta conjunto_bruto de salida")
    ap.add_argument("--sufijo", default="",
                    help="sufijo para el CSV de procedencia y el reporte (p. ej. _v2)")
    ap.add_argument("--simular", action="store_true",
                    help="solo analiza, no escribe archivos")
    ap.add_argument("--copiar", action="store_true",
                    help="copia fisica en lugar de enlaces duros (ocupa espacio)")
    args = ap.parse_args()

    repos = Path(args.repos).expanduser().resolve()
    destino = Path(args.destino).expanduser().resolve()

    if not repos.is_dir():
        print(f"ERROR: no existe {repos}")
        return
    if not args.simular and destino.exists() and any(destino.rglob("*.*")):
        print(f"ERROR: {destino} ya contiene imagenes. Use una carpeta nueva "
              f"(p. ej. conjunto_bruto_v2) para no generar nombres duplicados.")
        return

    print("=" * 62)
    print("  CONSOLIDACION DEL CONJUNTO BRUTO")
    print("=" * 62)
    print(f"  Repositorios : {repos}")
    print(f"  Destino      : {destino}")
    if args.simular:
        modo = "SIMULACION (no escribe nada)"
    elif args.copiar:
        modo = "COPIA FISICA (ocupa espacio)"
    else:
        modo = "ENLACES DUROS (sin espacio adicional)"
    print(f"  Modo         : {modo}")
    print()
    print("  Recolectando...")

    reg = recolectar(repos)
    if not reg:
        print("\n  No se encontro ninguna imagen. Verifique las rutas.")
        return

    print(f"  {len(reg)} imagenes recolectadas\n")

    # ---------- TABLA CRUZADA (medida M1) ----------
    cruz = defaultdict(lambda: defaultdict(int))
    for r in reg:
        cruz[r["especie"]][r["origen"]] += 1

    origenes = sorted({r["origen"] for r in reg})
    especies = [AEGYPTI, ALBOPICTUS]

    print("=" * 62)
    print("  TABLA CRUZADA ESPECIE x REPOSITORIO  (medida M1)")
    print("=" * 62)
    enc = f"  {'Repositorio':<18}" + "".join(f"{e.replace('aedes_',''):>18}" for e in especies)
    print(enc)
    print("  " + "-" * (len(enc) - 2))
    for o in origenes:
        fila = f"  {o:<18}"
        for e in especies:
            n = cruz[e][o]
            tot = sum(cruz[e].values())
            pct = 100 * n / tot if tot else 0
            fila += f"{f'{n} ({pct:.1f}%)':>18}"
        print(fila)
    print("  " + "-" * (len(enc) - 2))
    fila = f"  {'TOTAL':<18}"
    for e in especies:
        fila += f"{sum(cruz[e].values()):>18}"
    print(fila)

    # ---------- VERIFICACION M1 ----------
    print("\n" + "=" * 62)
    print(f"  VERIFICACION M1  (ninguna fuente > {UMBRAL_M1}%)")
    print("=" * 62)
    m1 = {}
    for e in especies:
        tot = sum(cruz[e].values())
        if not tot:
            continue
        o_max = max(cruz[e], key=cruz[e].get)
        pct = 100 * cruz[e][o_max] / tot
        cumple = pct <= UMBRAL_M1
        m1[e] = {"fuente_dominante": o_max, "porcentaje": round(pct, 1),
                 "cumple": cumple, "total": tot}
        estado = "CUMPLE" if cumple else "NO CUMPLE"
        print(f"  {e:<20} {o_max:<16} {pct:>6.1f}%   {estado}")

    if not all(v["cumple"] for v in m1.values()):
        print()
        print("  El conjunto bruto no cumple M1. Se resolvera mediante")
        print("  submuestreo dirigido de la fuente dominante durante el")
        print("  balanceo, priorizando diversidad de origen.")

    # ---------- ESPECIMENES ----------
    print("\n" + "=" * 62)
    print("  AGRUPAMIENTO POR ESPECIMEN")
    print("=" * 62)
    con_id = sum(1 for r in reg if r["especimen"])
    print(f"  Con identificador : {con_id}")
    print(f"  Sin identificador : {len(reg) - con_id}  (se agrupan en agrupar_especimenes.py)")

    grupos = defaultdict(int)
    for r in reg:
        if r["especimen"]:
            grupos[(r["origen"], r["especimen"])] += 1
    multi = {k: v for k, v in grupos.items() if v > 1}
    if multi:
        print(f"  Especimenes con varias imagenes: {len(multi)}")
        top = sorted(multi.items(), key=lambda x: -x[1])[:5]
        for (o, sp), n in top:
            print(f"    {o:<16} {sp:<12} {n} imagenes")

    # ---------- ESCRITURA ----------
    if not args.simular:
        print("\n" + "=" * 62)
        print("  ESCRIBIENDO CONJUNTO BRUTO")
        print("=" * 62)
        for e in especies:
            (destino / e).mkdir(parents=True, exist_ok=True)

        copiadas, fallos = 0, []
        enlazadas, copiadas_fis = 0, 0
        for i, r in enumerate(reg, 1):
            nombre = f"{r['origen']}__{r['ruta'].name}"
            dst = destino / r["especie"] / nombre
            if dst.exists():
                nombre = f"{r['origen']}__{i:05d}__{r['ruta'].name}"
                dst = destino / r["especie"] / nombre
            try:
                if args.copiar:
                    shutil.copy2(r["ruta"], dst)
                    copiadas_fis += 1
                else:
                    try:
                        os.link(r["ruta"], dst)
                        enlazadas += 1
                    except OSError:
                        # distinta particion: se recurre a copia fisica
                        shutil.copy2(r["ruta"], dst)
                        copiadas_fis += 1
                r["nombre_destino"] = nombre
                copiadas += 1
            except Exception as ex:
                fallos.append((str(r["ruta"]), str(ex)))
            if i % 1000 == 0:
                print(f"    {i}/{len(reg)}")
        print(f"  Total escritas     : {copiadas}")
        print(f"    enlaces duros    : {enlazadas}")
        print(f"    copias fisicas   : {copiadas_fis}")
        if fallos:
            print(f"  Fallos  : {len(fallos)}")
            for f, ex in fallos[:5]:
                print(f"    {f}: {ex[:60]}")
    else:
        for i, r in enumerate(reg, 1):
            r["nombre_destino"] = f"{r['origen']}__{r['ruta'].name}"

    # ---------- CSV DE PROCEDENCIA ----------
    salida = destino.parent if not args.simular else Path(".")
    salida.mkdir(parents=True, exist_ok=True)
    csv_path = salida / f"procedencia_conjunto_bruto{args.sufijo}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["nombre_destino", "especie", "origen", "especimen",
                    "detalle", "ruta_original"])
        for r in reg:
            w.writerow([r.get("nombre_destino", ""), r["especie"], r["origen"],
                        r["especimen"] or "", r["detalle"], str(r["ruta"])])

    reporte = {
        "version": 2,
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "repositorios": str(repos),
        "destino": str(destino),
        "modo": "simulacion" if args.simular else ("copia" if args.copiar else "enlace_duro"),
        "total_imagenes": len(reg),
        "tabla_cruzada": {e: dict(cruz[e]) for e in especies},
        "verificacion_m1": m1,
        "umbral_m1": UMBRAL_M1,
        "especimenes_con_id": con_id,
        "especimenes_multiples": len(multi),
    }
    json_path = salida / f"reporte_consolidacion{args.sufijo}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(reporte, fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 62)
    print("  ARCHIVOS GENERADOS")
    print("=" * 62)
    print(f"  {csv_path}")
    print(f"  {json_path}")
    if not args.simular:
        print(f"  {destino}/{AEGYPTI}/")
        print(f"  {destino}/{ALBOPICTUS}/")
    print("=" * 62)


if __name__ == "__main__":
    main()
