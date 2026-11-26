"""
============================================================
 FASE 3 - CORRECCION DE LA VERIFICACION MORFOLOGICA (E. 007)
============================================================
 Comparacion de arquitecturas de aprendizaje profundo para
 la clasificacion de mosquitos vectores del dengue
 Leandro Steven Batioja Mena - PUCE Esmeraldas 2026-2027

 PROPOSITO
 ---------
 Aplica las decisiones de la Entrada 007 sin modificar el
 resultado original, que se conserva para la bitacora:

 1. RECLASIFICACION DE LABORATORIO
    En Dryad y Ong (colonias de laboratorio, ejemplares secos)
    el veredicto "otro" se asigno por ausencia de escamas
    blancas, rasgo que estos ejemplares pierden al secarse
    (hallazgo ya documentado con Couret, Entrada 006). El
    criterio no es aplicable a esas fuentes: "otro" pasa a
    "indeterminable". Se recalculan las tasas por repositorio.

 2. EXCLUSIONES POR DISCREPANCIA (regla <= 15 %: corregir)
    Toda discrepancia que subsista tras la reclasificacion se
    excluye. En Mosquito Alert se excluye el reporte completo
    (todas sus fotos), para que no vuelva a entrar al
    rebalancear.

 3. FILTRO DE ESTADIO VITAL EN INATURALIST (criterio E7)
    La verificacion encontro larvas. Se excluyen las
    observaciones cuya anotacion comunitaria de "Life Stage"
    (atributo 1) indica un estadio no adulto (huevo, larva,
    pupa, ninfa, juvenil) y no fue rechazada (voto >= 0).

 USO (desde ~/Imagenes/tesis)
 ---
   python fase3/corregir_verificacion.py \
       --inat-meta "$DATASETS/6-INATURALIST/metadatos_observaciones.json"

 SALIDAS (en fase3/verificacion_morfologica/)
 -------
   verificacion_morfologica_corregida.csv
   resultado_verificacion_corregido.json   original y corregido
   exclusiones_007.csv                      especie, nombre_destino,
                                            origen, grupo, motivo
============================================================
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

LABORATORIO = {"dryad", "ong"}
NOTA_RECLASIF = ("reclasificado a indeterminable: ejemplar de colonia seco con perdida "
                 "de escamas; el criterio 'otro' no es aplicable (Entrada 007)")
# Atributo 1 = Life Stage. Valores de la API de iNaturalist.
ETAPAS = {2: "adulto", 3: "teneral", 4: "pupa", 5: "ninfa", 6: "larva",
          7: "huevo", 8: "juvenil", 16: "subimago"}
NO_ADULTO = {4, 5, 6, 7, 8}


def leer(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def estadisticas(clave, obs):
    est = defaultdict(Counter)
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
            disc.append({**c, "observada": o})
    res = {}
    for k, s in est.items():
        det = s["coinciden"] + s["discrepan"]
        res[k] = {"revisadas": s["revisadas"], "indeterminables": s["indeterminables"],
                  "coinciden": s["coinciden"], "discrepan": s["discrepan"],
                  "determinables": det,
                  "tasa_error": round(s["discrepan"] / det, 4) if det else None}
    return res, disc


def imprimir(titulo, res):
    print(f"\n  {titulo}")
    print(f"  {'Grupo':<28}{'Revis.':>8}{'Indet.':>8}{'Coinc.':>8}{'Discr.':>8}{'Error':>9}")
    for k in sorted(res, key=lambda k: ("|" in k, k)):
        s, t = res[k], res[k]["tasa_error"]
        print(f"  {k:<28}{s['revisadas']:>8}{s['indeterminables']:>8}{s['coinciden']:>8}"
              f"{s['discrepan']:>8}{(f'{100 * t:.1f} %' if t is not None else '-'):>9}")
    for k, s in res.items():
        t = s["tasa_error"]
        if "|" in k and t is not None and t > 0.15:
            print(f"  ATENCION {k}: {100 * t:.1f} % -> "
                  f"{'ampliar al 20 %' if t <= 0.25 else 'evaluar exclusion'}")


def obs_id(fila):
    m = re.search(r"inat_(\d+)_", fila["nombre_destino"])
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="fase3/verificacion_morfologica")
    ap.add_argument("--conjunto", default="fase3/conjunto_trabajo.csv")
    ap.add_argument("--especimenes", default="fase3/agrupacion/especimenes.csv")
    ap.add_argument("--inat-meta", required=True,
                    help="6-INATURALIST/metadatos_observaciones.json")
    args = ap.parse_args()
    sal = Path(args.salida)

    print("=" * 70)
    print("  CORRECCION DE LA VERIFICACION MORFOLOGICA (ENTRADA 007)")
    print("=" * 70)

    # ---------- 1. reclasificacion ----------
    clave = {r["id"]: r for r in leer(sal / "clave_verificacion.csv")}
    filas = leer(sal / "verificacion_morfologica.csv")
    original = {r["id"]: dict(r) for r in filas}
    res_orig, _ = estadisticas(clave, original)

    reclasif = []
    for r in filas:
        c = clave[r["id"]]
        if c["origen"] in LABORATORIO and r["especie_observada"].strip().lower() == "otro":
            r["especie_observada"] = "indeterminable"
            r["nota"] = (r["nota"] + "; " if r["nota"] else "") + NOTA_RECLASIF
            reclasif.append(r["id"])
    with open(sal / "verificacion_morfologica_corregida.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "especie_observada", "rasgo", "nota"])
        w.writeheader()
        w.writerows(filas)
    corregido = {r["id"]: r for r in filas}
    res_corr, disc = estadisticas(clave, corregido)

    imprimir("RESULTADO ORIGINAL", res_orig)
    print(f"\n  Reclasificadas (laboratorio, 'otro' -> indeterminable): {len(reclasif)}")
    imprimir("RESULTADO CORREGIDO", res_corr)

    # ---------- 2. exclusiones por discrepancia ----------
    trabajo = leer(args.conjunto)
    especimenes = leer(args.especimenes)
    excl = {}

    def excluir(f, motivo):
        k = f["nombre_destino"]
        if k not in excl:
            excl[k] = {"especie": f["especie"], "nombre_destino": k, "origen": f["origen"],
                       "grupo": f["grupo"], "motivo": motivo}

    print(f"\n  Discrepancias tras la correccion: {len(disc)}")
    for d in disc:
        motivo = f"007 verificacion #{d['id']}: etiqueta {d['especie_etiqueta']}, observada {d['observada']}"
        fuente = especimenes if d["origen"] == "mosquito_alert" else trabajo
        afectadas = [f for f in fuente if f["grupo"] == d["grupo"]] if d["origen"] == "mosquito_alert" \
            else [f for f in fuente if f["nombre_destino"] == d["nombre_destino"]]
        for f in afectadas:
            excluir(f, motivo)
        print(f"    #{d['id']:<4} {d['origen']:<15} {d['especie_etiqueta']:<11}-> {d['observada']:<11}"
              f"{len(afectadas)} imagen(es) excluida(s)")

    # ---------- 3. estadio vital en iNaturalist ----------
    meta = json.loads(Path(args.inat_meta).expanduser().read_text(encoding="utf-8"))
    etapa = {}
    for o in meta:
        vals = [a.get("controlled_value_id") for a in (o.get("annotations") or [])
                if a.get("controlled_attribute_id") == 1 and (a.get("vote_score") or 0) >= 0]
        if vals:
            etapa[o["id"]] = vals
    conteo = Counter()
    no_adultas = []
    for f in trabajo:
        if f["origen"] != "inaturalist":
            continue
        vals = etapa.get(obs_id(f))
        if not vals:
            conteo["sin anotacion"] += 1
            continue
        malas = [v for v in vals if v in NO_ADULTO]
        if malas:
            conteo["no adulto"] += 1
            nombre = ETAPAS.get(malas[0], f"valor {malas[0]}")
            no_adultas.append((f["nombre_destino"], nombre))
            excluir(f, f"E7 estadio no adulto en iNaturalist (anotacion: {nombre})")
        else:
            conteo["adulto u otro estadio adulto"] += 1
    print(f"\n  iNaturalist en el conjunto de trabajo: {sum(conteo.values())}")
    for k, v in conteo.most_common():
        print(f"    {k}: {v}")
    for n, e in no_adultas:
        print(f"    excluida: {n} ({e})")

    # ---------- salidas ----------
    with open(sal / "exclusiones_007.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["especie", "nombre_destino", "origen", "grupo", "motivo"])
        w.writeheader()
        w.writerows(sorted(excl.values(), key=lambda e: (e["especie"], e["origen"], e["nombre_destino"])))

    fuera = set(excl)
    comp = defaultdict(Counter)
    for f in trabajo:
        if f["nombre_destino"] not in fuera:
            comp[f["especie"]][f["origen"]] += 1
    print(f"\n  Exclusiones totales: {len(excl)} "
          f"(en el conjunto de trabajo: {sum(1 for f in trabajo if f['nombre_destino'] in fuera)})")
    print("  Conjunto de trabajo tras exclusiones (antes de rebalancear):")
    for e in sorted(comp):
        tot = sum(comp[e].values())
        det = ", ".join(f"{o} {n}" for o, n in sorted(comp[e].items()))
        print(f"    {e}: {tot}  ({det})")

    json.dump({
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "resultado_original": res_orig,
        "reclasificadas_laboratorio": reclasif,
        "criterio_reclasificacion": NOTA_RECLASIF,
        "resultado_corregido": res_corr,
        "discrepancias_corregidas": disc,
        "inaturalist_estadio": dict(conteo),
        "inaturalist_no_adultas": [{"nombre_destino": n, "estadio": e} for n, e in no_adultas],
        "exclusiones": len(excl),
        "conjunto_tras_exclusiones": {e: dict(comp[e]) for e in comp},
    }, open(sal / "resultado_verificacion_corregido.json", "w", encoding="utf-8"),
        indent=2, ensure_ascii=False)
    print(f"\n  Guardado: {sal / 'resultado_verificacion_corregido.json'}")
    print(f"            {sal / 'exclusiones_007.csv'}")
    print(f"            {sal / 'verificacion_morfologica_corregida.csv'}")


if __name__ == "__main__":
    main()
