"""
============================================================
 FASE 3 - DESCARGA DE AEDES AEGYPTI DESDE INATURALIST
============================================================
 Comparacion de arquitecturas de aprendizaje profundo para
 la clasificacion de mosquitos vectores del dengue
 Leandro Steven Batioja Mena - PUCE Esmeraldas 2026-2027

 PROPOSITO
 ---------
 Observacion del tutor (23/10/2026): casi todas las imagenes de
 Ae. albopictus son fotografias de celular (Mosquito Alert) y las
 de Ae. aegypti son mayoritariamente de laboratorio, de modo que el
 tipo de fotografia podria delatar la especie. Este programa
 obtiene fotografias de celular de Ae. aegypti desde iNaturalist.

 CRITERIOS DE SELECCION (fijados antes de descargar)
 ---------------------------------------------------
   - Taxon Aedes aegypti (incluye sus subespecies)
   - Grado de investigacion (identificacion confirmada por la
     comunidad)
   - Licencia de la foto: CC0, CC BY o CC BY-NC (criterio E9)
   - FUERA DE ECUADOR: las observaciones ecuatorianas se reservan
     para la validacion contextual. Se excluyen tambien las que no
     tienen ubicacion, porque no se puede verificar el pais.
   - Una sola foto por observacion (la primera con licencia
     valida): cada imagen es un especimen distinto y no hay que
     agrupar.
   - Maximo de fotos por observador (--max-por-usuario), para que
     ningun fotografo ni su montaje domine la fuente.
   - Muestreo aleatorio con semilla fija entre las observaciones
     elegibles.

 USO
 ---
   python descargar_inaturalist.py --repos ~/Imagenes/TESIS/DATASETS --n 1100
   python descargar_inaturalist.py --repos ... --solo-metadatos   (cuenta sin descargar)

 Se puede interrumpir y volver a ejecutar: reutiliza los metadatos
 guardados y no vuelve a descargar las fotos que ya existen.

 SALIDAS (en <repos>/6-INATURALIST/)
 -----------------------------------
   aedes_aegypti/inat_<observacion>_<foto>.jpg
   procedencia_inaturalist.csv   una fila por foto: observacion,
                                 foto, autor, licencia, fecha,
                                 lugar, coordenadas y URL
   resumen_inaturalist.json      conteos y exclusiones
   metadatos_observaciones.json  respuesta completa de la API (cache)
============================================================
"""

import argparse
import csv
import json
import random
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

API = "https://api.inaturalist.org/v1"
UA = "PUCE-Esmeraldas-tesis-clasificacion-Aedes/1.0 (uso academico)"
LICENCIAS = ("cc0", "cc-by", "cc-by-nc")
PAUSA = 1.1          # segundos entre peticiones (limite recomendado por iNaturalist)
SEMILLA = 42


def get_json(ruta, params=None, reintentos=4):
    url = f"{API}/{ruta}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for k in range(reintentos):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:
            espera = 5 * (k + 1)
            print(f"    aviso: {type(e).__name__} en {ruta}; reintento en {espera} s")
            time.sleep(espera)
    raise SystemExit(f"ERROR: no se pudo consultar {url}")


def id_taxon(nombre):
    res = get_json("taxa", {"q": nombre, "rank": "species", "per_page": 10})["results"]
    for t in res:
        if t.get("name", "").lower() == nombre.lower():
            return t["id"]
    raise SystemExit(f"ERROR: no se encontro el taxon {nombre}")


def id_pais(nombre):
    res = get_json("places/autocomplete", {"q": nombre, "per_page": 20})["results"]
    for p in res:
        if p.get("name", "").lower() == nombre.lower() and p.get("admin_level") == 0:
            return p["id"]
    raise SystemExit(f"ERROR: no se encontro el pais {nombre} en iNaturalist")


def descargar_metadatos(taxon, cache):
    if cache.exists():
        datos = json.loads(cache.read_text(encoding="utf-8"))
        print(f"  Metadatos en cache: {len(datos)} observaciones")
        return datos
    datos, ultimo = [], 0
    params = {"taxon_id": taxon, "quality_grade": "research", "photos": "true",
              "photo_license": ",".join(LICENCIAS), "per_page": 200,
              "order_by": "id", "order": "asc"}
    while True:
        params["id_above"] = ultimo
        res = get_json("observations", params)
        lote = res.get("results", [])
        if not lote:
            break
        datos.extend(lote)
        ultimo = lote[-1]["id"]
        print(f"    {len(datos)} / {res.get('total_results', '?')} observaciones")
        time.sleep(PAUSA)
    cache.write_text(json.dumps(datos), encoding="utf-8")
    return datos


def foto_valida(obs):
    for f in obs.get("photos") or []:
        if (f.get("license_code") or "").lower() in LICENCIAS and f.get("url"):
            return f
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", required=True, help="carpeta que contiene los repositorios (1-KAGGLE, ...)")
    ap.add_argument("--n", type=int, default=1100, help="fotos a descargar (margen para la depuracion)")
    ap.add_argument("--max-por-usuario", type=int, default=10)
    ap.add_argument("--pais-excluido", default="Ecuador")
    ap.add_argument("--solo-metadatos", action="store_true")
    args = ap.parse_args()

    base = Path(args.repos).expanduser() / "6-INATURALIST"
    dest = base / "aedes_aegypti"
    dest.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  DESCARGA DE AEDES AEGYPTI DESDE INATURALIST")
    print("=" * 62)
    taxon = id_taxon("Aedes aegypti")
    time.sleep(PAUSA)
    pais = id_pais(args.pais_excluido)
    print(f"  Taxon Aedes aegypti: {taxon}   {args.pais_excluido}: place_id {pais}")
    time.sleep(PAUSA)

    obs = descargar_metadatos(taxon, base / "metadatos_observaciones.json")

    # ---------- filtros ----------
    excl = Counter()
    elegibles = []
    for o in obs:
        nombre = ((o.get("taxon") or {}).get("name") or "")
        lugares = o.get("place_ids") or []
        f = foto_valida(o)
        if o.get("quality_grade") != "research":
            excl["no es grado de investigacion"] += 1
        elif not nombre.startswith("Aedes aegypti"):
            excl["taxon distinto"] += 1
        elif o.get("captive"):
            excl["cautivo o cultivado"] += 1
        elif not lugares:
            excl["sin ubicacion (pais no verificable)"] += 1
        elif pais in lugares:
            excl[f"observada en {args.pais_excluido}"] += 1
        elif f is None:
            excl["sin foto con licencia valida"] += 1
        else:
            elegibles.append((o, f))

    # ---------- muestreo con tope por observador ----------
    rnd = random.Random(SEMILLA)
    rnd.shuffle(elegibles)
    por_usuario, disponibles = Counter(), []
    for o, f in elegibles:
        u = (o.get("user") or {}).get("login", "?")
        if por_usuario[u] >= args.max_por_usuario:
            excl["supera el tope por observador"] += 1
            continue
        por_usuario[u] += 1
        disponibles.append((o, f))
    elegidas = sorted(disponibles[:args.n], key=lambda x: x[0]["id"])
    observadores = len({(o.get("user") or {}).get("login", "?") for o, _ in elegidas})

    print(f"\n  Observaciones con foto CC en grado de investigacion: {len(obs)}")
    for motivo, n in excl.most_common():
        print(f"    excluidas - {motivo}: {n}")
    print(f"  Elegibles: {len(elegibles)}   Disponibles tras el tope: {len(disponibles)}   "
          f"Seleccionadas: {len(elegidas)} de {observadores} observadores")

    # ---------- descarga ----------
    filas, fallos, nuevas = [], [], 0
    for k, (o, f) in enumerate(elegidas, 1):
        nombre = f"inat_{o['id']}_{f['id']}.jpg"
        destino = dest / nombre
        url = f["url"].replace("/square.", "/large.")
        coords = (o.get("geojson") or {}).get("coordinates") or ["", ""]
        filas.append({
            "archivo": nombre, "observacion": o["id"], "foto": f["id"],
            "uri": o.get("uri", f"https://www.inaturalist.org/observations/{o['id']}"),
            "autor": (o.get("user") or {}).get("login", ""),
            "licencia": f.get("license_code", ""), "atribucion": f.get("attribution", ""),
            "fecha": o.get("observed_on", ""), "lugar": o.get("place_guess", ""),
            "lat": coords[1], "lon": coords[0], "ubicacion_oculta": o.get("obscured", ""),
            "url_foto": url,
        })
        if args.solo_metadatos or destino.exists():
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                datos = r.read()
            destino.write_bytes(datos)
            nuevas += 1
        except Exception as e:
            fallos.append((nombre, type(e).__name__))
        if k % 50 == 0:
            print(f"    {k}/{len(elegidas)} fotos")
        time.sleep(PAUSA)

    with open(base / "procedencia_inaturalist.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()) if filas else ["archivo"])
        w.writeheader()
        w.writerows(filas)

    resumen = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "taxon_id": taxon, "pais_excluido": args.pais_excluido, "place_id_excluido": pais,
        "criterios": {"grado": "research", "licencias": LICENCIAS, "fotos_por_observacion": 1,
                      "max_por_usuario": args.max_por_usuario, "semilla": SEMILLA},
        "observaciones_consultadas": len(obs),
        "exclusiones": dict(excl),
        "elegibles": len(elegibles),
        "disponibles_tras_tope": len(disponibles),
        "seleccionadas": len(elegidas),
        "observadores": observadores,
        "licencias": dict(Counter(x["licencia"] for x in filas)),
        "descargadas_en_esta_ejecucion": nuevas,
        "fallos": fallos,
    }
    (base / "resumen_inaturalist.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  Descargadas ahora: {nuevas}   Fallos: {len(fallos)}")
    print(f"  Carpeta: {dest}")
    print(f"  Procedencia: {base / 'procedencia_inaturalist.csv'}")
    print(f"  Resumen: {base / 'resumen_inaturalist.json'}")


if __name__ == "__main__":
    main()
