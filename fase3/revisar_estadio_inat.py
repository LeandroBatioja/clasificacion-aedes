"""
============================================================
 FASE 3 - REVISION VISUAL DE ESTADIO VITAL EN INATURALIST
============================================================
 Comparacion de arquitecturas de aprendizaje profundo para
 la clasificacion de mosquitos vectores del dengue
 Leandro Steven Batioja Mena - PUCE Esmeraldas 2026-2027

 PROPOSITO
 ---------
 Completa el criterio E7 (solo adultos) en iNaturalist. El
 filtro por anotacion (corregir_verificacion.py) solo actua
 sobre las observaciones anotadas; este programa muestra las
 que no tienen anotacion de estadio para revisarlas a ojo.

 USO (desde ~/Imagenes/tesis)
 ---
 1) Generar hojas de contacto:
      python fase3/revisar_estadio_inat.py \
          --inat-meta "$DATASETS/6-INATURALIST/metadatos_observaciones.json"
    -> fase3/verificacion_morfologica/estadio_inat/hoja_XX.jpg
       fase3/verificacion_morfologica/estadio_inat/indice.csv
       fase3/verificacion_morfologica/estadio_inat/no_adultos.txt

 2) Escribir en no_adultos.txt los numeros de las imagenes que
    NO son mosquitos adultos (larva, pupa, huevo, exuvia u otro
    objeto), uno por linea o separados por comas. Dejar vacio si
    no hay ninguno.

 3) Aplicar:
      python fase3/revisar_estadio_inat.py --aplicar
    -> agrega esas imagenes a exclusiones_007.csv con el motivo
       "E7 revision visual" (sin duplicar las que ya estan)
============================================================
"""

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

Image.MAX_IMAGE_PIXELS = None
COLS, FILAS, CELDA = 8, 6, 260
POR_HOJA = COLS * FILAS


def fuente(n):
    for f in ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, n)
        except OSError:
            pass
    return ImageFont.load_default()


def obs_id(nombre):
    m = re.search(r"inat_(\d+)_", nombre)
    return int(m.group(1)) if m else None


def generar(args, d):
    meta = json.loads(Path(args.inat_meta).expanduser().read_text(encoding="utf-8"))
    anotadas = {o["id"] for o in meta
                if any(a.get("controlled_attribute_id") == 1 and (a.get("vote_score") or 0) >= 0
                       for a in (o.get("annotations") or []))}
    excl = {r["nombre_destino"] for r in csv.DictReader(open(args.exclusiones, encoding="utf-8"))}
    pendientes = sorted(
        f["nombre_destino"] for f in csv.DictReader(open(args.conjunto, encoding="utf-8"))
        if f["origen"] == "inaturalist" and f["nombre_destino"] not in excl
        and obs_id(f["nombre_destino"]) not in anotadas)

    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("hoja_*.jpg"):
        f.unlink()
    with open(d / "indice.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "nombre_destino"])
        for n, nom in enumerate(pendientes, 1):
            w.writerow([n, nom])

    f_num = fuente(26)
    base = Path(args.depurado) / "aedes_aegypti"
    hojas = 0
    for ini in range(0, len(pendientes), POR_HOJA):
        hoja = Image.new("RGB", (COLS * CELDA, FILAS * CELDA), "white")
        dib = ImageDraw.Draw(hoja)
        for k, nom in enumerate(pendientes[ini:ini + POR_HOJA]):
            n = ini + k + 1
            x, y = (k % COLS) * CELDA, (k // COLS) * CELDA
            try:
                with Image.open(base / nom) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im.thumbnail((CELDA - 8, CELDA - 8))
                    hoja.paste(im, (x + (CELDA - im.width) // 2, y + (CELDA - im.height) // 2))
            except Exception as e:
                dib.text((x + 10, y + 120), f"no legible ({type(e).__name__})", fill="red")
            dib.rectangle([x, y, x + 76, y + 34], fill="yellow")
            dib.text((x + 4, y + 3), str(n), fill="black", font=f_num)
            dib.rectangle([x, y, x + CELDA - 1, y + CELDA - 1], outline="#bbbbbb")
        hojas += 1
        hoja.save(d / f"hoja_{hojas:02d}.jpg", quality=88)

    txt = d / "no_adultos.txt"
    if not txt.exists():
        txt.write_text("", encoding="utf-8")
    print(f"  Observaciones sin anotacion de estadio (no excluidas): {len(pendientes)}")
    print(f"  Hojas: {hojas} ({POR_HOJA} por hoja) en {d}")
    print(f"  Escriba los numeros que NO son adultos en {txt} y ejecute --aplicar")


def aplicar(args, d):
    indice = {r["n"]: r["nombre_destino"] for r in csv.DictReader(open(d / "indice.csv", encoding="utf-8"))}
    nums = [t for t in re.split(r"[\s,;]+", (d / "no_adultos.txt").read_text(encoding="utf-8")) if t]
    malos = [t for t in nums if t not in indice]
    if malos:
        raise SystemExit(f"ERROR: numeros fuera del indice: {malos}")
    lista = list(csv.DictReader(open(args.conjunto, encoding="utf-8")))
    trabajo = {f["nombre_destino"]: f for f in lista if f["origen"] == "inaturalist"}
    exc_path = Path(args.exclusiones)
    filas = list(csv.DictReader(open(exc_path, encoding="utf-8")))
    ya = {(r["especie"], r["nombre_destino"]) for r in filas}
    nuevas = 0
    for t in nums:
        nom = indice[t]
        f = trabajo[nom]
        if (f["especie"], nom) in ya:
            continue
        filas.append({"especie": f["especie"], "nombre_destino": nom, "origen": f["origen"],
                      "grupo": f["grupo"], "motivo": f"E7 revision visual (hoja de estadio n.{t})"})
        ya.add((f["especie"], nom))
        nuevas += 1
    with open(exc_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["especie", "nombre_destino", "origen", "grupo", "motivo"])
        w.writeheader()
        w.writerows(sorted(filas, key=lambda e: (e["especie"], e["origen"], e["nombre_destino"])))
    json.dump({"fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "revisadas": len(indice),
               "no_adultos": [{"n": t, "nombre_destino": indice[t]} for t in nums],
               "agregadas_a_exclusiones": nuevas, "exclusiones_totales": len(filas)},
              open(d / "resultado_estadio.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    aeg = sum(1 for f in lista if f["especie"] == "aedes_aegypti" and (f["especie"], f["nombre_destino"]) not in ya)
    alb = sum(1 for f in lista if f["especie"] == "aedes_albopictus" and (f["especie"], f["nombre_destino"]) not in ya)
    print(f"  Revisadas: {len(indice)}   No adultos marcados: {len(nums)}   Agregadas: {nuevas}")
    print(f"  Exclusiones totales en {exc_path}: {len(filas)}")
    print(f"  Conjunto de trabajo tras exclusiones: aegypti {aeg}, albopictus {alb}")
    print(f"  Resultado: {d / 'resultado_estadio.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inat-meta", help="6-INATURALIST/metadatos_observaciones.json")
    ap.add_argument("--conjunto", default="fase3/conjunto_trabajo.csv")
    ap.add_argument("--depurado", default="fase3/conjunto_depurado_v2")
    ap.add_argument("--exclusiones", default="fase3/verificacion_morfologica/exclusiones_007.csv")
    ap.add_argument("--salida", default="fase3/verificacion_morfologica/estadio_inat")
    ap.add_argument("--aplicar", action="store_true")
    args = ap.parse_args()
    d = Path(args.salida)
    print("=" * 62)
    print("  REVISION VISUAL DE ESTADIO VITAL - INATURALIST")
    print("=" * 62)
    if args.aplicar:
        aplicar(args, d)
    else:
        if not args.inat_meta:
            raise SystemExit("ERROR: falta --inat-meta")
        generar(args, d)


if __name__ == "__main__":
    main()
