"""
============================================================
 FASE 3 - CAJAS DEL ESPECIMEN PARA GRAD-CAM
============================================================
 Comparacion de arquitecturas de aprendizaje profundo para
 la clasificacion de mosquitos vectores del dengue
 Leandro Steven Batioja Mena - PUCE Esmeraldas 2026-2027

 PROPOSITO
 ---------
 Protocolo de evaluacion v1.0, apartado 5.2: antes de entrenar
 se traza en cada imagen de la muestra de Grad-CAM una caja que
 encierra el especimen completo (cabeza, torax, abdomen y patas
 visibles). Las cajas no se modifican despues.

 USO (desde ~/Imagenes/tesis)
 ---
   python fase3/anotar_cajas.py              muestra de prueba (Fase 6)
   python fase3/anotar_cajas.py --piloto     muestra del piloto (Fase 4)
   python fase3/anotar_cajas.py --laminas    hojas de control con las cajas

 En la ventana:
   arrastrar con el raton   trazar la caja (se puede rehacer)
   Enter o espacio          guardar y pasar a la siguiente
   q                        salir (lo guardado se conserva; al
                            volver a ejecutar continua donde quedo)

 Las coordenadas se guardan en pixeles de la imagen original,
 con la orientacion EXIF ya aplicada (ImageOps.exif_transpose).
 Grad-CAM debe cargar las imagenes de la misma forma.

 SALIDAS (fase3/particion/)
 -------
   gradcam_cajas.csv / piloto_cajas.csv
     nombre_destino, especie, ancho, alto, x0, y0, x1, y1,
     fraccion_area, informativa (fraccion <= 0,60), fecha
   cajas_laminas/  hojas de control con la caja dibujada
============================================================
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

Image.MAX_IMAGE_PIXELS = None
UMBRAL_INFORMATIVA = 0.60
MAX_LADO = 1400
CAMPOS = ["nombre_destino", "especie", "ancho", "alto", "x0", "y0", "x1", "y1",
          "fraccion_area", "informativa", "fecha"]


def cargar(base, fila):
    with Image.open(base / fila["especie"] / fila["nombre_destino"]) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def leer_hechas(p):
    if not p.exists():
        return {}
    return {(r["especie"], r["nombre_destino"]): r for r in csv.DictReader(open(p, encoding="utf-8"))}


def anotar(muestra, salida, base):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    if matplotlib.get_backend().lower() in ("agg", "pdf", "svg", "ps", "cairo", "template"):
        sys.exit("ERROR: matplotlib no tiene interfaz grafica. Instale tkinter:\n"
                 "  sudo dnf install python3.11-tkinter\n"
                 "y vuelva a ejecutar.")

    filas = list(csv.DictReader(open(muestra, encoding="utf-8")))
    hechas = leer_hechas(salida)
    pendientes = [f for f in filas if (f["especie"], f["nombre_destino"]) not in hechas]
    print(f"  Muestra: {len(filas)}   ya anotadas: {len(hechas)}   pendientes: {len(pendientes)}")
    if not pendientes:
        return
    nuevo = not salida.exists()
    fh = open(salida, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=CAMPOS)
    if nuevo:
        w.writeheader()

    for i, f in enumerate(pendientes, 1):
        im = cargar(base, f)
        W, H = im.size
        esc = min(1.0, MAX_LADO / max(W, H))
        vista = im.resize((max(1, int(W * esc)), max(1, int(H * esc)))) if esc < 1 else im
        estado = {"caja": None, "accion": None}

        fig, ax = plt.subplots(figsize=(12, 9))
        ax.imshow(vista)
        ax.set_axis_off()
        ax.set_title(f"[{len(hechas) + i}/{len(filas)}] {f.get('estrato', '')} {f['nombre_destino']}\n"
                     "Arrastre una caja sobre el mosquito completo.  Enter = guardar   q = salir",
                     fontsize=10)

        def al_soltar(c, s):
            x0, x1 = sorted((c.xdata, s.xdata))
            y0, y1 = sorted((c.ydata, s.ydata))
            estado["caja"] = (x0 / esc, y0 / esc, x1 / esc, y1 / esc)

        def tecla(ev):
            if ev.key in ("enter", " "):
                if estado["caja"] is None:
                    ax.set_xlabel("Primero trace la caja", color="red")
                    fig.canvas.draw_idle()
                    return
                estado["accion"] = "guardar"
                plt.close(fig)
            elif ev.key == "q":
                estado["accion"] = "salir"
                plt.close(fig)

        sel = RectangleSelector(ax, al_soltar, useblit=True, button=[1], minspanx=5, minspany=5,
                                spancoords="pixels", interactive=True)
        fig.canvas.mpl_connect("key_press_event", tecla)
        plt.show()
        del sel

        if estado["accion"] != "guardar":
            print("  Salida solicitada. Progreso guardado.")
            break
        x0, y0, x1, y1 = [round(v) for v in estado["caja"]]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        frac = (x1 - x0) * (y1 - y0) / (W * H)
        w.writerow({"nombre_destino": f["nombre_destino"], "especie": f["especie"], "ancho": W, "alto": H,
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1, "fraccion_area": round(frac, 4),
                    "informativa": "si" if frac <= UMBRAL_INFORMATIVA else "no",
                    "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        fh.flush()
    fh.close()
    hechas = leer_hechas(salida)
    inf = sum(1 for r in hechas.values() if r["informativa"] == "si")
    print(f"  Anotadas: {len(hechas)}/{len(filas)}   informativas: {inf}   -> {salida}")


def laminas(muestra, salida, base, carpeta):
    filas = list(csv.DictReader(open(muestra, encoding="utf-8")))
    cajas = leer_hechas(salida)
    carpeta.mkdir(parents=True, exist_ok=True)
    for f in carpeta.glob(f"{salida.stem}_*.jpg"):
        f.unlink()
    C, cols, filas_h = 360, 6, 4
    por_hoja = cols * filas_h
    n = 0
    for ini in range(0, len(filas), por_hoja):
        hoja = Image.new("RGB", (cols * C, filas_h * C), "white")
        d = ImageDraw.Draw(hoja)
        for k, f in enumerate(filas[ini:ini + por_hoja]):
            im = cargar(base, f)
            W, H = im.size
            r = cajas.get((f["especie"], f["nombre_destino"]))
            if r:
                ImageDraw.Draw(im).rectangle([int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"])],
                                             outline=(255, 0, 0) if r["informativa"] == "si" else (255, 160, 0),
                                             width=max(3, W // 150))
            im.thumbnail((C - 10, C - 30))
            x, y = (k % cols) * C, (k // cols) * C
            hoja.paste(im, (x + (C - im.width) // 2, y + 26))
            txt = f"{ini + k + 1} {f.get('estrato', '')} " + (f"{float(r['fraccion_area']):.2f}" if r else "SIN CAJA")
            d.text((x + 6, y + 6), txt, fill="black" if r else "red")
        n += 1
        hoja.save(carpeta / f"{salida.stem}_{n:02d}.jpg", quality=88)
    print(f"  {n} hojas en {carpeta}  (rojo: informativa; naranja: caja > 60 % del area)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particion", default="fase3/particion")
    ap.add_argument("--depurado", default="fase3/conjunto_depurado_v2")
    ap.add_argument("--piloto", action="store_true", help="muestra del piloto (validacion)")
    ap.add_argument("--laminas", action="store_true", help="solo generar hojas de control")
    args = ap.parse_args()
    d = Path(args.particion)
    muestra = d / ("piloto_gradcam.csv" if args.piloto else "gradcam_muestra.csv")
    salida = d / ("piloto_cajas.csv" if args.piloto else "gradcam_cajas.csv")
    base = Path(args.depurado)
    print("=" * 62)
    print(f"  CAJAS DEL ESPECIMEN - {'PILOTO' if args.piloto else 'PRUEBA (FASE 6)'}")
    print("=" * 62)
    if args.laminas:
        laminas(muestra, salida, base, d / "cajas_laminas")
    else:
        anotar(muestra, salida, base)


if __name__ == "__main__":
    main()
