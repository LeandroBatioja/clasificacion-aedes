"""
============================================================
 AUDITORIA DE LA ESTRUCTURA DEL PROYECTO
============================================================
 Comparacion de arquitecturas de aprendizaje profundo para
 la clasificacion de mosquitos vectores del dengue
 Leandro Steven Batioja Mena - PUCE Esmeraldas 2026-2027

 Contrasta el inventario de archivos de bitacora_experimentos.md
 con el contenido real del disco y regenera estructura_proyecto.txt.

   1. Verifica que cada ruta del inventario exista
   2. Cuenta los archivos de conjunto_bruto/ y conjunto_depurado/
      y comprueba que sean enlaces duros
   3. Lista archivos en disco que no figuran en el inventario
   4. Comprueba que cada par de verificacion_dup/ tenga A y B
   5. Informa la fecha de modificacion de cada entrada, para
      completar las fechas en blanco del inventario
   6. Regenera el arbol colapsando carpetas masivas con su conteo

 USO (desde la raiz del proyecto, ~/Imagenes/tesis)
   python fase3/auditar_estructura.py
   python fase3/auditar_estructura.py --raiz ~/Imagenes/tesis --regenerar
============================================================
"""

import argparse
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ESPERADOS = {
    "fase3/conjunto_bruto": 16969,
    "fase3/conjunto_depurado": 16037,
    "fase3/conjunto_bruto_v2": 18467,
    "fase3/conjunto_depurado_v2": 17527,
}
EXCLUIR = {".venv", "__pycache__", ".git", ".ipynb_checkpoints"}
UMBRAL_COLAPSO = 50   # carpetas con mas archivos se muestran solo con su conteo


def leer_inventario(bitacora):
    texto = bitacora.read_text(encoding="utf-8")
    ini = texto.find("## Anexo — Inventario de archivos generados")
    fin = texto.find("## Anexo —", ini + 10)
    bloque = texto[ini:fin if fin > 0 else None]
    rutas = []
    for linea in bloque.splitlines():
        m = re.match(r"\|\s*([^|]*)\|\s*(F\d)\s*\|\s*`([^`]+)`\s*([^|]*)\|", linea)
        if m:
            rutas.append({"fecha": m.group(1).strip(), "fase": m.group(2),
                          "ruta": m.group(3).strip().rstrip("/"),
                          "es_dir": m.group(3).strip().endswith("/"),
                          "version": m.group(4).strip()})
    return rutas


def archivos_en(raiz):
    for p in raiz.rglob("*"):
        if any(parte in EXCLUIR for parte in p.relative_to(raiz).parts):
            continue
        if p.is_file():
            yield p


def fecha(p):
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def arbol(raiz):
    lineas = ["."]
    n_dirs, n_arch = 0, 0

    def rec(d, prefijo):
        nonlocal n_dirs, n_arch
        hijos = sorted((h for h in d.iterdir() if h.name not in EXCLUIR
                        and not h.name.startswith(".")), key=lambda h: h.name.lower())
        for i, h in enumerate(hijos):
            ult = i == len(hijos) - 1
            rama, sig = ("└── ", "    ") if ult else ("├── ", "│   ")
            if h.is_dir():
                n_dirs += 1
                directos = sum(1 for x in h.iterdir() if x.is_file())
                if directos > UMBRAL_COLAPSO:
                    total = sum(1 for _ in archivos_en(h))
                    lineas.append(f"{prefijo}{rama}{h.name}/  [{total} archivos, colapsada]")
                    n_arch += total
                else:
                    lineas.append(f"{prefijo}{rama}{h.name}")
                    rec(h, prefijo + sig)
            else:
                n_arch += 1
                lineas.append(f"{prefijo}{rama}{h.name}")

    rec(raiz, "")
    lineas += ["", f"{n_dirs} directories, {n_arch} files",
               f"Generado: {datetime.now():%Y-%m-%d %H:%M}. Excluidas: {', '.join(sorted(EXCLUIR))}.",
               f"Carpetas con mas de {UMBRAL_COLAPSO} archivos directos se muestran colapsadas con su conteo."]
    return "\n".join(lineas) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default=".", help="raiz del proyecto")
    ap.add_argument("--regenerar", action="store_true",
                    help="reescribe estructura_proyecto.txt")
    args = ap.parse_args()
    raiz = Path(args.raiz).expanduser().resolve()
    out = []
    p = out.append

    inv = leer_inventario(raiz / "bitacora_experimentos.md")
    p(f"AUDITORIA DE ESTRUCTURA — {raiz}\n{datetime.now():%Y-%m-%d %H:%M}\n")

    # 1. Inventario contra disco
    p("=" * 70 + "\n1. RUTAS DEL INVENTARIO\n" + "=" * 70)
    faltan = 0
    for e in inv:
        ruta = raiz / e["ruta"]
        estado = "OK   " if ruta.exists() else "FALTA"
        faltan += not ruta.exists()
        fm = fecha(ruta) if ruta.exists() else "—"
        p(f"  {estado} {e['fase']}  fecha_bitacora={e['fecha'] or '_____':<12} "
          f"mtime={fm:<16} {e['ruta']}{'/' if e['es_dir'] else ''} {e['version']}")
    p(f"\n  Entradas: {len(inv)}   Ausentes: {faltan}")

    # 2. Conjuntos de datos
    p("\n" + "=" * 70 + "\n2. CONJUNTOS DE DATOS\n" + "=" * 70)
    for rel, esperado in ESPERADOS.items():
        d = raiz / rel
        if not d.exists():
            p(f"  FALTA {rel}/  (esperados {esperado})")
            continue
        arch = list(archivos_en(d))
        enlaces = sum(1 for a in arch if a.stat().st_nlink > 1)
        por_sub = defaultdict(int)
        for a in arch:
            por_sub[a.relative_to(d).parts[0]] += 1
        ok = "OK  " if len(arch) == esperado else "DIFF"
        p(f"  {ok} {rel}/  {len(arch)} archivos (esperados {esperado}), "
          f"enlaces duros: {enlaces}")
        for s, c in sorted(por_sub.items()):
            p(f"         {s}: {c}")

    # 3. Archivos no inventariados
    p("\n" + "=" * 70 + "\n3. ARCHIVOS EN DISCO NO INVENTARIADOS\n" + "=" * 70)
    declarados = [e["ruta"] for e in inv]
    sin_inv = []
    for a in archivos_en(raiz):
        rel = a.relative_to(raiz).as_posix()
        if not any(rel == d or rel.startswith(d + "/") for d in declarados):
            sin_inv.append(rel)
    for r in sorted(sin_inv):
        p(f"  {r}")
    p(f"\n  Total: {len(sin_inv)}")

    # 4. Pares de verificacion_dup
    p("\n" + "=" * 70 + "\n4. PARES EN fase3/verificacion_dup/\n" + "=" * 70)
    vd = raiz / "fase3/verificacion_dup"
    if vd.exists():
        pares = defaultdict(set)
        for a in vd.iterdir():
            m = re.match(r"(par\d+)_[^_]+_([AB])_", a.name)
            if m:
                pares[m.group(1)].add(m.group(2))
        for par in sorted(pares, key=lambda s: int(s[3:])):
            lados = pares[par]
            p(f"  {'OK      ' if lados == {'A', 'B'} else 'INCOMPL.'} {par}: {''.join(sorted(lados))}")

    # 5. Muestra visual por repositorio
    p("\n" + "=" * 70 + "\n5. MUESTRA VISUAL POR REPOSITORIO\n" + "=" * 70)
    mv = raiz / "fase3/analisis_nitidez/muestra_visual"
    if mv.exists():
        for s in sorted(x for x in mv.iterdir() if x.is_dir()):
            etiquetas = defaultdict(int)
            for a in s.iterdir():
                m = re.match(r"var[\d.]+_(\w+?)_", a.name)
                if m:
                    etiquetas[m.group(1)] += 1
            total = sum(etiquetas.values())
            detalle = ", ".join(f"{k}={v}" for k, v in sorted(etiquetas.items()))
            p(f"  {s.name:<16} {total:>3}  ({detalle})")

    informe = "\n".join(out) + "\n"
    print(informe)
    (raiz / "auditoria_estructura.txt").write_text(informe, encoding="utf-8")
    print(f"Guardado: {raiz / 'auditoria_estructura.txt'}")

    if args.regenerar:
        (raiz / "estructura_proyecto.txt").write_text(arbol(raiz), encoding="utf-8")
        print(f"Regenerado: {raiz / 'estructura_proyecto.txt'}")


if __name__ == "__main__":
    main()
