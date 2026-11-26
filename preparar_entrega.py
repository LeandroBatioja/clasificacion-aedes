"""
============================================================
 PREPARAR LA ENTREGA: REPOSITORIO DE GITHUB Y CARPETA DE ONEDRIVE
============================================================
 Comparacion de arquitecturas de aprendizaje profundo para
 la clasificacion de mosquitos vectores del dengue
 Leandro Steven Batioja Mena - PUCE Esmeraldas 2026-2027

 Crea, sin modificar nada del proyecto:

   entrega/github/    repositorio publico: codigo, protocolos, bitacora,
                      listas (CSV) y resultados. Sin imagenes, sin
                      modelos y sin PDF de la bibliografia.
   entrega/onedrive/  carpeta para el tutor, ordenada por documentos:
                      articulo, documentacion, resultados, conjunto de
                      datos, modelos de las corridas medianas, codigo y
                      bibliografia.

 Nunca copia las carpetas de imagenes (conjunto_bruto*, conjunto_depurado*,
 conjunto_final, cache*): son enlaces duros y la copia ocuparia ~29 GB.

 USO (desde ~/Imagenes/tesis)
   python preparar_entrega.py --articulo ~/Descargas/Batioja_Leandro_Articulo_completo.docx
   python preparar_entrega.py --simular          (solo muestra que copiaria)
   python preparar_entrega.py --todos-los-modelos (los 20 checkpoints en OneDrive)
============================================================
"""

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

LIMITE_GITHUB_MB = 50          # GitHub rechaza archivos de mas de 100 MB y avisa desde 50 MB
ARQS = ["efficientnet_b0", "mobilenetv3_large_100", "resnet50", "vit_small_patch16_224"]
PROTOCOLO_SHA = "0a88a99c6235d9f68432fd908e25f3cb3c7ee927954e7cfda98f98618376cb31"
ANEXO_SHA = "e1695322fcc6b7868fb6f00145b4eb2d19fb8c2f48d443151569ac394709ebfe"

# carpetas que nunca se recorren (imagenes, entornos, caches, salidas)
CARPETAS_FUERA = ["conjunto_bruto*", "conjunto_depurado*", "conjunto_final*", "cache*", ".venv", "venv",
                  "__pycache__", ".git", "entrega", "DATASETS", ".ipynb_checkpoints"]
# archivos de terceros que no se redistribuyen en el repositorio (se descargan de su fuente)
TERCEROS = {"mosquito_alert_photos_clean.csv": "metadatos de terceros"}
# extensiones que van al repositorio
EXT_GITHUB = {".py", ".md", ".txt", ".csv", ".json", ".tsv", ".yaml", ".yml", ".cfg", ".toml"}


def sha256(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def mb(n):
    return n / 2 ** 20


def fuera(rel):
    partes = Path(rel).parts
    return any(fnmatch.fnmatch(p, pat) for p in partes for pat in CARPETAS_FUERA)


def recorrer(raiz):
    """Recorre el proyecto sin entrar en las carpetas excluidas (no lee las de imagenes)."""
    for actual, dirs, files in os.walk(raiz):
        dirs[:] = sorted(d for d in dirs if not any(fnmatch.fnmatch(d, pat) for pat in CARPETAS_FUERA))
        for f in sorted(files):
            p = Path(actual) / f
            if p.is_symlink() or not p.is_file():
                continue
            yield p, p.relative_to(raiz)


# ------------------------------------------------------------
def regla_github(p, rel):
    """Devuelve None si entra, o el motivo de exclusion."""
    s = rel.as_posix()
    if p.name in TERCEROS:
        return TERCEROS[p.name]
    if s.startswith("bibliografia/") and p.suffix.lower() == ".pdf":
        return "PDF con derechos de autor"
    if s.startswith("fase6/Dataset_Dengue_Aedes_Ecuador/"):
        return None if p.name in ("metadata.csv", "README.md") else "imagen del conjunto ecuatoriano"
    if p.suffix.lower() == ".png" and "/figuras/" in "/" + s:
        return None                                    # graficos generados (no son fotografias)
    if p.suffix.lower() not in EXT_GITHUB:
        return f"extension {p.suffix or '(sin extension)'}"
    if p.stat().st_size > LIMITE_GITHUB_MB * 2 ** 20:
        return f"archivo grande ({mb(p.stat().st_size):.0f} MB)"
    return None


GITIGNORE = """# Entorno y caches
.venv/
venv/
__pycache__/
*.pyc
fase4/cache_512/
fase6/cache_ecuador_512/

# Imagenes: no se redistribuyen (se obtienen de los repositorios de origen)
fase3/conjunto_bruto*/
fase3/conjunto_depurado*/
fase3/conjunto_final/
*.jpg
*.jpeg
*.JPG
*.tif
*.tiff
*.webp
fase6/Dataset_Dengue_Aedes_Ecuador/*/

# Modelos (se publican como release)
*.pt
*.pth
*.npz

# Paquetes y documentos binarios
*.zip
*.pdf
*.docx
entrega/
"""

LICENCIA_MIT = """MIT License

Copyright (c) 2026 Leandro Steven Batioja Mena

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def readme(resumen, sha_p, sha_a):
    tabla = ""
    if resumen:
        filas = []
        for a, d in resumen.items():
            filas.append(f"| {d['nombre']} | {d['f1']} | {d['estab']} | {d['ms']} |")
        tabla = ("| Arquitectura | Macro F1 en prueba (corrida mediana) | Macro F1 en 5 corridas | ms por imagen (CPU) |\n"
                 "|---|---|---|---|\n" + "\n".join(filas) + "\n")
    return f"""# Clasificación de *Aedes aegypti* y *Aedes albopictus* con aprendizaje profundo

Comparación de EfficientNet-B0, MobileNetV3-Large, ResNet50 y ViT-S/16 para clasificar los dos vectores del dengue a
partir de imágenes de repositorios públicos, con un conjunto construido para que el tipo de fotografía no delate la
especie.

**Autor:** Leandro Steven Batioja Mena — Pontificia Universidad Católica del Ecuador, Sede Esmeraldas (2026–2027)

## Resultados principales

{tabla}
Evaluación sobre 589 imágenes de prueba, un subconjunto de control (Dryad y Ong y Ahmad, 254 imágenes) y un conjunto
de validación contextual de Ecuador. El detalle está en `bitacora_experimentos.md` (Entradas 012 a 015) y en `fase6/`.

## Contenido

| Carpeta | Contenido |
|---|---|
| `fase2/` | Entorno (`requirements.txt`), ficha técnica, criterios de depuración y protocolos de evaluación |
| `fase3/` | Construcción del conjunto: consolidación, depuración, agrupación por espécimen, verificación de etiquetas, partición y cajas de Grad-CAM. Incluye la procedencia de cada imagen y la partición (`fase3/particion/particion.csv`) |
| `fase4/` | `entrenar.py` (piloto, curva de aprendizaje, búsqueda y corridas definitivas) y resultados del piloto |
| `fase5/` | Búsqueda de hiperparámetros y resumen de las 20 corridas definitivas |
| `fase6/` | `evaluar.py` y resultados de la evaluación: métricas, pruebas estadísticas, Grad-CAM, conjunto ecuatoriano, eficiencia y figuras |
| `bitacora_experimentos.md` | Registro cronológico de todas las decisiones, incidencias y resultados |

## Protocolos registrados antes de evaluar

| Documento | SHA-256 |
|---|---|
| `fase2/protocolo_evaluacion.md` (v1.0, antes de la partición) | `{sha_p}` |
| `fase2/protocolo_evaluacion_anexo_fase6.md` (v1.0, antes de evaluar) | `{sha_a}` |

Comprobación: `sha256sum fase2/protocolo_evaluacion.md fase2/protocolo_evaluacion_anexo_fase6.md`

## Qué no incluye este repositorio

- **Imágenes.** No se redistribuyen. `fase3/particion/particion.csv` y los archivos de procedencia identifican cada
  imagen por su repositorio de origen, su identificador y su licencia, de modo que pueden obtenerse de las fuentes:
  Dryad (Nolte et al., doi:10.5061/dryad.b8gtht7mx), Ong y Ahmad (doi:10.1038/s41597-022-01541-w), Mosquito Alert
  (EMBL-EBI BioImage Archive, S-BIAD249) e iNaturalist.
- **Modelos entrenados.** Los *checkpoints* de las corridas medianas se publican como *release* del repositorio.
- **Bibliografía en PDF**, por derechos de autor.
- **Relación foto–reporte de Mosquito Alert** (`fase3/agrupacion/mosquito_alert_photos_clean.csv`), usada para agrupar
  las fotografías por reporte. Es un archivo público del servidor de Mosquito Alert, descargado el 23/10/2026; el
  resultado de la agrupación sí se incluye (`fase3/agrupacion/especimenes.csv`).

## Reproducción

Entorno: Python 3.11.13, PyTorch 2.4.1 (CPU), timm 1.0.9, scikit-learn 1.5.2, grad-cam 1.5.4.

```bash
python3.11 -m venv fase2/.venv && source fase2/.venv/bin/activate
pip install -r fase2/requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

Orden de ejecución (detalle y parámetros en la bitácora):

1. `fase3/consolidar_conjunto.py` → `fase3/analizar_nitidez.py` → `fase3/depurar_conjunto.py`
2. `fase3/descargar_inaturalist.py` y `fase3/agrupar_especimenes.py`
3. `fase3/verificacion_morfologica.py` → `fase3/corregir_verificacion.py` → `fase3/revisar_estadio_inat.py`
4. `fase3/particion_conjunto.py` y `fase3/anotar_cajas.py`
5. `fase4/entrenar.py --modo piloto`, `--modo curva`, `--modo busqueda` y `--modo definitiva`
6. `fase6/evaluar.py`

Todas las selecciones aleatorias usan la semilla 42; las corridas definitivas, las semillas 42 a 46.

## Licencia

Código bajo licencia MIT (`LICENSE`). Las imágenes conservan la licencia de su repositorio de origen.
"""


def leer_resumen(raiz):
    try:
        rep = json.load(open(raiz / "fase6/reporte_fase6.json", encoding="utf-8"))
        efi = rep.get("eficiencia", {})
        nombres = {"efficientnet_b0": "EfficientNet-B0", "mobilenetv3_large_100": "MobileNetV3-Large",
                   "resnet50": "ResNet50", "vit_small_patch16_224": "ViT-S/16"}
        out = {}
        for a in ARQS:
            m = rep["metricas_prueba"][a]
            e = rep["estabilidad"][a]["prueba"]
            out[a] = {"nombre": nombres[a], "f1": f"{m['macro_f1']:.3f}".replace(".", ","),
                      "estab": f"{e['media']:.3f} ± {e['de']:.3f}".replace(".", ","),
                      "ms": f"{efi[a]['ms_por_imagen_lote1_mediana']:.1f}".replace(".", ",") if a in efi else "—"}
        return out, rep.get("corrida_mediana", {})
    except Exception:
        return {}, {}


# ------------------------------------------------------------
# OneDrive: (patron relativo, carpeta destino)
ONEDRIVE = [
    ("bitacora_experimentos.md", "02_documentacion"),
    ("fase2/ficha_tecnica.md", "02_documentacion"),
    ("fase2/protocolo_evaluacion.md", "02_documentacion"),
    ("fase2/protocolo_evaluacion_anexo_fase6.md", "02_documentacion"),
    ("fase2/criterios_depuracion.md", "02_documentacion"),
    ("fase2/requirements.txt", "02_documentacion"),
    ("estructura_proyecto.txt", "02_documentacion"),
    ("fase6/log_evaluacion.txt", "03_resultados/fase6_evaluacion"),
    ("fase6/*.json", "03_resultados/fase6_evaluacion"),
    ("fase6/*.csv", "03_resultados/fase6_evaluacion"),
    ("fase6/figuras/*", "03_resultados/fase6_evaluacion/figuras"),
    ("fase6/gradcam_laminas/*", "03_resultados/fase6_evaluacion/gradcam_laminas"),
    ("fase5/log_definitivas.txt", "03_resultados/fase5_entrenamiento"),
    ("fase5/log_busqueda.txt", "03_resultados/fase5_entrenamiento"),
    ("fase5/definitivas/resumen_definitivas.json", "03_resultados/fase5_entrenamiento"),
    ("fase5/busqueda/reporte_busqueda.json", "03_resultados/fase5_entrenamiento"),
    ("fase5/busqueda/configuracion_seleccionada.json", "03_resultados/fase5_entrenamiento"),
    ("fase4/log_piloto.txt", "03_resultados/fase4_piloto"),
    ("fase4/log_curva.txt", "03_resultados/fase4_piloto"),
    ("fase4/piloto/reporte_piloto.json", "03_resultados/fase4_piloto"),
    ("fase4/curva/reporte_curva.json", "03_resultados/fase4_piloto"),
    ("fase3/particion/*.csv", "04_conjunto_de_datos/particion"),
    ("fase3/particion/*.json", "04_conjunto_de_datos/particion"),
    ("fase3/particion/cajas_laminas/*", "04_conjunto_de_datos/particion/cajas_laminas"),
    ("fase3/procedencia_conjunto_bruto_v2.csv", "04_conjunto_de_datos/procedencia"),
    ("fase3/inaturalist/procedencia_inaturalist.csv", "04_conjunto_de_datos/procedencia"),
    ("fase3/depuracion_v2/registro_imagenes_eliminadas.csv", "04_conjunto_de_datos/depuracion"),
    ("fase3/depuracion_v2/reporte_depuracion.json", "04_conjunto_de_datos/depuracion"),
    ("fase3/agrupacion/ong_uniones_manuales.csv", "04_conjunto_de_datos/agrupacion"),
    ("fase3/agrupacion/reporte_agrupacion.json", "04_conjunto_de_datos/agrupacion"),
    ("fase3/verificacion_morfologica/*.csv", "04_conjunto_de_datos/verificacion_morfologica"),
    ("fase3/verificacion_morfologica/*.json", "04_conjunto_de_datos/verificacion_morfologica"),
    ("fase3/verificacion_morfologica/revision/*.pdf", "04_conjunto_de_datos/verificacion_morfologica/revision"),
    ("fase6/Dataset_Dengue_Aedes_Ecuador/metadata.csv", "04_conjunto_de_datos/ecuador"),
    ("fase6/Dataset_Dengue_Aedes_Ecuador/README.md", "04_conjunto_de_datos/ecuador"),
]


def leeme(articulo, medianas, todos, enlace):
    m = "\n".join(f"     - {a}_semilla_{s}.pt" for a, s in medianas.items()) or "     (no encontrados)"
    return f"""ENTREGA DE LA TESIS - Leandro Steven Batioja Mena (PUCE Esmeraldas)
Comparación de arquitecturas de aprendizaje profundo para la clasificación de
Aedes aegypti y Aedes albopictus, vectores del dengue
Preparada el {datetime.now().strftime('%d/%m/%Y')}

Repositorio del código: {enlace}

CONTENIDO

01_articulo/
   Artículo con resultados, discusión y conclusiones{' (' + articulo + ')' if articulo else ' (no incluido)'}.

02_documentacion/
   bitacora_experimentos.md   registro de todas las fases (Entradas 001 a 016)
   ficha_tecnica.md           configuración experimental y rendimiento medido
   protocolo_evaluacion.md    protocolo registrado antes de la partición
                              (SHA-256 {PROTOCOLO_SHA[:8]}...)
   protocolo_evaluacion_anexo_fase6.md
                              detalles fijados antes de evaluar
                              (SHA-256 {ANEXO_SHA[:8]}...)
   Los .md se leen con cualquier editor de texto o en GitHub.

03_resultados/
   fase6_evaluacion/   métricas en prueba, control, Grad-CAM, Ecuador,
                       pruebas estadísticas, eficiencia, figuras y láminas
   fase5_entrenamiento/  búsqueda de hiperparámetros y 20 corridas definitivas
   fase4_piloto/       prueba piloto y curva de aprendizaje

04_conjunto_de_datos/
   Partición (qué imagen está en entrenamiento, validación o prueba),
   procedencia y licencia de cada imagen, imágenes eliminadas por criterio,
   verificación de etiquetas (con los tres documentos de revisión en PDF),
   cajas de Grad-CAM y metadatos del conjunto ecuatoriano. Las imágenes no se incluyen: se obtienen de los repositorios
   de origen.

05_modelos/
   Checkpoints de la corrida mediana de cada arquitectura{' (y los 20 en todas_las_corridas/)' if todos else ''}:
{m}

06_codigo/
   codigo_repositorio.zip: copia del repositorio de GitHub.

07_bibliografia/
   PDF de las referencias del plan, con su índice (si están disponibles).
"""


# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default=".")
    ap.add_argument("--salida", default="entrega")
    ap.add_argument("--articulo", default=None, help="ruta del .docx del artículo")
    ap.add_argument("--enlace", default="(añadir el enlace de GitHub)")
    ap.add_argument("--todos-los-modelos", action="store_true")
    ap.add_argument("--simular", action="store_true")
    args = ap.parse_args()

    raiz = Path(args.raiz).resolve()
    if not (raiz / "bitacora_experimentos.md").exists() or not (raiz / "fase4").exists():
        sys.exit(f"ERROR: ejecute desde la carpeta del proyecto (no se encontró la bitácora en {raiz})")
    sal = raiz / args.salida
    gh, od = sal / "github", sal / "onedrive"
    if sal.exists() and not args.simular:
        sys.exit(f"ERROR: {sal} ya existe. Bórrela o use --salida otra_carpeta")

    print("=" * 66)
    print("  PREPARAR ENTREGA" + ("  (SIMULACION)" if args.simular else ""))
    print("=" * 66)
    sha_p, sha_a = sha256(raiz / "fase2/protocolo_evaluacion.md"), sha256(raiz / "fase2/protocolo_evaluacion_anexo_fase6.md")
    print(f"  Protocolo v1.0: {'coincide' if sha_p == PROTOCOLO_SHA else 'NO COINCIDE'}   "
          f"Anexo: {'coincide' if sha_a == ANEXO_SHA else 'NO COINCIDE'}")

    # ---------- GitHub ----------
    entran, excluidos = [], {}
    for p, rel in recorrer(raiz):
        motivo = regla_github(p, rel)
        if motivo is None:
            entran.append((p, rel))
        else:
            excluidos.setdefault(motivo.split(" (")[0], []).append((rel, p.stat().st_size))
    total = sum(p.stat().st_size for p, _ in entran)
    print(f"\n  GITHUB: {len(entran)} archivos, {mb(total):.1f} MB")
    for carpeta in sorted({rel.parts[0] if len(rel.parts) > 1 else "(raíz)" for _, rel in entran}):
        n = [r for _, r in entran if (r.parts[0] if len(r.parts) > 1 else "(raíz)") == carpeta]
        print(f"    {carpeta:<28}{len(n):>6} archivos")
    print("  Excluidos de GitHub:")
    for motivo, lst in sorted(excluidos.items(), key=lambda kv: -len(kv[1])):
        print(f"    {motivo:<40}{len(lst):>7} archivos  {mb(sum(s for _, s in lst)):>9.1f} MB")
    grandes = excluidos.get("archivo grande", [])
    for rel, s in grandes:
        print(f"      grande: {rel} ({mb(s):.0f} MB)")

    resumen, medianas = leer_resumen(raiz)
    if not args.simular:
        for p, rel in entran:
            d = gh / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, d)
        (gh / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
        (gh / "README.md").write_text(readme(resumen, sha_p, sha_a), encoding="utf-8")
        (gh / "LICENSE").write_text(LICENCIA_MIT, encoding="utf-8")
        # el anexo y el protocolo deben conservar su huella
        assert sha256(gh / "fase2/protocolo_evaluacion.md") == sha_p

    # ---------- OneDrive ----------
    copias = []
    for patron, destino in ONEDRIVE:
        hallados = [p for p in sorted(raiz.glob(patron)) if p.is_file()]
        if not hallados:
            print(f"    (no encontrado: {patron})")
        copias += [(p, Path(destino) / p.name) for p in hallados]
    if args.articulo:
        art = Path(args.articulo).expanduser()
        if art.exists():
            copias.append((art, Path("01_articulo") / art.name))
        else:
            print(f"    (no encontrado el artículo: {art})")
    for a, s in medianas.items():
        ck = raiz / "fase5/definitivas" / a / f"semilla_{s}" / "modelo.pt"
        if ck.exists():
            copias.append((ck, Path("05_modelos") / f"{a}_semilla_{s}.pt"))
    if args.todos_los_modelos:
        for ck in sorted(raiz.glob("fase5/definitivas/*/semilla_*/modelo.pt")):
            copias.append((ck, Path("05_modelos/todas_las_corridas") / f"{ck.parent.parent.name}_{ck.parent.name}.pt"))
    bib = raiz / "bibliografia"
    if bib.exists():
        for p in sorted(bib.iterdir()):
            if p.is_file():
                copias.append((p, Path("07_bibliografia") / p.name))
    total_od = sum(p.stat().st_size for p, _ in copias)
    print(f"\n  ONEDRIVE: {len(copias)} archivos, {mb(total_od):.1f} MB, más el código comprimido")
    for carpeta in sorted({d.parts[0] for _, d in copias}):
        n = [p for p, d in copias if d.parts[0] == carpeta]
        print(f"    {carpeta:<28}{len(n):>6} archivos  {mb(sum(p.stat().st_size for p in n)):>9.1f} MB")

    if args.simular:
        print("\n  Simulación: no se copió nada.")
        return
    for p, d in copias:
        (od / d).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, od / d)
    (od / "06_codigo").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(od / "06_codigo" / "codigo_repositorio.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(gh.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(gh))
    (od / "00_LEEME.txt").write_text(
        leeme(Path(args.articulo).name if args.articulo else "", medianas, args.todos_los_modelos, args.enlace),
        encoding="utf-8")
    print(f"\n  Listo:\n    {gh}\n    {od}")


if __name__ == "__main__":
    main()
