# Clasificación de *Aedes aegypti* y *Aedes albopictus* con aprendizaje profundo

Comparación de EfficientNet-B0, MobileNetV3-Large, ResNet50 y ViT-S/16 para clasificar los dos vectores del dengue a
partir de imágenes de repositorios públicos, con un conjunto construido para que el tipo de fotografía no delate la
especie.

**Autor:** Leandro Steven Batioja Mena — Pontificia Universidad Católica del Ecuador, Sede Esmeraldas (2026–2027)

## Resultados principales

| Arquitectura | Macro F1 en prueba (corrida mediana) | Macro F1 en 5 corridas | ms por imagen (CPU) |
|---|---|---|---|
| EfficientNet-B0 | 0,825 | 0,824 ± 0,028 | 17,6 |
| MobileNetV3-Large | 0,808 | 0,813 ± 0,010 | 12,5 |
| ResNet50 | 0,815 | 0,819 ± 0,009 | 50,4 |
| ViT-S/16 | 0,854 | 0,855 ± 0,005 | 43,7 |

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
| `fase2/protocolo_evaluacion.md` (v1.0, antes de la partición) | `0a88a99c6235d9f68432fd908e25f3cb3c7ee927954e7cfda98f98618376cb31` |
| `fase2/protocolo_evaluacion_anexo_fase6.md` (v1.0, antes de evaluar) | `e1695322fcc6b7868fb6f00145b4eb2d19fb8c2f48d443151569ac394709ebfe` |

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
