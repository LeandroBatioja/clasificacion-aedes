# Dataset de mosquitos vectores del DENGUE — Ecuador (Aedes aegypti y Aedes albopictus)

Conjunto de imágenes de los **dos mosquitos vectores del dengue**, con procedencia de
**Ecuador**, recopilado de iNaturalist bajo **licencia Creative Commons**. Preparado para
el estudio de clasificación de vectores del dengue mediante Deep Learning, delimitado a
estas dos especies.

## Contenido
| Carpeta | Imágenes |
|---|---:|
| `Aedes_aegypti/`    | 119 |
| `Aedes_albopictus/` | 8 |
| **TOTAL** | **127** |

De ese total, **75 imágenes tienen identificación confirmada** por la comunidad
(`quality_grade = research`); el resto está como `needs_id`. La columna `quality_grade`
del `metadata.csv` permite filtrar el subconjunto confirmado si se desea un set de prueba
más estricto.

## Procedencia
- Fuente: iNaturalist (API), país = Ecuador, género *Aedes*, especies *aegypti* y *albopictus*.
- Solo fotos con licencia Creative Commons (se excluyeron "all rights reserved").
- Tamaño de imagen: versión `large` de iNaturalist (~1024 px).

## ⚠️ Advertencias importantes
1. **Es un conjunto de validación / prueba con imágenes reales de campo**, no para entrenar
   desde cero. El entrenamiento debe apoyarse en datasets públicos grandes (p. ej. Mosquito
   Alert, IEEE DataPort, MosqVision) y luego validar aquí.
2. ***Aedes albopictus* tiene muy pocas imágenes (8)**: es la clase escasa. Debe reforzarse
   con datasets públicos y, de ser posible, mediante gestión con el INSPI / Centro Nacional
   de Referencia de Vectores.
3. Ambas son **especies** (no géneros): *Aedes aegypti* y *Aedes albopictus*, los dos
   vectores principales del dengue.

## metadata.csv (OBLIGATORIO citarlo)
Una fila por imagen con atribución y licencia: `filename, clase, taxon, taxon_rank,
quality_grade, photo_id, observation_id, license_code, attribution, observed_on,
place_guess, latitude, longitude, observation_url`.

## Atribución
Imágenes de observadores de iNaturalist bajo licencias Creative Commons (ver columna
`attribution` en `metadata.csv`). Fuente: iNaturalist.org.
