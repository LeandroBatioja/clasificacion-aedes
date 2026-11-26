# Protocolo de evaluación — Anexo de la Fase 6

**Proyecto:** Comparación de arquitecturas de aprendizaje profundo para la clasificación de mosquitos vectores del dengue
**Autor:** Leandro Steven Batioja Mena — PUCE Esmeraldas
**Versión:** 1.0 — 2026-11-24
**Ubicación:** `fase2/protocolo_evaluacion_anexo_fase6.md`
**Complementa a:** `fase2/protocolo_evaluacion.md` v1.0 (SHA-256 `0a88a99c…cb31`), que no se modifica

---

## 0. Estatus

Este anexo fija los detalles de aplicación que el plan (3.7 y 3.8) y el protocolo v1.0 dejan abiertos. Se registra **antes de evaluar ningún modelo sobre el conjunto de prueba o el conjunto ecuatoriano**. A la fecha de registro:

- existen los 20 *checkpoints* de la Fase 5, seleccionados solo con validación;
- del conjunto ecuatoriano se revisaron `metadata.csv` y una hoja de miniaturas de las 127 imágenes, sin ninguna predicción.

No añade ni cambia umbrales de decisión del protocolo v1.0. Su huella SHA-256 se registra en la bitácora.

---

## 1. Modelos y niveles de análisis (plan 3.8)

| Nivel | Unidad | Modelos | Uso |
|---|---|---|---|
| 1. Estabilidad | Corrida | Los 5 de cada arquitectura (semillas 42–46) | Media, desviación estándar (n − 1), mínimo y máximo de cada métrica en prueba. Sin pruebas de significancia |
| 2. Comparación | Imagen de prueba | Corrida mediana de cada arquitectura según macro F1 en validación: EfficientNet-B0 semilla 46, MobileNetV3-Large 43, ResNet50 44 y ViT-S/16 46 (Entrada 011) | Métricas, pruebas estadísticas, protocolo v1.0 y validación contextual |

Las imágenes se procesan igual que en validación: orientación EXIF, copia con lado mayor ≤ 512 px y redimensionado bicúbico a 224 × 224 con la normalización de cada modelo.

---

## 2. Pruebas estadísticas (plan 3.8)

| Elemento | Aplicación |
|---|---|
| Q de Cochran | Sobre la matriz acierto/error de las 589 imágenes × 4 arquitecturas (`statsmodels`) |
| W de Cochran | W = Q / (n · (k − 1)), con n = 589 y k = 4 |
| McNemar | Exacta (binomial) para los 6 pares. Corrección de Bonferroni: p × 6, con máximo 1. Los pares se interpretan solo si la Q de Cochran es significativa (p < 0,05) |
| Diferencia de proporciones | Diferencia de exactitudes del par, con IC 95 % de Wald para proporciones pareadas |
| Análisis de sensibilidad | La misma diferencia con IC 95 % por *bootstrap* de grupos, porque las pruebas anteriores tratan cada imagen como independiente y varias imágenes pueden ser del mismo espécimen. Si las dos lecturas discrepan, se reportan ambas |

---

## 3. *Bootstrap* por grupos (protocolo v1.0, apartados 2 y 3)

- Unidad de remuestreo: el grupo (espécimen) de `particion.csv`.
- Remuestreo **estratificado por estrato** (especie × repositorio): en cada réplica se toman con reemplazo tantos grupos de cada estrato como tiene la prueba. Así se mantiene la composición fijada por diseño.
- 2.000 réplicas, semilla 42. Las mismas réplicas se usan para las cuatro arquitecturas y para todos los subconjuntos, de modo que Δ_control y las diferencias entre arquitecturas son pareadas.
- IC 95 % por percentiles (2,5 y 97,5).

---

## 4. Jerarquía de selección (plan 3.8): reglas de aplicación

1. **Macro F1** en prueba (corrida mediana). Se consideran empatadas con la mejor las arquitecturas cuya diferencia sea menor de 0,02 (2 puntos porcentuales).
2. Entre las empatadas, **sensibilidad de *Ae. aegypti***. Hay empate si la diferencia con la mejor es menor de 0,02.
3. Entre las que sigan empatadas, **menor desviación estándar del macro F1 en prueba** de las 5 corridas.
4. Solo si la desviación estándar coincide a cuatro decimales: **menor tiempo de inferencia por imagen** y, después, menor tamaño del modelo.

La jerarquía se aplica también sobre el subconjunto de control, como análisis de sensibilidad (protocolo v1.0, apartado 3).

---

## 5. Eficiencia computacional

Cada arquitectura se mide en **un proceso aparte**, cargando el *checkpoint* de su corrida mediana. Así se corrige la medida acumulada de la Fase 5 (bitácora, incidencia del 2026-11-24).

| Medida | Procedimiento |
|---|---|
| Tiempo de inferencia por imagen | Lote de 1 imagen, tensores ya en memoria, 6 hilos; 10 imágenes de calentamiento y mediana de 100 imágenes de prueba |
| Rendimiento en lote | Lotes de 64 con las mismas imágenes; ms por imagen |
| RAM | RAM residente tras la inferencia y RAM residente base (librerías cargadas, sin modelo). La diferencia es la RAM atribuible al modelo y a la inferencia. Se reporta también el pico del proceso |
| Tamaño | Tamaño del archivo del *checkpoint* y número de parámetros |

---

## 6. Validación contextual: conjunto ecuatoriano

### 6.1 Composición observada antes de evaluar

| | Imágenes | Observaciones |
|---|---|---|
| *Ae. aegypti*, `research` | 71 | 19 |
| *Ae. albopictus*, `research` | 4 | 4 |
| *Ae. aegypti*, todas | 119 | 41 |
| *Ae. albopictus*, todas | 8 | 7 |

Las imágenes no son independientes: el conjunto incluye todas las fotografías de cada observación. Una sola observación (156329843, Manta, 2023) aporta **32 de las 71** imágenes `research` de *Ae. aegypti* (45 %). Son acercamientos al microscopio de partes de un mismo ejemplar: patas, tórax, abdomen y alas, muchas sin el mosquito completo. Un mismo observador aporta 7 de las 19 observaciones (43 imágenes).

### 6.2 Análisis principal (plan 3.8)

- ***Ae. aegypti*, 71 imágenes `research`:** sensibilidad (proporción clasificada como *Ae. aegypti*). En este subconjunto es igual a la exactitud.
- **Exactitud y F1 de *Ae. aegypti*** sobre las 75 imágenes `research`, con *Ae. aegypti* como clase positiva. Las 4 *Ae. albopictus* aportan los posibles falsos positivos.
- IC 95 % por *bootstrap* **por observación** (2.000 réplicas, semilla 42), estratificado por especie.
- Se comparan de forma descriptiva con la sensibilidad de *Ae. aegypti* en la prueba internacional: global y del estrato I-ae (iNaturalist fuera de Ecuador).

### 6.3 Análisis secundarios, fijados de antemano (descriptivos)

| Análisis | Motivo |
|---|---|
| **Por observación:** media de la proporción de aciertos de cada observación (cada observación pesa 1) y número de observaciones con mayoría de aciertos | Contrarresta el peso de las observaciones con muchas fotos |
| **Sin la observación 156329843** | Sus 32 imágenes son vistas parciales de un mismo ejemplar, distintas del resto de los conjuntos |
| ***Ae. aegypti*, 119 imágenes** (todas las calidades) | Análisis de sensibilidad previsto en el plan |
| ***Ae. albopictus*, 4 `research` y 8 en total** | Aciertos y errores caso por caso, sin métricas agregadas (plan 3.8). Las *Ae. albopictus* clasificadas como *Ae. aegypti* se leen además como posible atajo por plataforma (protocolo v1.0, apartado 4) |
| Estabilidad: sensibilidad de las 5 corridas de cada arquitectura sobre las 71 imágenes | Descriptivo |

No se excluye ninguna imagen del análisis principal: el plan fija el subconjunto `research`. Las imágenes no pasan por la depuración E1–E9; se registra como limitación.

### 6.4 Comprobación de fuga

Antes de evaluar, se cruzan `observation_id` y `photo_id` con `fase3/inaturalist/procedencia_inaturalist.csv`. Una imagen que coincida con el entrenamiento se excluye del análisis ecuatoriano y se reporta. La descarga de la Fase 3 excluyó Ecuador, así que no se espera ninguna coincidencia.

---

## 7. Salidas

`fase6/evaluar.py` escribe en `fase6/`:

| Archivo | Contenido |
|---|---|
| `predicciones/<arquitectura>/semilla_<s>.csv` | Predicción y probabilidad por imagen (prueba y Ecuador), para los 20 modelos |
| `estabilidad_5_corridas.csv` | Nivel 1: métricas de cada corrida en prueba y en el control |
| `metricas_por_estrato.csv` | Métricas por arquitectura y subconjunto, con IC (protocolo, 2) |
| `control_dryad_ong.json` | Δ_control, Δ_campo, IC y orden de arquitecturas (protocolo, 3) |
| `pruebas_estadisticas.json` | Q de Cochran, W y McNemar con Bonferroni |
| `seleccion_mejor_modelo.json` | Jerarquía aplicada en prueba y en control, paso a paso |
| `atajo_plataforma.csv` | M-ae y *Ae. albopictus* ecuatorianos, caso por caso (protocolo, 4) |
| `gradcam_indicadores.csv`, `gradcam_resumen.json`, `gradcam_laminas/` | Protocolo, 5 |
| `ecuador.json`, `ecuador_casos.csv` | Validación contextual |
| `eficiencia.json` | Apartado 5 |
| `figuras/` | Curvas de entrenamiento, macro F1 con IC y matrices de confusión |
| `reporte_fase6.json` | Todo lo anterior resumido, con las huellas SHA-256 de ambos protocolos |
