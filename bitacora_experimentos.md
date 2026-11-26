# Bitácora de experimentos

> **Instrumento 3.6.2.2** del plan de investigación
> **Proyecto:** Comparación de arquitecturas de aprendizaje profundo para la clasificación de mosquitos vectores del dengue mediante reconocimiento de imágenes
> **Autor:** Leandro Steven Batioja Mena — PUCE Esmeraldas 2026–2027
> **Período de ejecución:** septiembre 2026 – enero 2027

---

## Cómo usar esta bitácora

Cada actividad relevante se registra como una entrada con fecha. El propósito es la **trazabilidad**: cualquier tercero debe poder reconstruir qué se hizo, cuándo, con qué parámetros y qué resultó.

**Reglas de registro:**

1. Anotar **el mismo día**, no después. La memoria distorsiona.
2. Registrar también **lo que falló**. Los errores y sus soluciones son parte del método.
3. Cada decisión que se desvíe del plan debe quedar **justificada**.
4. Los números van completos, sin redondear a conveniencia.
5. Si una cifra del documento cambia por lo medido aquí, anotar **qué sección se actualizó**.

---

## Registro de avance por fase

| Fase | Denominación | Semanas | Estado | Fecha de cierre |
|---|---|---|---|---|
| F1 | Revisión y diagnóstico | S1–S3 | Completada | |
| F2 | Diseño metodológico | S4–S6 | Completada | |
| F3 | Construcción del conjunto de datos | S7–S10 | Completada (datos). Informe pendiente | 2026-10-24 |
| F4 | Prueba piloto | S11 | Completada | 2026-11-16 |
| F5 | Implementación y entrenamiento | S12–S16 | Completada | 2026-11-22 |
| F6 | Pruebas y recolección de datos | S17–S18 | Completada | 2026-11-25 |
| F7 | Análisis, recomendaciones y redacción | S19–S22 | En curso: análisis completado; borrador de resultados, discusión y conclusiones redactado; pendientes las correcciones y el traslado a la plantilla de la revista | |

---

# FASE 2 — Diseño metodológico

## Entrada 001 — Instalación del entorno

**Fecha:** 19 de septiembre de 2026
**Actividad:** Creación del entorno virtual e instalación de dependencias
**Duración:** 01:48 min

**Comandos ejecutados:**

```bash
sudo dnf install -y python3.11 python3.11-devel python3.11-pip
cd ~/Imágenes/tesis/fase2
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple
pip uninstall -y opencv-python-headless
```

**Resultado:** instalación completada con dos incidencias, ambas resueltas.

**Incidencia 1 — Versión de Python incompatible**

El sistema operativo AlmaLinux 9.8 incorpora Python 3.9.25 por defecto. La instalación falló con el mensaje `Could not find a version that satisfies the requirement scipy==1.14.1`, dado que scipy 1.14.1 exige Python ≥ 3.10. Se instaló Python 3.11 desde los repositorios oficiales de la distribución mediante `dnf install python3.11` y se recreó el entorno virtual con dicha versión, conforme a lo declarado en la sección 3.3 del plan. Versión final verificada: **Python 3.11.13**.

**Incidencia 2 — Coexistencia de dos versiones de OpenCV**

Tras la instalación se detectó la presencia simultánea de `opencv-python-headless 4.10.0.84`, declarada en `requirements.txt`, y `opencv-python 5.0.0.93`, instalada como dependencia transitiva del paquete `grad-cam`. La ambigüedad en la resolución del módulo `cv2` comprometía la reproducibilidad del entorno, ya que el intérprete podía cargar cualquiera de las dos. Se desinstaló la versión headless y se fijó `opencv-python 5.0.0.93` como única, actualizando el archivo `requirements.txt` en consecuencia.

**Desviaciones respecto a `requirements.txt`:**

| Librería | Versión declarada | Versión instalada | Motivo |
|---|---|---|---|
| opencv-python-headless | 4.10.0.84 | — (desinstalada) | Conflicto con la dependencia de `grad-cam` |
| opencv-python | — | 5.0.0.93 | Requerida por `grad-cam`; se fija como única |

---

## Entrada 002 — Verificación del entorno

**Fecha:** 19 de septiembre de 2026, 23:09
**Script:** `verificar_entorno.py`
**Salida guardada en:** `salidas/log_verificacion.txt`, `salidas/reporte_entorno.json`

**Datos del entorno:**

| Campo | Valor obtenido |
|---|---|
| Sistema operativo | Linux 5.14.0-687.46.1.el9_8.x86_64 (AlmaLinux 9.8) |
| Versión de Python | 3.11.13 |
| CPU | Intel Core i5-10400 @ 2.90 GHz |
| RAM total | 23,0 GB |
| Hilos de PyTorch | 6 |
| CUDA disponible | `False` — conforme a lo declarado |

**Arquitecturas cargadas:** las cuatro se cargaron correctamente con pesos de ImageNet-1k y capa clasificadora adaptada a dos salidas.

| Arquitectura | ID timm | Parámetros medidos (M) | Declarado en 2.2.8 (M) |
|---|---|---|---|
| EfficientNet-B0 | `efficientnet_b0` | 4,01 | ~5,3 |
| MobileNetV3-Large | `mobilenetv3_large_100` | 4,20 | ~5,4 |
| ResNet50 | `resnet50` | 23,51 | ~25,6 |
| ViT-S/16 | `vit_small_patch16_224` | 21,67 | ~22,0 |

**Hallazgo — Discrepancia en el conteo de parámetros**

Las cifras medidas resultan inferiores a las declaradas en la sección 2.2.8. La causa es que los valores citados en la literatura corresponden a los modelos con su capa clasificadora original de mil salidas, mientras que este estudio la sustituye por una de dos salidas. La diferencia es mayor en las arquitecturas ligeras, cuya capa final representa una proporción considerable del total. Ambos valores son correctos; corresponde precisar en el documento a cuál se refiere cada cifra.

**Parámetros de normalización registrados:**

| Arquitectura | Media (RGB) | Desviación estándar (RGB) |
|---|---|---|
| EfficientNet-B0 | 0,485 · 0,456 · 0,406 | 0,229 · 0,224 · 0,225 |
| MobileNetV3-Large | 0,485 · 0,456 · 0,406 | 0,229 · 0,224 · 0,225 |
| ResNet50 | 0,485 · 0,456 · 0,406 | 0,229 · 0,224 · 0,225 |
| ViT-S/16 | 0,500 · 0,500 · 0,500 | 0,500 · 0,500 · 0,500 |

**Observaciones:** ViT-S/16 emplea parámetros de normalización distintos a las tres arquitecturas convolucionales, lo que confirma la pertinencia de lo declarado en la sección 3.3 respecto a normalizar cada modelo con los valores que timm le asocia. La prueba funcional devolvió salidas de forma (2, 2) en las cuatro arquitecturas, confirmando la correcta adaptación a la clasificación binaria.

---

## Entrada 003 — Benchmark de rendimiento en CPU

**Fecha:** 19 de septiembre de 2026
**Script:** `benchmark_cpu.py`
**Parámetros:** batch = 16, lotes medidos = 5 (se reporta la mediana)
**Salida guardada en:** `salidas/log_benchmark.txt`, `salidas/reporte_benchmark.json`

**Rendimiento medido:**

| Arquitectura | Inferencia (img/s) | Entrenamiento (img/s) | RAM pico (MB) | Horas/corrida |
|---|---|---|---|---|
| EfficientNet-B0 | 43,7 | 39,3 | 399 | 1,14 |
| MobileNetV3-Large | 74,6 | 66,1 | 434 | 0,68 |
| ResNet50 | 17,1 | 16,6 | 506 | 2,71 |
| ViT-S/16 | 24,3 | 25,9 | 525 | 1,74 |

**Contraste con la estimación teórica previa:**

| Arquitectura | Estimado (img/s) | Medido (img/s) | Desviación |
|---|---|---|---|
| EfficientNet-B0 | 12,0 | 39,3 | +228 % |
| MobileNetV3-Large | 12,0 | 66,1 | +451 % |
| ResNet50 | 5,0 | 16,6 | +232 % |
| ViT-S/16 | 5,0 | 25,9 | +418 % |

**Presupuesto de cómputo:**

| Concepto | Valor |
|---|---|
| Horas totales requeridas (medido) | 35,1 |
| Horas declaradas previamente (estimación teórica) | 143,6 |
| Horas disponibles (Fase 5) | 240,0 |
| Margen | 204,9 h (+85 %) |

**Veredicto del script:** el presupuesto cierra con margen suficiente.

**Análisis crítico de la medición**

El resultado debe interpretarse con cautela por tres razones que lo hacen optimista:

1. La medición se realizó sobre **tensores sintéticos generados en memoria**, sin incorporar el costo de lectura de disco, decodificación de archivos JPEG ni aplicación del aumento de datos en tiempo de ejecución, operaciones que en un entorno sin GPU suelen representar una fracción considerable del tiempo total.
2. El ajuste fino aplicado durante la medición se restringió a la **capa clasificadora**, mientras que la sección 3.2 declara el entrenamiento de los últimos bloques junto con dicha capa, lo que incrementa el costo del paso hacia atrás.
3. La medición empleó cinco lotes por arquitectura, muestra adecuada para una estimación preliminar pero no para caracterizar la variabilidad del rendimiento a lo largo de un entrenamiento completo.

En consecuencia, el tiempo real esperado será superior al medido. No obstante, aun aplicando un factor de corrección conservador de tres, el requerimiento se situaría en torno a las 105 horas, holgadamente por debajo de las 240 disponibles. El presupuesto se sostiene.

**Decisión tomada:** se conservan los parámetros declarados en el plan (15 épocas, factor de aumento ×2, 5 corridas por arquitectura). No se aplican medidas correctivas. La prueba piloto de la Fase 4 verificará el rendimiento en condiciones reales de carga de datos.

**¿Requiere actualizar el documento?** Sí.

| Sección | Qué cambia |
|---|---|
| Tabla 2 | Sustituir la estimación teórica por los tiempos medidos y actualizar la nota metodológica |
| 3.7, Fase 5 | Reemplazar el párrafo del presupuesto por la formulación basada en medición empírica |
| 2.2.8 | Precisar que las cifras de parámetros citadas corresponden al clasificador original de mil clases |
| 3.3 | Registrar Python 3.11.13, AlmaLinux 9.8 y `opencv-python 5.0.0.93` |

---

## Entrada 004 — Cierre de la Fase 2

**Fecha:** 20/09/2026

**Productos generados:**

| Producto | Archivo | Estado |
|---|---|---|
| Entorno PyTorch + timm verificado | `fase2/.venv/` | Completado |
| Reporte de verificación del entorno | `fase2/salidas/reporte_entorno.json` | Completado |
| Caracterización del rendimiento | `fase2/salidas/reporte_benchmark.json` | Completado |
| Ficha técnica de configuración experimental | `fase2/ficha_tecnica.md` | Completado |
| Protocolo de criterios de calidad, etiquetado y selección | `fase2/criterios_depuracion.md` | Completado |
| Script de determinación empírica del umbral de desenfoque | `fase3/analizar_nitidez.py` | Completado |

**Productos esperados (según Tabla 4):**

- [x] Entorno PyTorch + timm configurado y verificado
- [x] Rendimiento del equipo caracterizado empíricamente
- [x] Protocolo experimental documentado en la ficha técnica
- [x] Criterios de calidad, etiquetado y selección formalizados

**Observaciones sobre el protocolo de criterios**

Se formalizaron los criterios declarados en las secciones 3.4, 3.6.1.2, 3.6.2.1 y 3.6.3.3 del plan en un documento operativo que establece el orden de aplicación de los nueve criterios de exclusión, los umbrales, las reglas de decisión ante error de etiquetado y las medidas de control del sesgo por repositorio. El principio rector adoptado es que los criterios se fijan antes de examinar los datos y no se modifican en función de los resultados obtenidos.

El umbral de desenfoque constituye la única excepción, por cuanto el plan declara su determinación empírica. Se desarrolló el script `analizar_nitidez.py`, que calcula la distribución de nitidez del conjunto bruto, simula el efecto de distintos umbrales candidatos sobre cada especie y extrae las treinta imágenes menos nítidas para inspección visual. El umbral se fijará al inicio de la Fase 3 y quedará registrado en la Entrada 006.

**Pendientes que pasan a la Fase 3:**

1. Descarga del conjunto bruto desde Kaggle, Mosquito Alert y Dryad
2. Ejecución de `analizar_nitidez.py` y determinación del umbral de desenfoque
3. Construcción del pipeline de depuración conforme al protocolo

# FASE 3 — Construcción del conjunto de datos

_(Entradas a partir de S7)_

## Entrada 005 — Recopilación y consolidación del conjunto bruto

**Fecha:** 20/09/2026
**Script:** `fase3/consolidar_conjunto.py`
**Salida guardada en:** `fase3/salidas/log_consolidacion.txt`, `fase3/reporte_consolidacion.json`, `fase3/procedencia_conjunto_bruto.csv`

**Repositorios consolidados:**

| Repositorio | Contenido utilizado | *Ae. aegypti* | *Ae. albopictus* | Total | Licencia |
|---|---|---|---|---|---|
| Kaggle (Isawasan) | `aegypti_0`, `aegypti_0_testing` | 603 | 0 | 603 | Abierta |
| Mosquito Alert (EBI S-BIAD249) | Imágenes ciudadanas | 73 | 14.778 | 14.851 | CC |
| Dryad (Nolte et al., 2024) | `DepictionDataset/body` | 398 | 398 | 796 | CC0 |
| Dryad (Couret) | Imágenes NEWORLEANS | 219 | 0 | 219 | CC0 |
| Mendeley (Ong y Ahmad, 2022) | Solo condición *landing* | 250 | 250 | 500 | CC BY 4.0 |
| **Total bruto** | | **1.543** | **15.426** | **16.969** | |

**Contenido excluido en la recopilación:**

| Repositorio | Excluido | Motivo |
|---|---|---|
| Kaggle | Carpetas `aegypti_90`, `_180`, `_270` | Rotaciones de las mismas imágenes; el aumento de datos se aplica después de la partición |
| Dryad (Nolte) | `DeviceDataset` | Las clases corresponden al dispositivo de captura, no a la especie; sin metadatos para mapearlas |
| Dryad (Nolte) | Imágenes de alas (`wing`) | El estudio se centra en la morfología del cuerpo completo |
| Dryad (Nolte) | Clases JA y KO | *Ae. japonicus* y *Ae. koreicus*, especies fuera del alcance |
| Ong y Ahmad | Condición *smashed* | Especímenes aplastados; contradice el criterio de exclusión por deterioro |
| Ong y Ahmad | *Culex quinquefasciatus* | Especie fuera del alcance |
| Mendeley (Pise et al.) | Todo el conjunto | Mayoritariamente aumentado; originales insuficientes y sin *Ae. albopictus* |

**Tabla cruzada especie × repositorio (medida M1):**

| Especie | Fuente dominante | Porcentaje | Resultado |
|---|---|---|---|
| *Ae. aegypti* | Kaggle | 39,1 % | Cumple |
| *Ae. albopictus* | Mosquito Alert | 95,8 % | No cumple |

La medida M1 se resolverá durante el balanceo mediante submuestreo dirigido de Mosquito Alert. Al igualar *Ae. albopictus* al tamaño de *Ae. aegypti*, las fuentes distintas de Mosquito Alert aportan 648 imágenes, por lo que la participación de esta última quedaría en torno al 58 %, por debajo del umbral del 70 %.

**Agrupamiento por espécimen:**

| Concepto | Valor |
|---|---|
| Imágenes con identificador de espécimen | 16.469 |
| Imágenes sin identificador (Ong) | 500, tratadas como especímenes únicos |
| Especímenes con varias imágenes | 548 |

Se identificaron especímenes con múltiples tomas en Kaggle (sufijos a–d sobre un mismo identificador) y en Dryad (dos tomas por espécimen). Se detectaron además tres archivos duplicados en Kaggle (`- Copy`), agrupados con su original para su eliminación en la etapa de depuración.

**Decisión técnica — enlaces duros:** el repositorio original ocupa 29 GB y el disco disponía de 21 GB libres. Se construyó el conjunto bruto mediante enlaces duros en lugar de copias físicas, lo que no consume espacio adicional y preserva los archivos originales. Como consecuencia, las imágenes del conjunto bruto no deben modificarse en su lugar.

**Observaciones:**

El tamaño bruto de *Ae. aegypti* (1.543 imágenes) es inferior al previsto en la sección 3.4 del plan (aproximadamente 2.700 por especie tras la depuración). La cifra definitiva se actualizará en el documento una vez concluida la depuración. Si tras aplicar los criterios de exclusión *Ae. aegypti* quedara por debajo del umbral mínimo de 1.000 imágenes, se recurrirá al repositorio IEEE DataPort (Pise et al., ~900 imágenes de *Ae. aegypti*) conforme a lo previsto en el protocolo.

---

## Entrada 006 — Determinación del umbral de desenfoque y depuración

**Fecha:** 20/09/2026 (partes A a C) y 24/09/2026 (partes D y E) — corregida en la Entrada 006 bis
**Script:** `fase3/analizar_nitidez.py` (versión 2)
**Salida guardada en:** `fase3/salidas/log_nitidez.txt`, `fase3/analisis_nitidez/metricas_por_imagen.csv`, `fase3/analisis_nitidez/reporte_nitidez.json`

### Parte A — Análisis de nitidez

**Parámetros:** 16.969 imágenes, normalización a 512 px de lado mayor, 11 procesos. **Duración:** 6,6 min.

| Concepto | Valor |
|---|---|
| Imágenes procesadas | 16.921 |
| Ilegibles (E1) | 48, todas de *Ae. albopictus* de Mosquito Alert |

**Distribución por repositorio (varianza del Laplaciano):**

| Repositorio | n | Mínimo | p5 | p25 | Mediana | Máximo |
|---|---|---|---|---|---|---|
| Couret | 219 | 18,3 | 33,0 | 51,8 | 67,9 | 121,9 |
| Dryad | 796 | 35,6 | 69,7 | 113,8 | 231,3 | 1.049,2 |
| Kaggle | 603 | 11,9 | 37,3 | 45,4 | 52,2 | 103,2 |
| Mosquito Alert | 14.803 | 2,4 | 22,8 | 51,6 | 98,5 | 15.889,5 |
| Ong | 500 | 69,5 | 116,9 | 163,8 | 212,2 | 555,8 |

**Simulación del descarte por repositorio:**

| Umbral | Couret | Dryad | Kaggle | Mosquito Alert | Ong |
|---|---|---|---|---|---|
| 10 | 0,0 % | 0,0 % | 0,0 % | 0,4 % | 0,0 % |
| 20 | 0,9 % | 0,0 % | 0,7 % | 3,5 % | 0,0 % |
| 30 | 3,7 % | 0,0 % | 1,2 % | 9,5 % | 0,0 % |
| 50 | 22,4 % | 0,5 % | **42,1 %** | 23,7 % | 0,0 % |
| 75 | **61,6 %** | 6,5 % | **98,3 %** | **39,6 %** | 0,8 % |

**Techo de contención:** 30. A partir de 50, Kaggle supera el 30 % de descarte. Por especie, un umbral de 50 habría parecido aceptable (20,4 % y 22,8 %), lo que confirma la necesidad del desglose por repositorio.

### Parte B — Inspección visual

Se examinaron dos muestras:

1. **Ocho imágenes de valores bajos** (2,43 a 30,92) de Mosquito Alert, Kaggle y Couret.
2. **Doce imágenes aleatorias de la franja 5–10**, que el umbral candidato eliminaría.

| Varianza | Fuente | Observación |
|---|---|---|
| 2,43 | Mosquito Alert | Archivo truncado, casi completamente gris |
| 4,40 | Mosquito Alert | Desenfocada; tórax no legible |
| 11,86 | Kaggle | Utilizable con limitaciones |
| 13,15 | Mosquito Alert | Tórax no legible por el tamaño reducido del espécimen, no por desenfoque |
| 18,32 | Couret | Posible error de etiqueta (ver hallazgos) |
| 26,82 | Kaggle | Nítida; abdomen con bandas y patas anilladas visibles. Varianza baja por fondo uniforme |

**Franja 5–10 (12 imágenes):** 4 degradadas por desenfoque, 4 con espécimen demasiado pequeño, 3 aceptables con limitaciones y 1 claramente útil (9,17; franja blanca central del escudo torácico visible).

### Parte C — Decisión

**Umbral de desenfoque adoptado: 10**

**Justificación:** la franja 5–10 es mayoritariamente no apta para morfología fina, aunque incluye alguna imagen útil. El umbral de 10 no elimina ninguna imagen de *Ae. aegypti* (mínimo 11,9), la clase limitante, y solo afecta al 0,4 % de Mosquito Alert, fuente con un amplio excedente de *Ae. albopictus*.

### Hallazgos

1. **La métrica no mide solo desenfoque.** La varianza del Laplaciano también disminuye con fondos uniformes y con especímenes pequeños en el encuadre. Se emplea únicamente como filtro de degradación severa.
2. **Archivos truncados.** Los JPEG incompletos se decodifican parcialmente rellenando con gris y superan una verificación permisiva. El criterio E1 se endurece: se aplicará sin tolerancia a truncamiento.
3. **Posible error de etiquetado en Couret.** La imagen de varianza 18,32, etiquetada como *Ae. aegypti*, no muestra sus rasgos diagnósticos. Couret es un repositorio de vectores de malaria; se revisará íntegramente en la verificación morfológica.
4. **Riesgo de atajo por tamaño del espécimen.** Las fuentes de *Ae. aegypti* son mayoritariamente primeros planos; Mosquito Alert contiene muchas fotografías con el espécimen diminuto, casi todas de *Ae. albopictus*. Se vigilará mediante métricas por repositorio (M4) y Grad-CAM.

### Parte D — Depuración del conjunto

**Fecha:** 24/09/2026, de 21:30 (primera simulación) a 22:12 (ejecución definitiva)
**Script:** `fase3/depurar_conjunto.py`
**Salida guardada en:** `fase3/salidas/log_depuracion.txt`, `fase3/depuracion/registro_imagenes_eliminadas.csv`, `fase3/depuracion/metricas_depuracion.csv`, `fase3/depuracion/reporte_depuracion.json`
**Duración:** 9,0 min sobre 16.969 imágenes con 11 procesos

**Imágenes eliminadas por criterio:**

| Criterio | *Ae. aegypti* | *Ae. albopictus* |
|---|---|---|
| E1 — Corrupto, ilegible o truncado | 0 | 50 |
| E2 — Resolución insuficiente | 0 | 0 |
| E3 — Duplicado exacto (MD5) | 4 | 496 |
| E4 — Duplicado perceptual (pHash 256 bits + correlación) | 8 | 133 |
| E5 — Exposición deficiente (rango dinámico < 40) | 2 | 189 |
| E6 — Desenfoque (varianza < 10) | 0 | 50 |
| E7 — Estadio no adulto | 0 | 0 |
| E8 — Etiqueta ambigua o ausente | 0 | 0 |
| E9 — Licencia incompatible | 0 | 0 |
| **Total eliminadas** | **14** | **918** |
| **Total aceptadas** | **1.529** | **14.508** |
| **Porcentaje aceptado** | **99,1 %** | **94,0 %** |

E7 y E9 quedan controlados en la recopilación: se seleccionaron conjuntos de ejemplares adultos y se verificaron las licencias por repositorio, de modo que ninguna imagen llega a la depuración con esos defectos.

**Duplicados detectados:**

| Concepto | Valor |
|---|---|
| Candidatos propuestos por el pHash | 895 |
| Confirmados por correlación o MD5 | 758 |
| Rechazados por la verificación | 137 |
| Componentes de duplicados | 552 |
| Conflictos entre especies (E8) | 0 |

**Conjunto depurado, especie × repositorio:**

| Repositorio | *Ae. aegypti* | *Ae. albopictus* |
|---|---|---|
| Couret | 219 (14,3 %) | 0 |
| Dryad | 398 (26,0 %) | 398 (2,7 %) |
| Kaggle | 600 (39,2 %) | 0 |
| Mosquito Alert | 68 (4,4 %) | 13.862 (95,5 %) |
| Ong | 244 (16,0 %) | 248 (1,7 %) |
| **Total** | **1.529** | **14.508** |

**Umbral mínimo de muestra:** ambas especies superan las 1.000 imágenes. Tamaño de balanceo previsto: **1.529 por especie**.

**Proyección de la medida M1 tras el balanceo:** al igualar *Ae. albopictus* a 1.529, las fuentes distintas de Mosquito Alert aportan 646 imágenes (398 de Dryad y 248 de Ong), por lo que Mosquito Alert contribuiría con 883, un 57,7 %, por debajo del umbral del 70 %.

**Escritura:** 16.037 enlaces duros, ninguna copia física.

### Parte E — Correcciones aplicadas durante la depuración

La primera ejecución produjo resultados inconsistentes que obligaron a reoperacionalizar dos criterios. El proceso queda documentado por su valor metodológico.

**Primera ejecución.** E8 eliminó 2.385 imágenes por supuesto duplicado entre especies y E4 otras 353. *Ae. aegypti* quedó en 932, por debajo del umbral mínimo. La inspección de los casos señalados mostró que eran especímenes distintos: el par 3 confrontaba una hembra con un macho y el par 4 dos hembras en poses diferentes, todas del mismo montaje de laboratorio.

**Diagnóstico.** El pHash de 64 bits con distancia ≤ 6 no distingue dos ejemplares distintos fotografiados con el mismo fondo, encuadre e iluminación. Además, al agrupar por componentes conectados, cadenas de falsos positivos arrastraban decenas de imágenes: el par 3 estaba a distancia 10, fuera del umbral, y entró por encadenamiento.

| Caso | pHash 64 bits | pHash 256 bits | Correlación |
|---|---|---|---|
| Duplicado real | 0 | 0 | 1,000 |
| Dos hembras distintas | 4 | 48 | 0,900 |
| Hembra frente a macho | 10 | 118 | 0,602 |

**Corrección de E4.** Se adoptó pHash de 256 bits para proponer candidatos y una verificación por correlación de Pearson ≥ 0,98 sobre miniaturas de 64 × 64. Solo se unen los pares verificados, con lo que desaparece el encadenamiento. E8 se restringió a duplicados exactos por MD5 entre especies.

**Segunda ejecución.** *Ae. aegypti* subió a 1.435 y los conflictos entre especies bajaron a cero, pero E5 eliminó 96 imágenes de esa especie, 93 de ellas de Dryad. La inspección de seis casos mostró fotografías de microscopio correctamente expuestas: el criterio descartaba por el fondo blanco, que ocupaba entre el 26 % y el 49 % de píxeles casi saturados.

| Tipo de imagen | Brillo medio | Rango dinámico (p99 − p1) |
|---|---|---|
| Laboratorio, fondo blanco | 235–239 | 184–201 |
| Sobreexposición real | 247 | 14 |
| Subexposición real | 7 | 14 |

**Corrección de E5.** Se sustituyó el brillo medio por el rango dinámico, con umbral de 40. Una imagen realmente mal expuesta pierde el rango tonal; una fotografía de laboratorio lo conserva porque el espécimen sigue siendo oscuro.

**Tercera ejecución.** Resultado definitivo: 1.529 imágenes de *Ae. aegypti* (99,1 % de aceptación) y Dryad íntegro con sus 398 por especie.

**Hallazgo metodológico.** Tres criterios fallaron por la misma causa: medían el fondo en lugar del espécimen.

| Criterio | Métrica inicial | Qué medía en realidad |
|---|---|---|
| E6 | Varianza del Laplaciano global | Uniformidad del fondo y tamaño del espécimen |
| E4 | pHash de 64 bits | Composición general de la escena |
| E5 | Brillo medio | Proporción de fondo claro |

El conjunto reúne fotografías de microscopio con fondo controlado y fotografías ciudadanas con fondo natural, de modo que cualquier métrica global queda dominada por esa diferencia. Los criterios de calidad de imagen de uso habitual no son directamente transferibles a conjuntos heterogéneos de vectores y requieren verificación sobre las propias fuentes antes de aplicarse. El hallazgo se considera aportable al documento final.

**Corrección del hallazgo sobre Couret.** La sospecha de error de etiquetado registrada en la Entrada 006 quedó descartada. Los archivos corresponden a la cepa *New Orleans*, colonia de laboratorio de *Aedes aegypti* establecida en 1970 y empleada como referencia susceptible a insecticidas. La tonalidad parda observada es propia de ejemplares de colonia secos, que pierden escamas. Se mantiene como observación de calidad que en ellos los rasgos diagnósticos se aprecian con dificultad.

---

## Entrada 006 bis — Auditoría de la estructura del proyecto

**Fecha:** 27/09/2026
**Script:** `fase3/auditar_estructura.py`
**Salida guardada en:** `auditoria_estructura.txt`, `estructura_proyecto.txt` (regenerado)
**Motivo:** el tutor detectó que `estructura_proyecto.txt` no coincidía con el contenido real de la carpeta `fase3/`.

**Discrepancia detectada.** El árbol entregado no mostraba las carpetas `fase3/conjunto_bruto/` y `fase3/conjunto_depurado/`, declaradas en el inventario y empleadas por los tres scripts de la fase. Listaba además siete archivos que no figuraban en el inventario. La auditoría contrastó cada ruta del inventario con el disco.

**Resultados:**

| Verificación | Resultado |
|---|---|
| Rutas del inventario presentes en disco | 38 de 38 |
| `conjunto_bruto/` | 16.969 archivos (1.543 *Ae. aegypti*, 15.426 *Ae. albopictus*); 16.969 enlaces duros |
| `conjunto_depurado/` | 16.037 archivos (1.529 *Ae. aegypti*, 14.508 *Ae. albopictus*); 16.037 enlaces duros |
| Coincidencia con las Entradas 005 y 006 | Exacta en ambos conjuntos y en ambas especies |
| Archivos en disco no inventariados | 7 |
| Pares completos en `verificacion_dup/` | 2 de 4 |
| Imágenes en `muestra_visual/` | 73 (15 por repositorio; 13 en Couret) |

Los conjuntos de datos existen, están completos y no ocupan espacio adicional. La discrepancia se limitaba al árbol, que se regeneró mostrando cada carpeta de más de 50 archivos colapsada con su conteo.

**Archivos no inventariados y decisión:**

| Archivo | Decisión |
|---|---|
| `entrega de las fases 2 y 3.zip` | Se retira de la carpeta del proyecto: un paquete de entrega no forma parte de su contenido |
| `estructura_proyecto.txt` | Se inventaría |
| `fase3/auditar_estructura.py` | Se inventaría |
| `franja_5_10.zip`, `revision.zip`, `verificacion_dup.zip`, `verificacion_e5.zip` | Se inventarían como copias comprimidas de las carpetas homónimas |

**Muestra visual de Couret.** Contiene 13 imágenes en lugar de 15 porque no incluye el percentil 1. Con 219 imágenes, el p01 cae dentro de las cinco de menor varianza, ya seleccionadas como mínimas, y el script descarta las repetidas. No es un error.

**Pares 1 y 2 de `verificacion_dup/`.** Corresponden a conflictos entre especies (E8) de la primera ejecución y solo conservan la imagen eliminada. Sin la imagen con la que se emparejaron, esa decisión no puede verificarse de forma independiente. Los pares 3 y 4, que sustentan la comparación de la Parte E de la Entrada 006 (dos hembras distintas; hembra frente a macho), están completos. Se registra como incidencia abierta.

**Reconstrucción de fechas.** Las fechas en blanco del inventario, de los registros de incidencias y de los cambios al documento se completaron a partir de la fecha de modificación de cada archivo. La secuencia es coherente con lo narrado en las Entradas 005 y 006:

| Momento | Actividad |
|---|---|
| 20/09, 22:07–22:11 | Simulación y consolidación del conjunto bruto |
| 20/09, 22:22–22:39 | `analizar_nitidez.py` v2, análisis e inspección visual |
| 24/09, 21:30 | Primera simulación de depuración |
| 24/09, 21:33 | Inspección de pares duplicados (`verificacion_dup/`) |
| 24/09, 21:47 | Segunda simulación |
| 24/09, 21:49 | Inspección de exposición (`verificacion_e5/`) |
| 24/09, 22:02 | Tercera simulación |
| 24/09, 22:12 | Ejecución definitiva y escritura del conjunto depurado |

**Desviaciones de las reglas de registro detectadas:**

1. La Entrada 006 se fechó el 24/09 aunque las partes A a C se ejecutaron el 20/09. Se incumplió la regla 1 (anotar el mismo día). Fecha corregida.
2. `analizar_nitidez.py` v1 y `criterios_depuracion.md` v2 se sobrescribieron en la misma ruta y no se conservan. La fecha exacta de la v2 del protocolo no es recuperable.
3. `fase2/requirements.txt` y `fase2/ficha_tecnica.md` se modificaron el 20/09 (09:03 y 09:08) sin registro. Se añaden al inventario.
4. El encabezado de `consolidar_conjunto.py` describe la operación como copia, aunque el script crea enlaces duros por defecto. El comportamiento queda confirmado por la auditoría.

**Medida adoptada:** a partir de esta entrada, versiones sucesivas de un mismo script o protocolo se conservarán con sufijo (`_v1`, `_v2`), y `estructura_proyecto.txt` se regenerará con `auditar_estructura.py --regenerar` antes de cada entrega.

---

## Entrada 006 ter — Repositorio de la bibliografía

**Fecha:** 01/10/2026
**Script:** `descargar_bibliografia.py`
**Salida guardada en:** `bibliografia/` (`indice_bibliografia.md`, `verificacion_doi.md` y los PDF), `arbol_bibliografia.txt`
**Motivo:** el tutor solicitó acceso a los documentos citados en la bibliografía del plan.

**Procedimiento.** El script contiene las 48 referencias del plan y realiza tres tareas:

1. Verifica cada DOI en Crossref y compara el título registrado con el citado.
2. Descarga los PDF de acceso abierto, mediante Unpaywall a partir del DOI o desde enlaces directos (arXiv, PMLR, CVF).
3. Nombra cada archivo con el número de cita y el título (`06_Potentials and limitations....pdf`) y genera un índice con el estado de cada referencia.

Los documentos de suscripción o que el sitio impedía descargar automáticamente se obtuvieron de forma manual, y la normativa ecuatoriana desde los portales oficiales.

**Resultados:**

| Situación | Referencias | Cantidad |
|---|---|---|
| DOI verificados en Crossref | Todas las que tienen DOI | 27 de 27 correctos |
| Con PDF en la carpeta | El resto | 39 |
| Sitios web citados, sin PDF | [11]–[14] | 4 |
| Normas ISO, de pago y no redistribuibles | [44]–[47] | 4 |
| No disponible | [16] Lapo-Talledo (2024): revista de suscripción; la biblioteca de la PUCE no tiene acceso | 1 |

**Revisión del contenido de los PDF.** Se extrajo la primera página de cada archivo para comprobar que corresponde al documento citado. Tres archivos no son el documento completo o no son el citado:

| Ref. | Problema |
|---|---|
| [09] | Contiene solo el resumen (1 página), no el artículo |
| [38] | Es un extracto de 3 páginas de la Constitución, no el texto completo |
| [48] | Es el documento CD63/7 de la OPS (estrategia 2026–2035, julio de 2026), no el documento de 2016 citado |

**Reemplazo.** El mismo día se sustituyeron dos de los tres archivos:

- [38]: texto completo actualizado de la Constitución, 163 páginas, con la última reforma del Registro Oficial 568 (30-V-2024), obtenido de la biblioteca digital de la Defensoría Pública.
- [48]: documento CD55/16, *Strategy for Arboviral Disease Prevention and Control*, aprobado en el 55.º Consejo Directivo de la OPS (septiembre de 2016). Se decide citar este documento, porque es el de 2016, y corregir el título en el plan.
- [09]: el texto completo es de suscripción. Se conserva el resumen, con una nota en el índice.

**Errores detectados en las referencias del plan:**

| Ref. | Error | Corrección |
|---|---|---|
| [3] | Sin DOI | Agregar doi: 10.1097/EDE.0000000000001874 |
| [5] | Número de la revista | Vol. 100, n.º 6 (no n.º 7) |
| [20] | Nombre de la conferencia | 2020 International Research Conference on Smart Computing and Systems Engineering (SCSE), no ICITR |
| [38] | Autor | Asamblea Constituyente, no Asamblea Nacional |
| [39] | Autor | Congreso Nacional; la Asamblea Nacional no existía en 2006 |
| [48] | Título | Se cita la *Strategy for Arboviral Disease Prevention and Control* (documento CD55/16, 2016). La *Integrated Management Strategy* es un documento posterior |

En [19] Qian y [25] García-Aguirre, Crossref registra 2024 y el plan cita 2025. Corresponde a la diferencia entre la publicación en línea y el volumen impreso; se mantiene el año del volumen.

---

## Entrada 006 quater — Agrupación por espécimen antes de la partición

**Fecha:** 23/10/2026
**Script:** `fase3/agrupar_especimenes.py` (versión 4)
**Salida guardada en:** `fase3/agrupacion/`
**Motivo:** observación del tutor (14/10/2026): agrupar las fotografías del mismo mosquito antes de dividir el conjunto, en Mosquito Alert (incidencia del 27/09/2026) y en Ong, que no tiene identificador de espécimen. Si dos fotos del mismo ejemplar quedan una en entrenamiento y otra en prueba, las métricas se inflan.

**Fuente del grupo según repositorio:**

| Repositorio | Fuente del grupo | Imágenes | Grupos |
|---|---|---|---|
| Kaggle | Número de imagen sin sufijo (procedencia) | 600 | 150 |
| Dryad | Código de espécimen (procedencia) | 796 | 398 |
| Couret | Una imagen por individuo (procedencia) | 219 | 219 |
| Mosquito Alert | Identificador del reporte ciudadano | 13.930 | 10.386 |
| Ong | Similitud del recorte del mosquito y revisión visual | 492 | 267 |
| **Total** | | **16.037** | **11.420** |

### Parte A — Mosquito Alert

La carpeta descargada de Mosquito Alert contiene solo las fotografías, nombradas por su UUID, sin metadatos. La relación entre foto y reporte se obtuvo del archivo público `mosquito_alert_photos_clean.csv` del servidor de Mosquito Alert (columnas `uuid` y `report_id`), descargado el 23/10/2026 y guardado en `fase3/agrupacion/`.

| Resultado | Valor |
|---|---|
| Fotografías con reporte identificado | 10.091 de 13.930 (72,4 %) |
| Reportes con varias fotografías | 2.132 (5.676 imágenes; el mayor, 21) |
| Reportes con fotografías de ambas especies | 0 |
| Fotografías sin reporte en el archivo público | 3.839 (15 *Ae. aegypti*, 3.824 *Ae. albopictus*) |

El 56 % de las fotografías con reporte pertenece a reportes con varias fotos, lo que confirma que la incidencia del 27/09 era real.

**Intento de recuperar las fotografías sin reporte.** Se ensayó unir cada foto sin reporte a su vecina más parecida (correlación de miniaturas de 64 × 64 px), con un umbral calibrado sobre las 7.615 fotos con reporte conocido: el umbral elegido debía lograr que la vecina perteneciera al mismo reporte en al menos el 95 % de los casos. Ningún umbral lo consiguió; la precisión máxima fue 0,61 con r ≥ 0,97 (0,74 considerando solo reportes con varias fotos) y 0,19 con r ≥ 0,85. Las fotos de un mismo reporte no se parecen entre sí más que las de reportes distintos, de modo que la unión por similitud **no se aplicó**.

**Decisión:** en Mosquito Alert solo entran al conjunto de trabajo las fotografías cuyo reporte es verificable (columna `reporte_verificable` de `especimenes.csv`). En *Ae. albopictus* no tiene costo, porque el balanceo toma unas 883 fotos de Mosquito Alert y hay 10.038 con reporte. En *Ae. aegypti* excluye 15 imágenes: la especie pasa de 1.529 a 1.514, por encima del mínimo de 1.000. La regla es la misma para ambas especies.

### Parte B — Ong

Ong no ofrece identificador de espécimen; las fotos están numeradas de 0 a 249 por especie. Se ensayaron cuatro formulaciones, cada una motivada por la inspección visual de la anterior:

| Versión | Criterio | Problema observado |
|---|---|---|
| 1 | Correlación de la foto completa ≥ 0,85 | El fondo de piel unía mosquitos distintos por encadenamiento (grupo de 56 fotos en *albopictus*) y separaba el mismo mosquito con r = 0,848 |
| 2 | r ≥ 0,80 y numeración a ≤ 10 de distancia | Persistía un grupo de 34 mosquitos distintos en *albopictus*: la piel clara uniforme domina la foto, y fotos consecutivas suelen ser de mosquitos distintos. Quedaba separado el mismo mosquito fotografiado en otra sesión (107/108 con 184, r = 0,987) |
| 3 | Correlación de un recorte centrado en el mosquito, con la regla anterior o r ≥ 0,92 a cualquier distancia | Se resolvieron los dos casos, pero los rangos se solapan: mismo mosquito con r = 0,77–0,88 y mosquitos distintos con r = 0,90 cuando el recorte falla. Ningún umbral los separa |
| 4 | Versión 3 más uniones manuales verificadas visualmente | — |

Es el mismo problema de los criterios E4 y E5 (Entrada 006): una métrica sobre la imagen completa mide el fondo y no el espécimen.

**Revisión visual (versión 4).** El script genera, para cada especie, los 40 pares más parecidos que quedaron en grupos distintos (`ong_casi/`) y hojas de contacto con todas las fotos en orden y su grupo (`ong_contacto/`). Cada par identificado como el mismo mosquito se anotó en `ong_uniones_manuales.csv` con su motivo. Se hicieron dos rondas:

| Ronda | Pares revisados | Hojas de contacto | Uniones añadidas |
|---|---|---|---|
| 1 | 80 | 14 | 137 |
| 2 | 80 | 14 | 12 |
| **Total** | | | **149** (76 *aegypti*, 73 *albopictus*) |

En la segunda ronda no aparecieron pares claros del mismo mosquito; la agrupación se dio por cerrada. De las 149 uniones, 37 están marcadas `DUDOSO`: misma escena y postura parecida sin certeza de que sea el mismo ejemplar. Se unieron por precaución, porque agrupar de más solo reduce la granularidad de la partición, mientras que agrupar de menos produce fuga de datos.

| Resultado en Ong | *Ae. aegypti* | *Ae. albopictus* |
|---|---|---|
| Imágenes | 244 | 248 |
| Especímenes resultantes | 109 | 158 |
| Grupos con varias fotos | 50 | 33 |
| Mayor grupo | 13 | 11 |
| Uniones manuales aplicadas | 76 | 73 |

En las hojas de contacto, las fotos 0–72 de *albopictus* no presentan repeticiones: son mosquitos distintos fotografiados uno tras otro. Por eso la numeración por sí sola no sirve como criterio.

**Limitación:** la agrupación de Ong depende de un juicio visual. Queda documentada fila por fila en `ong_uniones_manuales.csv` y puede verificarse con las láminas conservadas.

### Parte C — Implicaciones para la Entrada 008

1. La partición se hará por grupo (`especimenes.csv`), estratificada por especie y repositorio.
2. En Mosquito Alert solo se usarán las fotos con `reporte_verificable = si`.
3. Siguiendo la observación del tutor, se reservará como subconjunto de control la parte de prueba procedente de Dryad y Ong, las únicas fuentes con ambas especies fotografiadas con el mismo montaje. Las métricas se informarán en tres niveles: global, por repositorio y en el subconjunto de control. El criterio para concluir que un modelo aprende el contexto de captura se fijará antes del entrenamiento.

**Desviación de la regla de registro:** las versiones 1 a 3 de `agrupar_especimenes.py` se sobrescribieron en la misma ruta, contra la medida adoptada en la Entrada 006 bis. Los cambios de cada versión quedan descritos en el encabezado del script y en esta entrada.

---

## Entrada 006 quinquies — Ampliación con fotografías de celular y composición del conjunto de trabajo

**Fecha:** 23–24/10/2026
**Scripts:** `fase3/descargar_inaturalist.py`, `fase3/consolidar_conjunto.py` (versión 2), `fase3/depurar_conjunto.py` (sin cambios), `fase3/agrupar_especimenes.py` (versión 4)
**Salida guardada en:** `fase3/inaturalist/`, `fase3/conjunto_bruto_v2/`, `fase3/conjunto_depurado_v2/`, `fase3/depuracion_v2/`, `fase3/salidas/log_consolidacion_v2.txt`, `fase3/salidas/log_depuracion_v2.txt`, `fase3/agrupacion/`
**Motivo:** observación del tutor (23/10/2026). Casi todas las imágenes de *Ae. albopictus* son fotografías de celular (Mosquito Alert) y las de *Ae. aegypti*, mayoritariamente de laboratorio. El modelo podría aprender a reconocer el tipo de fotografía y no el mosquito. El tutor indicó incorporar fotografías de celular de *Ae. aegypti* desde iNaturalist, excluyendo Ecuador, antes de dividir el conjunto.

### Parte A — iNaturalist

Criterios fijados antes de descargar:

| Criterio | Valor |
|---|---|
| Taxón | *Aedes aegypti*, con sus subespecies |
| Calidad | Grado de investigación (identificación confirmada por la comunidad) |
| Licencia de la foto (E9) | CC0, CC BY o CC BY-NC |
| País | Fuera de Ecuador; se excluyen también las observaciones sin ubicación, cuyo país no es verificable |
| Fotos por observación | Una (la primera con licencia válida): cada imagen es un espécimen distinto |
| Máximo por observador | 10, para que ningún fotógrafo ni su montaje domine la fuente |
| Selección | Aleatoria entre las elegibles, semilla 42 |

| Resultado | Valor |
|---|---|
| Observaciones con foto CC en grado de investigación | 4.084 |
| Excluidas por estar en Ecuador | 27 |
| Excluidas por no tener ubicación | 6 |
| Seleccionadas | 1.100, de 790 observadores |
| Descargadas | 1.100 (23/10/2026) |
| Aceptadas tras la depuración | 1.093 (7 excluidas por E5) |
| Duplicados con otras fuentes | 0 |

Las 27 observaciones ecuatorianas no entran al conjunto de trabajo: se reservan para la validación contextual. La procedencia de cada foto (observación, autor, licencia, lugar, coordenadas y URL) consta en `fase3/inaturalist/procedencia_inaturalist.csv`.

### Parte B — Corrección: fotografías de celular de Dryad

Al revisar la estructura del repositorio de Nolte et al. se detectó un error en la Entrada 005. `DeviceDataset` se excluyó completo con el argumento de que "sus clases corresponden al dispositivo de captura y no a la especie". En realidad, `DeviceDataset/body` contiene carpetas por especie (AE, AL) y fotografías tomadas con celular (`Phone_`) de los mismos especímenes de `DepictionDataset`, identificables por el código del nombre (`Phone_AEa04_…` corresponde a `Macro_AEa04_…` y `Micro_AEa04_…`). Se incorporaron solo las `Phone_`; las `Macro_` y `Micro_` de `DeviceDataset` repiten las de `DepictionDataset`.

| | *Ae. aegypti* | *Ae. albopictus* |
|---|---|---|
| Fotografías `Phone_` incorporadas | 199 | 199 |
| Aceptadas tras la depuración | 199 | 198 (1 excluida por E6) |
| Dryad total (Macro, Micro y Phone) | 597 | 596 |

Se agrupan con su espécimen: Dryad queda con 199 grupos por especie, de 3 fotografías cada uno.

**Pendiente:** describir el fondo de las fotografías `Phone_` (controlado o natural).

### Parte C — Reconsolidación y depuración

Se reconsolidaron los seis repositorios en carpetas nuevas con sufijo `_v2`. Los archivos de la versión 1 se conservan como evidencia de las Entradas 005 y 006. La depuración se aplicó con los mismos criterios y umbrales, sin modificaciones.

| | *Ae. aegypti* | *Ae. albopictus* |
|---|---|---|
| Conjunto bruto v2 | 2.842 | 15.625 |
| Eliminadas (E1 a E9) | 21 | 919 |
| Conjunto depurado v2 | 2.821 | 14.706 |

**Reproducibilidad:** para las imágenes de la versión 1 el resultado es idéntico al de la Entrada 006 (mismas 932 eliminadas, 758 duplicados confirmados, 0 conflictos entre especies). Las diferencias corresponden solo a las imágenes nuevas.

**Agrupación v2:** 17.527 imágenes en 12.513 grupos. Mosquito Alert y Ong conservan los grupos de la Entrada 006 quater; las 149 uniones manuales de Ong se aplicaron sin cambios.

### Parte D — Composición del conjunto de trabajo (decisión)

Clasificando cada especie por tipo de fotografía:

| Tipo | *Ae. aegypti* | *Ae. albopictus* |
|---|---|---|
| Mismo montaje para ambas especies (Dryad + Ong) | 841 | 844 |
| Celular de campo (iNaturalist; Mosquito Alert con reporte verificable) | 1.146 (1.093 + 53) | 10.038 |
| Laboratorio, solo *Ae. aegypti* (Kaggle + Couret) | 819 | — |

Con el mismo tamaño por especie, igualar el número de fotografías de campo obliga a igualar también la parte de laboratorio, y *Ae. albopictus* no dispone de más laboratorio que Dryad y Ong. Se evaluaron dos opciones:

| Opción | Composición | Por especie | Fotos de campo |
|---|---|---|---|
| **A (adoptada)** | Dryad + Ong + celular de campo | ≈ 1.987 | 1.146 frente a 1.143 |
| B | Además, Kaggle y Couret en *Ae. aegypti* | ≈ 2.806 | 41 % frente a 70 % |

**Decisión:** opción A. Es la única que iguala la composición por tipo de fotografía entre especies. Además elimina dos fuentes que solo contienen *Ae. aegypti* (Kaggle y Couret), que por sí mismas también delataban la especie.

| Conjunto de trabajo | *Ae. aegypti* | *Ae. albopictus* |
|---|---|---|
| Dryad | 597 | 596 |
| Ong | 244 | 248 |
| iNaturalist | 1.093 | — |
| Mosquito Alert (solo reporte verificable) | 53 | 1.143 (submuestreo de 10.038) |
| **Total** | **1.987** | **1.987** |

Kaggle (600) y Couret (219) quedan fuera del entrenamiento. Se conservan depuradas, agrupadas y documentadas, y pueden usarse como prueba adicional descriptiva. La fuente dominante en *Ae. albopictus* pasa a ser Mosquito Alert con el 57,5 % y en *Ae. aegypti* iNaturalist con el 55,0 %: la medida M1 (ninguna fuente sobre el 70 %) se cumple en ambas especies.

### Desviaciones respecto al plan

| Aspecto | Plan | Ejecución | Justificación |
|---|---|---|---|
| Repositorios de ampliación | "distintos de iNaturalist" (3.4) | iNaturalist fuera de Ecuador | Indicación del tutor. Se mantiene el propósito del plan: las observaciones ecuatorianas siguen reservadas para la validación contextual |
| Kaggle | Fuente declarada (3.4) | Fuera del conjunto de trabajo | Solo contiene *Ae. aegypti* de laboratorio; impide igualar la composición por tipo de fotografía |
| Tamaño por especie | ≈ 2.700 | 1.987 | Composición igualada (opción A). Supera el mínimo de 1.000 |
| DeviceDataset de Dryad | Excluido (Entrada 005) | Fotos `Phone_` incluidas | Corrección de un error de la Entrada 005 |

### Siguientes pasos, antes de la partición

1. Verificación morfológica de etiquetas (Entrada 007) sobre el conjunto de trabajo de la opción A, incluidas iNaturalist y las fotografías de celular de Dryad.
2. Registrar el protocolo de evaluación: métricas por repositorio, subconjunto de control (Dryad + Ong) y procedimiento de Grad-CAM, con sus criterios de decisión fijados antes de entrenar. *Hecho: Entrada 007 bis.*
3. Partición por grupo (Entrada 008).

---

## Entrada 007 — Verificación morfológica de etiquetas

**Fecha:** 2026-10-23 (preparación) a 2026-10-24 (revisión y evaluación)

**Script:** `fase3/verificacion_morfologica.py` (v3) y `fase3/corregir_verificacion.py`

### Parte A — Muestra y procedimiento

Se fijó el conjunto de trabajo de la opción A en `fase3/conjunto_trabajo.csv` (1.987 imágenes por especie; Mosquito Alert *Ae. albopictus* submuestreado por reportes completos, semilla 42). De él se extrajo una muestra de 200 imágenes por especie (10,1 %), estratificada por repositorio con reparto por resto mayor:

| Repositorio | *Ae. aegypti* | *Ae. albopictus* |
|---|---|---|
| Dryad | 60 | 60 |
| Ong | 25 | 25 |
| iNaturalist | 110 | — |
| Mosquito Alert | 5 | 115 |
| **Total** | **200** | **200** |

Las 400 imágenes se mezclaron y numeraron. La etiqueta quedó en `clave_verificacion.csv`, que no se abrió hasta la evaluación (revisión a ciegas). Categorías: `aegypti` (escudo con lira), `albopictus` (una franja blanca central), `otro` (no es *Ae. aegypti* ni *Ae. albopictus*; cuenta como discrepancia) e `indeterminable` (el escudo no se ve; no cuenta).

**Ejecución de la revisión.**

1. Las láminas con recorte automático (v2) se descartaron: el recorte encuadraba a menudo patas, fondo o el abdomen en lugar del tórax. Se exportó la muestra completa a 1.600 px, renombrada solo con su número (`--exportar-muestra`).
2. **Propuesta a ciegas con asistencia de IA:** un asistente de IA (Claude, Anthropic) examinó las 400 imágenes completas, con acercamiento al tórax cuando hizo falta, y propuso para cada una especie, rasgo observado, nota y grado de confianza, sin acceso a la clave.
3. **Confirmación del investigador:** el investigador revisó las 112 imágenes en que la propuesta determinaba especie o asignaba `otro` (tres documentos con la imagen y el veredicto), confirmó 108 y decidió las 4 marcadas como dudosas. Dos de ellas (#30 y #295) quedaron como `indeterminable` por no ser visible el rasgo diagnóstico. Las 288 indeterminables no se revisaron, porque no intervienen en la tasa.

### Parte B — Resultado original

| Grupo | Revisadas | Indet. | Coinciden | Discrepan | Error |
|---|---|---|---|---|---|
| *Ae. aegypti* | 200 | 110 | 34 | 56 | 62,2 % |
| *Ae. albopictus* | 200 | 180 | 11 | 9 | 45,0 % |
| *aegypti* · Dryad | 60 | 9 | 0 | 51 | 100 % |
| *aegypti* · iNaturalist | 110 | 71 | 34 | 5 | 12,8 % |
| *aegypti* · Mosquito Alert | 5 | 5 | 0 | 0 | — |
| *aegypti* · Ong | 25 | 25 | 0 | 0 | — |
| *albopictus* · Dryad | 60 | 52 | 0 | 8 | 100 % |
| *albopictus* · Mosquito Alert | 115 | 103 | 11 | 1 | 8,3 % |
| *albopictus* · Ong | 25 | 25 | 0 | 0 | — |

Aplicada literalmente, la regla del protocolo (> 25 %: evaluar exclusión) señalaba Dryad en ambas especies.

### Parte C — Análisis de Dryad y corrección

Las 59 discrepancias de Dryad son veredictos `otro` asignados por "aspecto de *Culex*": cuerpo pardo, sin escamas blancas y patas sin bandas. Se concluye que el error está en el criterio aplicado, no en la etiqueta:

1. Es el aspecto de los ejemplares de colonia secos, que pierden las escamas. El hallazgo ya estaba documentado con Couret (Entrada 006) y figura como limitación en el informe de la Fase 3.
2. Aparece en las dos especies: todos los ejemplares de Dryad valorados (51 de *aegypti* y 8 de *albopictus*) se marcaron como `otro` y ninguno coincidió con su etiqueta. Un error de etiquetado no haría que ninguna de las dos especies tuviera un solo ejemplar reconocible; un criterio que no puede aplicarse a esa fuente, sí.
3. Dryad es una colección de laboratorio: la especie procede de la colonia de cría, no de una identificación posterior.

**Corrección.** En las fuentes de laboratorio (Dryad, Ong) el veredicto `otro` basado en la ausencia de escamas se reclasificó como `indeterminable`: el criterio no es aplicable a ejemplares desescamados. **La corrección se decidió después de conocer la clave**; se declara como tal y se conservan los dos resultados. Se apoya en un criterio documentado antes de la verificación, no en el resultado obtenido.

| Grupo | Determinables | Coinciden | Discrepan | Error |
|---|---|---|---|---|
| *Ae. aegypti* | 39 | 34 | 5 | 12,8 % |
| *Ae. albopictus* | 12 | 11 | 1 | 8,3 % |
| *aegypti* · iNaturalist | 39 | 34 | 5 | 12,8 % |
| *albopictus* · Mosquito Alert | 12 | 11 | 1 | 8,3 % |
| Dryad, Ong y Mosquito Alert *aegypti* | 0 | — | — | — |

### Parte D — Decisión por repositorio

| Repositorio | Resultado | Medida |
|---|---|---|
| iNaturalist (*aegypti*) | 12,8 % ≤ 15 % | Continuar y corregir: se excluyen las 5 imágenes discrepantes |
| Mosquito Alert (*albopictus*) | 8,3 % ≤ 15 % | Continuar y corregir: se excluye el reporte completo de la #326 |
| Dryad y Ong | Sin imágenes determinables | Plan para indeterminables, paso de revisión documental: la especie la respalda la publicación de cada colección (colonias identificadas). Sin entomólogo disponible en esta fase |
| Mosquito Alert (*aegypti*) | 5 en la muestra, todas indeterminables | Etiqueta validada por los expertos de la plataforma; sin medida adicional |

**Discrepancias excluidas:**

| # | Repositorio | Etiqueta | Observada | Motivo |
|---|---|---|---|---|
| 20 | iNaturalist | *aegypti* | otro | Aspecto de *Culex* |
| 54 | iNaturalist | *aegypti* | otro | Aspecto de *Culex* |
| 99 | iNaturalist | *aegypti* | *albopictus* | Franja central única |
| 152 | iNaturalist | *aegypti* | otro | Larva |
| 294 | iNaturalist | *aegypti* | otro | Larva (al microscopio) |
| 326 | Mosquito Alert | *albopictus* | *aegypti* | Lira visible; se excluye el reporte completo |

**Estadio vital en iNaturalist (criterio E7).** La aparición de dos larvas en 110 imágenes indica que E7 no actuó sobre iNaturalist: la descarga filtró por grado de investigación, pero no por estadio. Se excluyen las observaciones con anotación comunitaria de estadio no adulto (huevo, larva, pupa, ninfa o juvenil) no rechazada. Resultado de `corregir_verificacion.py`:

| Concepto | Valor |
|---|---|
| Observaciones de iNaturalist en el conjunto de trabajo | 1.093 |
| Con anotación de adulto | 412 |
| Con anotación de estadio no adulto (excluidas) | 26, todas larvas |
| Sin anotación de estadio | 655 (59,9 %) |
| Exclusiones totales de la Entrada 007 (`exclusiones_007.csv`) | 30: 5 de iNaturalist por verificación, 1 de Mosquito Alert y 26 larvas; las dos larvas de la muestra (#152 y #294) estaban anotadas y figuran en ambos motivos |

Entre las observaciones anotadas, el 5,9 % son larvas. Las dos larvas de la muestra estaban anotadas y no apareció ninguna en las aproximadamente 66 imágenes no anotadas de la muestra, lo que sugiere que las larvas tienden a anotarse.

**Revisión visual de las no anotadas** (`fase3/revisar_estadio_inat.py`). Para completar E7, el investigador revisó en hojas de miniaturas las 652 observaciones sin anotación de estadio que seguían en el conjunto; las otras 3 ya estaban excluidas por la verificación. **Resultado: 0 imágenes de estadio no adulto.** E7 queda aplicado sobre la totalidad de iNaturalist: 26 larvas excluidas por anotación y ninguna entre las no anotadas.

**Conjunto de trabajo tras las exclusiones (antes de rebalancear):**

| Repositorio | *Ae. aegypti* | *Ae. albopictus* |
|---|---|---|
| Dryad | 597 | 596 |
| Ong | 244 | 248 |
| iNaturalist | 1.064 (−29) | — |
| Mosquito Alert | 53 | 1.142 (−1) |
| **Total** | **1.958** | **1.986** |

El rebalanceo a 1.958 por especie se hará en la Entrada 008, reduciendo Mosquito Alert *Ae. albopictus* por reportes completos.

### Resumen

| Concepto | Valor |
|---|---|
| Imágenes revisadas (mín. 10 % por especie) | 200 + 200 (10,1 % de cada especie) |
| Imágenes con rasgo diagnóstico visible | 110 de 400 (27,5 %); 51 tras la corrección de laboratorio |
| Discrepancias detectadas | 65 (original); 6 tras la corrección |
| Tasa de error de etiquetado | Original: 62,2 % y 45,0 %. Corregida: 12,8 % (*aegypti*) y 8,3 % (*albopictus*) |
| ¿Supera el 15 %? | No, en los repositorios con imágenes determinables (tras la corrección) |
| ¿Supera el 25 %? | Solo Dryad en el resultado original, por inaplicabilidad del criterio |
| Medida aplicada | Exclusión de 30 imágenes: 5 de iNaturalist y 1 reporte de Mosquito Alert por verificación, y 26 larvas de iNaturalist por anotación de estadio (2 coinciden con la verificación); Dryad y Ong por revisión documental |

### Hallazgos

1. **La verificación visual de etiquetas tiene poco poder en fotografías de campo.** Solo el 27,5 % de las imágenes muestra el escudo: predominan vistas laterales, ejemplares diminutos en el encuadre e imágenes de 224 px. La tasa de error se estima sobre 51 imágenes, por lo que su intervalo de confianza es amplio.
2. **El criterio de exclusión de especie basado en el color no es transferible a ejemplares de laboratorio secos.** Es el mismo tipo de hallazgo que el de la Entrada 006: un criterio válido en una fuente mide otra cosa en otra.
3. **Los filtros de calidad de la plataforma no garantizan el estadio adulto.** El grado de investigación de iNaturalist confirma el taxón, no el estadio.

### Limitaciones declaradas

- La propuesta inicial la generó un asistente de IA y el investigador la confirmó; no intervino un entomólogo. Dryad y Ong se validan por su documentación, no por inspección.
- La tasa de error de Mosquito Alert *Ae. albopictus* se apoya en 12 imágenes determinables.
- La revisión de estadio de las 652 observaciones no anotadas se hizo sobre miniaturas: descarta larvas y pupas evidentes, no estadios ambiguos.
- La reclasificación de laboratorio se decidió tras conocer la clave (ver Parte C).

### Desviaciones respecto al protocolo

| Aspecto | Protocolo | Ejecución | Justificación |
|---|---|---|---|
| Ejecutor | Investigador principal | Propuesta a ciegas asistida por IA y confirmación del investigador | Volumen de la muestra; la decisión final y los casos dudosos quedaron a cargo del investigador |
| Categorías | `aegypti`, `albopictus`, `indeterminable` | Se añade `otro` | Aparecieron ejemplares que no son *Aedes* y una larva |
| Lectura de la tasa | Directa | Reclasificación de `otro` en fuentes de laboratorio | Criterio no aplicable a ejemplares desescamados (Parte C) |

---

## Entrada 007 bis — Protocolo de evaluación (registro previo a la partición)

**Fecha:** 2026-10-24
**Documento:** `fase2/protocolo_evaluacion.md` (v1.0)
**SHA-256:** `0a88a99c6235d9f68432fd908e25f3cb3c7ee927954e7cfda98f98618376cb31`

Atiende la observación del tutor del 23/10/2026: resultados aparte con Dryad y Ong, métricas por repositorio y Grad-CAM para comprobar si el modelo mira el mosquito o el fondo, todo fijado antes de dividir el conjunto. La huella SHA-256 permite verificar que el protocolo no se modificó después de ver resultados (`sha256sum fase2/protocolo_evaluacion.md`).

| Elemento | Decisión fijada |
|---|---|
| Modelo evaluado | Corrida mediana de cada arquitectura (igual que en la sección 3.8) |
| Estratos | Dryad, Ong, iNaturalist y Mosquito Alert, por especie |
| Subconjunto de control | Dryad + Ong de prueba: ambas especies con el mismo procedimiento fotográfico |
| Intervalos | *Bootstrap* por grupo (espécimen), 2.000 réplicas, semilla 42 |
| Criterio de control | Δ_control = macro F1 (prueba) − macro F1 (control) ≥ 0,05 con IC 95 % que excluye 0 → limitación declarada |
| Sensibilidad | Jerarquía de selección de la sección 3.8 repetida sobre el control; si el orden cambia, la conclusión se formula con salvedad |
| Atajo por plataforma | *Ae. aegypti* de Mosquito Alert en prueba y *Ae. albopictus* ecuatorianos, de forma descriptiva |
| Grad-CAM | Unas 68 imágenes de prueba estratificadas, fijadas en la partición; caja del espécimen trazada antes de entrenar; indicadores: pico dentro de la caja y concentración |
| Alerta de fondo | > 25 % de picos fuera de la caja (total o estrato con ≥ 8) o mediana de concentración ≤ 1 |
| Piloto | Mismo procedimiento sobre 20 imágenes de validación; > 50 % de picos fuera → consulta antes de la Fase 5 |

Grad-CAM conserva su carácter complementario: no interviene en la selección del mejor modelo ni en las pruebas estadísticas.

### Desviaciones respecto al plan

| Aspecto | Plan | Protocolo | Justificación |
|---|---|---|---|
| Muestra de Grad-CAM | 20 imágenes, 10 por especie | ≈ 68, estratificadas por repositorio | Observación del tutor; permite leer el resultado por tipo de fotografía |
| Lectura de Grad-CAM | Ilustrativa, sin categorización (2.2.9, 3.1) | Dos indicadores simples con umbral de alerta fijado de antemano | Responder con un criterio verificable si el modelo mira el mosquito o el fondo |
| "Diferencia sustancial" entre fuentes (3.4) | Sin umbral | Δ ≥ 0,05 en macro F1 con IC 95 % que excluye 0 | Fijar el criterio antes de ver resultados |

---

## Entrada 008 — Rebalanceo y división del conjunto

**Fecha:** 2026-10-24
**Scripts:** `fase3/particion_conjunto.py` y `fase3/anotar_cajas.py`
**Protocolo aplicado:** `fase2/protocolo_evaluacion.md` v1.0 (SHA-256 verificado en el equipo antes de dividir)

**Decisión de calendario.** El tutor no respondió a tiempo sobre el protocolo de evaluación. Para cumplir el cronograma, se avanzó con la partición sin su visto bueno. Se le informó por Teams y el protocolo quedó registrado con su huella antes de dividir. Si el tutor pide cambios, se aplicarán como desviaciones registradas: la partición no se regenera.

### Parte A — Rebalanceo

| Paso | *Ae. aegypti* | *Ae. albopictus* |
|---|---|---|
| Conjunto de trabajo (opción A) | 1.987 | 1.987 |
| Exclusiones de la Entrada 007 | −29 | −1 |
| Tras exclusiones | 1.958 | 1.986 |
| Rebalanceo: 17 reportes completos de Mosquito Alert retirados al azar (semilla 42) | — | −28 |
| **Conjunto final** | **1.958** | **1.958** |

Solo se retiraron reportes de Mosquito Alert *Ae. albopictus*, la fuente sobrante; las demás fuentes se conservan íntegras. Lista en `fase3/particion/rebalanceo_retiradas.csv`.

### Parte B — Partición

70 / 15 / 15, fija, agrupada por espécimen y estratificada por especie × repositorio, con semilla 42. Dentro de cada estrato, los grupos se ordenan por tamaño (desempate aleatorio) y cada uno se asigna al subconjunto con mayor déficit respecto a su cuota. Esto sustituye a las funciones de partición agrupada de scikit-learn mencionadas en el plan (3.4), porque permite estratificar a la vez por especie y por repositorio con grupos de tamaño desigual y respetar exactamente las proporciones.

| Subconjunto | Especie | Dryad | Ong | iNaturalist | Mosquito Alert | Total | Fuente mayor |
|---|---|---|---|---|---|---|---|
| Entrenamiento | *Ae. aegypti* | 417 | 170 | 744 | 37 | 1.368 | 54,4 % |
| Entrenamiento | *Ae. albopictus* | 416 | 173 | — | 780 | 1.369 | 57,0 % |
| Validación | *Ae. aegypti* | 90 | 37 | 160 | 8 | 295 | 54,2 % |
| Validación | *Ae. albopictus* | 90 | 38 | — | 167 | 295 | 56,6 % |
| Prueba | *Ae. aegypti* | 90 | 37 | 160 | 8 | 295 | 54,2 % |
| Prueba | *Ae. albopictus* | 90 | 37 | — | 167 | 294 | 56,8 % |
| **Total** | | **1.193** | **492** | **1.064** | **1.167** | **3.916** | |

| Subconjunto | *Ae. aegypti* | *Ae. albopictus* | Total |
|---|---|---|---|
| Entrenamiento (69,9 %) | 1.368 | 1.369 | 2.737 |
| Validación (15,1 %) | 295 | 295 | 590 |
| Prueba (15,0 %) | 295 | 294 | 589 |

| Verificación | Resultado |
|---|---|
| Estratificación por especie | Sí: diferencia máxima de 1 imagen entre especies por subconjunto |
| Estratificación por repositorio | Sí: cada repositorio aporta 70/15/15 dentro de cada especie |
| Agrupamiento por espécimen (sin fuga) | 2.459 grupos; **0** repartidos entre subconjuntos |
| Exclusiones de la Entrada 007 presentes | 0 |
| Mosquito Alert solo con reporte verificable | Sí (heredado del conjunto de trabajo) |
| Subconjunto de control en prueba (Dryad + Ong) | 127 *Ae. aegypti* y 127 *Ae. albopictus* |
| Composición por tipo de fotografía igualada entre especies (opción A) | Sí: control 43 % y campo 57 % en ambas especies |
| Medida M1 (ninguna fuente > 70 %) | Cumple en los seis cortes; máximo 57,0 % |
| Imágenes tras aumento ×2 en entrenamiento | 8.211 previstas (2.737 originales + 5.474 aumentadas); se generan en la Fase 5 |
| Enlaces duros en `fase3/conjunto_final/` | 3.916 |

### Parte C — Muestras fijadas en la partición

| Muestra | Contenido | Archivo |
|---|---|---|
| Grad-CAM (Fase 6) | 64 imágenes de prueba, una por grupo: 10 por estrato D-ae, D-al, O-ae, O-al, I-ae y M-al, y 4 de M-ae (las 8 imágenes de prueba de ese estrato pertenecen a 4 reportes) | `gradcam_muestra.csv` |
| Piloto, entrenamiento (Fase 4) | 400 imágenes (200 por especie), proporcionales por repositorio y por grupos completos | `piloto_entrenamiento.csv` |
| Piloto, Grad-CAM (Fase 4) | 20 imágenes de validación (10 por especie) | `piloto_gradcam.csv` |

### Parte D — Cajas del espécimen (protocolo, 5.2)

El investigador trazó las 84 cajas (64 + 20) con `anotar_cajas.py` el 2026-10-24, **antes de que exista ningún modelo entrenado**. Se revisaron en las láminas de `fase3/particion/cajas_laminas/`.

| Estrato | Informativas (caja ≤ 60 % del área) | Comentario |
|---|---|---|
| Dryad (D-ae + D-al) | 3 de 20 | Primeros planos de laboratorio: el ejemplar con las patas ocupa casi toda la imagen |
| Ong (O-ae + O-al) | 20 de 20 | Mosquito pequeño sobre la piel |
| iNaturalist (I-ae) | 8 de 10 | |
| Mosquito Alert (M-al) | 10 de 10 | |
| Mosquito Alert (M-ae) | 2 de 4 | |
| **Prueba, total** | **43 de 64** | |
| Piloto | 18 de 20 | |

**Consecuencia prevista por el protocolo.** En Dryad el indicador de Grad-CAM casi no discrimina entre mosquito y fondo, así que se reportará de forma descriptiva y sin criterio de alerta por estrato (hay menos de 8 informativas). No afecta a la pregunta del tutor: en Dryad el fondo es el mismo para las dos especies y no puede delatar la especie. La lectura del control frente al fondo descansa en Ong.

### Estado

Con esta entrada el conjunto queda depurado, verificado, balanceado y dividido. La Fase 3 se cierra en cuanto a datos; quedan pendientes las actualizaciones documentales (informe de la Fase 3 y correcciones al plan) registradas en el anexo de cambios.

---

# FASE 4 — Prueba piloto

_(Entradas a partir de S11)_

## Entrada 009 — Prueba piloto

**Fecha:** 2026-10-27 (ejecución terminada a las 08:54)
**Script:** `fase4/entrenar.py --modo piloto`
**Salidas:** `fase4/log_piloto.txt`, `fase4/piloto/reporte_piloto.json` y, por arquitectura, `curvas.csv`, `metricas_val.json`, `predicciones_val.csv`, `modelo.pt`, `gradcam.csv` y `gradcam.jpg`

**Parámetros declarados:** 400 imágenes de entrenamiento (200 por especie, `piloto_entrenamiento.csv`), 5 épocas, 4 arquitecturas, 1 configuración, 1 semilla.

| Parámetro | Valor |
|---|---|
| Configuración | lr 5e-4 (valor central del espacio de búsqueda), batch 32, AdamW con weight decay 0,01, entropía cruzada |
| Aumento | ×2: 1.200 imágenes por época |
| Validación | Subconjunto de validación completo (590 imágenes) |
| Semilla | 42 |
| Entorno | Python 3.11.13, torch 2.4.1+cpu, timm 1.0.9, scikit-learn 1.5.2, grad-cam 1.5.4, 6 hilos |
| Copia reducida | 990 imágenes a ≤ 512 px en 51 s |

**Criterios de aceptación:**

| Criterio | EfficientNet-B0 | MobileNetV3-Large | ResNet50 | ViT-S/16 |
|---|---|---|---|---|
| Completa sin errores ni agotamiento de memoria | Sí | Sí | Sí | Sí |
| Pérdida con tendencia decreciente (0,94 → 0,07 / 1,28 → 0,04 / 0,58 → 0,02 / 0,71 → 0,04) | Sí | Sí | Sí | Sí |
| Métricas calculadas sobre validación | Sí | Sí | Sí | Sí |
| Grad-CAM se ejecuta sin fallos | Sí | Sí | Sí | Sí |
| Tiempo por corrida completa ≤ 6 h | 1,20 h | 0,70 h | 2,78 h | 2,27 h |

**Resultado: las cuatro arquitecturas superan los criterios de aceptación.**

**Extrapolación** (regla del plan, con las imágenes reales por época: 2.737 × 3 = 8.211):

| Concepto | Valor |
|---|---|
| Tiempo de entrenamiento del piloto | 200 s, 116 s, 461 s y 379 s respectivamente |
| Tiempo extrapolado por corrida completa (15 épocas, con validación) | 1,20 h, 0,70 h, 2,78 h y 2,27 h |
| Total extrapolado (20 definitivas + 12 de búsqueda) | 34,8 h + 4,2 h = **38,9 h** |
| ¿Excede las 240 h? | No (16 % del disponible) |
| Medida correctiva aplicada | Ninguna: se mantienen 224 × 224 px y 15 épocas |

**Contraste con el benchmark de la Fase 2:**

| Arquitectura | Fase 2 (img/s) | Piloto (img/s) | Factor real |
|---|---|---|---|
| EfficientNet-B0 | 39,3 | 30,0 | 0,76 |
| MobileNetV3-Large | 66,1 | 52,0 | 0,79 |
| ResNet50 | 16,6 | 13,0 | 0,78 |
| ViT-S/16 | 25,9 | 15,8 | 0,61 |

La Fase 2 midió con tensores en memoria y entrenando solo el clasificador. Con lectura de disco, aumento y entrenamiento de los últimos bloques, el rendimiento cae entre un 21 % y un 39 %. La caída es mucho menor que el factor conservador de 3 supuesto en la Fase 2, porque la copia reducida elimina la decodificación de fotografías grandes. ViT-S/16 es la que más pierde, porque entrena bloques completos de atención.

**Métricas en validación** (solo verifican que el cálculo funciona; con 400 imágenes y 5 épocas no estiman el rendimiento final):

| Arquitectura | Macro F1 | Exactitud | Mejor época | RAM pico |
|---|---|---|---|---|
| EfficientNet-B0 | 0,708 | 0,709 | 4 | 1.774 MB |
| MobileNetV3-Large | 0,710 | 0,710 | 1 | 1.774 MB |
| ResNet50 | 0,718 | 0,719 | 4 | 2.038 MB |
| ViT-S/16 | 0,759 | 0,759 | 3 | 2.136 MB |

La pérdida de entrenamiento llega a 0,02–0,07 mientras el macro F1 de validación se estanca hacia 0,70–0,76: hay sobreajuste, esperable con 400 imágenes. En la Fase 5 lo controlan la selección del *checkpoint* por validación y la detención temprana.

**Grad-CAM en el piloto** (protocolo de evaluación, apartado 6; 20 imágenes de validación, 18 informativas):

| Arquitectura | Picos fuera de la caja | Mediana de la concentración | Alerta (> 50 %) |
|---|---|---|---|
| EfficientNet-B0 | 6 de 18 (33 %) | 2,12 | No |
| MobileNetV3-Large | 8 de 18 (44 %) | 1,45 | No |
| ResNet50 | 3 de 18 (17 %) | 3,13 | No |
| ViT-S/16 | 11 de 18 (61 %) | 1,33 | **Sí** |

**Lectura de la alerta de ViT-S/16.** Sus mapas muestran manchas puntuales repartidas por toda la imagen, fondo incluido, en lugar de una región continua. En las redes convolucionales la activación se concentra sobre el ejemplar. Ese patrón es conocido en Grad-CAM aplicado a transformadores, donde algunos *tokens* de fondo acumulan activaciones altas, y no indica por sí solo que la decisión dependa del fondo: la concentración se mantiene por encima de 1 y ViT-S/16 obtuvo el mejor macro F1 del piloto. El protocolo ya declara que las visualizaciones de ambas familias no son equivalentes.

**Decisión.** El protocolo prevé documentar y consultar con el tutor antes de la Fase 5. Conforme a la decisión de calendario de la Entrada 008, se informa al tutor y se continúa. No se aplica ninguna medida correctiva: el piloto es descriptivo y el criterio de alerta definitivo se aplica en la Fase 6 sobre la muestra de prueba. El hallazgo se reportará como limitación de la interpretabilidad de ViT-S/16.

### Curva de aprendizaje (plan 3.4)

El plan prevé verificar el umbral de 1.000 imágenes por especie en el piloto mediante la curva de aprendizaje. Se entrena MobileNetV3-Large, la arquitectura más rápida, con la configuración del piloto sobre fracciones anidadas del entrenamiento (25, 50, 75 y 100 %), por grupos completos y estratificadas por especie × repositorio.

**Criterio fijado antes de ejecutar:** el rendimiento se considera estabilizado si el macro F1 en validación mejora menos de 0,02 entre el 75 % y el 100 %.

**Ejecución:** `fase4/entrenar.py --modo curva` (salida en `fase4/log_curva.txt` y `fase4/curva/reporte_curva.json`). Configuración del piloto: lr 5e-4, batch 32, 5 épocas, aumento ×2, semilla 42; validación completa (590).

| Fracción | Imágenes (*aegypti* / *albopictus*) | Macro F1 en validación | Mejor época |
|---|---|---|---|
| 25 % | 689 (345 / 344) | 0,7253 | 2 |
| 50 % | 1.376 (688 / 688) | 0,7838 | 4 |
| 75 % | 2.057 (1.030 / 1.027) | 0,8051 | 4 |
| 100 % | 2.737 (1.368 / 1.369) | 0,8220 | 5 |
| Mejora 75 → 100 % | | **+0,0169** | |
| ¿Estabilizado según el criterio (< 0,02)? | | **Sí** | |

**Lectura.** La ganancia se reduce a cada paso (+0,059, +0,021 y +0,017), de modo que la curva se aplana, y el criterio fijado se cumple. El tamaño de 1.958 imágenes por especie se considera suficiente y no se amplía la recopilación (plan 3.4).

Hay dos matices que se declaran. Primero, la mejora final (0,0169) queda cerca del umbral y la curva todavía sube: con más datos el rendimiento podría crecer algo más, pero con ganancias marginales. Segundo, con una sola semilla y 5 épocas el macro F1 oscila hasta 0,05 entre épocas consecutivas (de 0,7253 a 0,6698 con el 25 %), una variación mayor que la diferencia evaluada. La conclusión es orientativa: indica rendimientos decrecientes, no un techo exacto.

El tiempo medido con el conjunto completo (≈ 146 s por época, 56 img/s) confirma la extrapolación del piloto para MobileNetV3-Large.

### Cierre de la Fase 4

| Producto esperado (Tabla 4) | Estado |
|---|---|
| Reporte de validación del entorno experimental | Completado: cuatro arquitecturas aceptadas |
| Estimación del tiempo por corrida | Completado: 0,70 a 2,78 h; 38,9 h para la Fase 5 frente a 240 h |
| Verificación del tamaño del conjunto (plan 3.4) | Completado: curva estabilizada según el criterio |
| Grad-CAM en el piloto (protocolo, 6) | Completado: alerta en ViT-S/16, documentada |

---

# FASE 5 — Implementación y entrenamiento

_(Entradas a partir de S12)_

## Entrada 010 — Búsqueda de hiperparámetros

**Fecha:** 2026-11-17 (terminada a las 02:21:36, según `reporte_busqueda.json`)
**Script:** `fase4/entrenar.py --modo busqueda`
**Salidas:** `fase5/log_busqueda.txt`, `fase5/busqueda/reporte_busqueda.json`, `fase5/busqueda/configuracion_seleccionada.json` y, por configuración, `curvas.csv`, `metricas_val.json` y `predicciones_val.csv` (los *checkpoints* de la búsqueda no se conservan)

| Parámetro | Valor |
|---|---|
| Espacio común | lr {1e-3, 5e-4, 1e-4} × batch {16, 32} (ficha técnica, 5) |
| Configuraciones evaluadas | 3, elegidas al azar con semilla 42 y **las mismas para las cuatro arquitecturas**: (1e-3, 16), (1e-4, 16), (1e-4, 32) |
| Submuestra | 825 imágenes (30 % del entrenamiento, por grupos completos y estratificada por especie × repositorio); 2.475 por época con el aumento ×2 |
| Épocas | Máximo 10, detención tras 5 sin mejora |
| Criterio de selección | Mayor macro F1 en validación (590 imágenes); el conjunto de prueba no interviene |

**Macro F1 en validación (mejor época de cada configuración):**

| Arquitectura | lr 1e-3, batch 16 | lr 1e-4, batch 16 | lr 1e-4, batch 32 | Seleccionada |
|---|---|---|---|---|
| EfficientNet-B0 | **0,7559** (8) | 0,7188 (8) | 0,7267 (10) | lr 1e-3, batch 16 |
| MobileNetV3-Large | **0,7423** (8) | 0,7339 (8) | 0,7270 (5) | lr 1e-3, batch 16 |
| ResNet50 | **0,7814** (10) | 0,7475 (6) | 0,7420 (6) | lr 1e-3, batch 16 |
| ViT-S/16 | 0,8186 (7) | **0,8488** (4) | 0,8401 (3) | lr 1e-4, batch 16 |

Entre paréntesis, la época del mejor macro F1.

**Duración y rendimiento** (`reporte_busqueda.json`):

| Arquitectura | Horas (3 configuraciones) | Entrenamiento (img/s) |
|---|---|---|
| EfficientNet-B0 | 0,66 | ≈ 34–40 |
| MobileNetV3-Large | 0,39 | ≈ 58–64 |
| ResNet50 | 1,57 | ≈ 15–16 |
| ViT-S/16 | 1,36 | ≈ 13–18 |
| **Total** | **3,98** (14.322 s; estimado en el piloto: 4,2 h) | |

Dos configuraciones terminaron antes del límite por detención temprana: ViT-S/16 con lr 1e-4 y batch 16 (9 épocas) y con batch 32 (8 épocas).

**Observaciones.**

1. Las tres redes convolucionales eligieron la tasa más alta del espacio (1e-3) y ViT-S/16 la más baja (1e-4). Es coherente con la práctica habitual: los transformadores se ajustan con tasas menores. El espacio está fijado en el plan y no se amplía; se registra que en las CNN el óptimo queda en el borde del espacio explorado.
2. La configuración (5e-4, ·), usada en el piloto, no salió en el sorteo. Al tratarse de una búsqueda aleatoria con presupuesto fijo, es un resultado esperable y no se corrige.
3. Las diferencias entre configuraciones de una misma arquitectura (0,01 a 0,04) son del mismo orden que la oscilación entre épocas (hasta 0,08 en ViT-S/16 con lr 1e-3), de modo que la elección es sensible al ruido de una sola semilla. Es la limitación inherente a un presupuesto de tres configuraciones, igual para todas las arquitecturas.
4. Con lr 1e-3, ViT-S/16 fue la más inestable (de 0,8186 a 0,7359 entre épocas consecutivas).
5. ViT-S/16 alcanza su máximo pronto (épocas 3–4): en las corridas definitivas es previsible la detención temprana.

**Configuración que pasa a las corridas definitivas:**

| Arquitectura | lr | Batch |
|---|---|---|
| EfficientNet-B0 | 1e-3 | 16 |
| MobileNetV3-Large | 1e-3 | 16 |
| ResNet50 | 1e-3 | 16 |
| ViT-S/16 | 1e-4 | 16 |

**Estimación de las corridas definitivas** con el rendimiento medido a batch 16 (8.211 imágenes por época, 15 épocas como máximo): EfficientNet-B0 ≈ 1,0 h, MobileNetV3-Large ≈ 0,6 h, ResNet50 ≈ 2,3 h y ViT-S/16 ≈ 2,2 h por corrida; **≈ 30 h para las 20 corridas**, sin contar la detención temprana.

---

## Entrada 011 — Corridas definitivas

**Fecha:** 2026-11-21, 07:00:40, a 2026-11-22, 04:01:37 (una sola ejecución, sin interrupciones)
**Script:** `fase4/entrenar.py --modo definitiva` (lanzado con `python -u`)
**Salidas:** `fase5/log_definitivas.txt`, `fase5/definitivas/resumen_definitivas.json` y, por arquitectura y semilla (`fase5/definitivas/<arquitectura>/semilla_<s>/`), `modelo.pt`, `curvas.csv`, `metricas_val.json` y `predicciones_val.csv`

| Parámetro | Valor |
|---|---|
| Configuración | La seleccionada en la Entrada 010 (`configuracion_seleccionada.json`): lr 1e-3 y batch 16 en las tres CNN; lr 1e-4 y batch 16 en ViT-S/16 |
| Entrenamiento | 2.737 imágenes; 8.211 por época con el aumento ×2 |
| Validación | 590 imágenes; el conjunto de prueba no se usó |
| Semillas | 42, 43, 44, 45 y 46, las mismas para las cuatro arquitecturas (20 corridas) |
| Épocas | Máximo 15, detención tras 5 sin mejora; *checkpoint* de la época con mayor macro F1 en validación |
| Entorno | Python 3.11.13, torch 2.4.1+cpu, timm 1.0.9, scikit-learn 1.5.2, 6 hilos |

La semilla cambia el orden de las imágenes, el aumento aleatorio y la inicialización de la cabeza de clasificación; los pesos de ImageNet son los mismos en las cinco corridas. La tabla siguiente recoge las 20 corridas.

**Resultado por corrida** (macro F1 en validación de la mejor época):

| Arquitectura | Semilla | Épocas | Mejor época | Detención temprana | Macro F1 | Exactitud | Horas |
|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | 42 | 12 | 7 | Sí | 0,8320 | 0,8322 | 0,747 |
| EfficientNet-B0 | 43 | 10 | 5 | Sí | 0,8485 | 0,8492 | 0,598 |
| EfficientNet-B0 | 44 | 9 | 4 | Sí | 0,8185 | 0,8186 | 0,542 |
| EfficientNet-B0 | 45 | 14 | 9 | Sí | 0,8610 | 0,8610 | 0,843 |
| EfficientNet-B0 | 46 | 15 | 14 | No | 0,8475 | 0,8475 | 0,908 |
| MobileNetV3-Large | 42 | 7 | 2 | Sí | 0,8305 | 0,8305 | 0,263 |
| MobileNetV3-Large | 43 | 9 | 4 | Sí | 0,8203 | 0,8203 | 0,339 |
| MobileNetV3-Large | 44 | 12 | 7 | Sí | 0,8196 | 0,8203 | 0,453 |
| MobileNetV3-Large | 45 | 11 | 6 | Sí | 0,8185 | 0,8186 | 0,415 |
| MobileNetV3-Large | 46 | 15 | 13 | No | 0,8271 | 0,8271 | 0,565 |
| ResNet50 | 42 | 14 | 9 | Sí | 0,8711 | 0,8712 | 2,116 |
| ResNet50 | 43 | 8 | 3 | Sí | 0,8610 | 0,8610 | 1,229 |
| ResNet50 | 44 | 7 | 2 | Sí | 0,8491 | 0,8492 | 1,074 |
| ResNet50 | 45 | 7 | 2 | Sí | 0,8470 | 0,8475 | 1,068 |
| ResNet50 | 46 | 6 | 1 | Sí | 0,8424 | 0,8424 | 0,910 |
| ViT-S/16 | 42 | 15 | 11 | No | 0,8607 | 0,8610 | 2,083 |
| ViT-S/16 | 43 | 9 | 4 | Sí | 0,8675 | 0,8678 | 1,238 |
| ViT-S/16 | 44 | 14 | 9 | Sí | 0,8780 | 0,8780 | 1,920 |
| ViT-S/16 | 45 | 15 | 15 | No | 0,8845 | 0,8847 | 2,056 |
| ViT-S/16 | 46 | 12 | 7 | Sí | 0,8745 | 0,8746 | 1,648 |

**Resumen por arquitectura (validación):**

| Arquitectura | Media macro F1 | DE | Mínimo | Máximo | Semilla de la corrida mediana | Horas (5 corridas) |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | 0,8415 | 0,0165 | 0,8185 | 0,8610 | 46 (0,8475) | 3,64 |
| MobileNetV3-Large | 0,8232 | 0,0053 | 0,8185 | 0,8305 | 43 (0,8203) | 2,04 |
| ResNet50 | 0,8541 | 0,0117 | 0,8424 | 0,8711 | 44 (0,8491) | 6,40 |
| ViT-S/16 | 0,8730 | 0,0092 | 0,8607 | 0,8845 | 46 (0,8745) | 8,95 |

La corrida mediana es la tercera de las cinco ordenadas por macro F1 en validación (plan 3.8; protocolo de evaluación). No hubo empates. Son los modelos que se evalúan en la Fase 6: EfficientNet-B0 semilla 46, MobileNetV3-Large semilla 43, ResNet50 semilla 44 y ViT-S/16 semilla 46.

**Duración y eficiencia:**

| Arquitectura | Horas por corrida (media) | Estimado en la Entrada 010 | Entrenamiento (img/s) | Inferencia (ms/img, mediana) | Tamaño del modelo |
|---|---|---|---|---|---|
| EfficientNet-B0 | 0,73 | ≈ 1,0 | ≈ 40 | 18,98 | 15,57 MB |
| MobileNetV3-Large | 0,41 | ≈ 0,6 | ≈ 64 | 11,86 | 16,23 MB |
| ResNet50 | 1,28 | ≈ 2,3 | ≈ 16 | 50,68 | 89,99 MB |
| ViT-S/16 | 1,79 | ≈ 2,2 | ≈ 17 | 39,83 | 82,70 MB |
| **Total** | **21,03 h** (20 corridas, 221 épocas) | ≈ 30 h | | | |

**Cómputo total de la Fase 5:** 3,98 h de búsqueda + 21,03 h de corridas definitivas = **25,0 h**, frente a 240 h disponibles (10 %) y a 38,9 h estimadas en el piloto. La diferencia con lo estimado se debe a la detención temprana: 16 de las 20 corridas pararon antes de la época 15.

**Alcance de las medidas de eficiencia.**

1. La inferencia (ms/img) es el tiempo de validar la mejor época dividido entre 590. Se midió con lotes de 64 imágenes y 2 procesos de lectura, e incluye la lectura y el redimensionado desde la copia reducida. Es un rendimiento por imagen en lote, no la latencia de una imagen aislada. Es coherente con la Fase 2 (22,9 / 13,4 / 58,6 / 41,1 ms/img con lotes de 16).
2. La RAM pico registrada no sirve para comparar arquitecturas (ver incidencia del 2026-11-24). Se volverá a medir en la Fase 6.

**Observaciones.**

1. **El orden en validación** es ViT-S/16 > ResNet50 > EfficientNet-B0 > MobileNetV3-Large. Las cinco corridas de ViT-S/16 (mínimo 0,8607) superan la media de ResNet50 (0,8541). MobileNetV3-Large es la más estable (DE 0,0053) y EfficientNet-B0 la menos estable (DE 0,0165). **No es el resultado de la comparación:** la selección del mejor modelo se hace con el conjunto de prueba en la Fase 6.
2. **El macro F1 de validación es optimista.** El *checkpoint* es el máximo de hasta 15 evaluaciones sobre las mismas 590 imágenes, y el macro F1 oscila mucho entre épocas consecutivas: de 0,8178 a 0,7630 en EfficientNet-B0 con la semilla 43, de 0,8475 a 0,7879 con la semilla 46 y de 0,8337 a 0,7850 en ViT-S/16 con la semilla 45. Es previsible que el macro F1 en prueba quede por debajo del de validación. Por eso la comparación se hace sobre el conjunto de prueba, que no intervino ni en la búsqueda ni en la selección del *checkpoint*.
3. **ResNet50 alcanza su máximo muy pronto.** En 4 de 5 corridas la mejor época fue la 1, 2 o 3, y la pérdida de entrenamiento baja a 0,02–0,04 mientras la validación se estanca. Con lr 1e-3 y 14,97 M de parámetros entrenables, se sobreajusta rápido. La detención temprana y la selección del *checkpoint* lo contienen. Se registra como posible efecto de que la búsqueda le asignara la tasa más alta del espacio (Entrada 010, observación 1).
4. **Cuatro corridas llegaron al límite de 15 épocas sin detenerse**: EfficientNet-B0 46 (mejor época 14), MobileNetV3-Large 46 (13), ViT-S/16 42 (11) y ViT-S/16 45 (15). En ViT-S/16 45 la mejor época fue la última, así que podría haber mejorado con más épocas. El límite de 15 está fijado en el plan y no se modifica. Contra lo previsto en la Entrada 010, ViT-S/16 solo se detuvo antes en 3 de 5 corridas.
5. En MobileNetV3-Large la pérdida de entrenamiento no siempre baja de una época a otra (por ejemplo, 0,1129 → 0,1929 con la semilla 42). Es ruido del optimizador con lr 1e-3 y lotes de 16; no impidió que fuera la arquitectura más estable entre semillas.
6. El aviso de Hugging Face sobre solicitudes sin autenticar es informativo: los pesos se descargaron sin errores en las 20 corridas.

**Estado:** con esta entrada se completa la Fase 5. Los 20 *checkpoints* se conservan; la Fase 6 usa los cuatro de las corridas medianas.

---

## Entrada 011 bis — Anexo del protocolo para la Fase 6 (registro previo a la evaluación)

**Fecha:** 2026-11-24
**Documento:** `fase2/protocolo_evaluacion_anexo_fase6.md` (v1.0)
**SHA-256:** `e1695322fcc6b7868fb6f00145b4eb2d19fb8c2f48d443151569ac394709ebfe`
**Script:** `fase6/evaluar.py`

Se registra **antes de evaluar ningún modelo sobre la prueba o sobre el conjunto ecuatoriano**. Fija los detalles de aplicación que el plan (3.7 y 3.8) y el protocolo v1.0 dejan abiertos. No cambia ningún umbral del protocolo v1.0.

| Elemento | Decisión fijada |
|---|---|
| Niveles de análisis | Nivel 1: las 20 corridas en prueba (media, DE, mínimo y máximo). Nivel 2: corrida mediana de cada arquitectura (EfficientNet-B0 46, MobileNetV3-Large 43, ResNet50 44, ViT-S/16 46) |
| Pruebas | Q de Cochran y W = Q / (n · (k − 1)); McNemar exacta con Bonferroni (× 6), interpretada solo si la Q es significativa; diferencia de exactitudes con IC de Wald pareado y, como sensibilidad, IC por *bootstrap* de grupos |
| *Bootstrap* | Por grupo, estratificado por especie × repositorio; 2.000 réplicas, semilla 42; las mismas réplicas para todas las arquitecturas |
| Jerarquía (3.8) | Empate en macro F1 o en sensibilidad de *Ae. aegypti* si la diferencia es menor de 0,02; después, menor DE del macro F1 en prueba; después, ms por imagen y tamaño. Se repite sobre el control |
| Eficiencia | Cada arquitectura en un proceso aparte: ms por imagen con lote de 1 (mediana de 100), ms en lote de 64, RAM residente y tamaño del *checkpoint*. Corrige la medida de RAM de la Fase 5 |
| Conjunto ecuatoriano | Análisis principal según el plan, con IC por *bootstrap* **por observación**. Análisis secundarios fijados de antemano: por observación, sin la observación 156329843, con las 119 imágenes y caso por caso para *Ae. albopictus* |
| Fuga | Cruce de `observation_id` y `photo_id` con la procedencia de iNaturalist del entrenamiento |

**Hallazgo al revisar el conjunto ecuatoriano (sin predicciones).** Las 71 imágenes `research` de *Ae. aegypti* proceden de solo **19 observaciones**. Una observación (156329843) aporta 32 imágenes (45 %): acercamientos al microscopio de partes de un mismo ejemplar, muchas sin el mosquito completo. Un mismo observador aporta 7 observaciones (43 imágenes). Las imágenes no son independientes. Por eso el IC se calcula por observación y se fijan de antemano los análisis sin esa observación y por observación. El conjunto incluye además imágenes con varios mosquitos, ejemplares diminutos y una etiqueta escrita con el nombre de la especie (observación `casual`, fuera del análisis principal). Las imágenes no pasan por la depuración E1–E9; se declara como limitación.

### Desviaciones respecto al plan

| Aspecto | Plan | Anexo | Justificación |
|---|---|---|---|
| IC en el conjunto ecuatoriano | No previsto | *Bootstrap* por observación | 71 imágenes de 19 observaciones; una aporta el 45 % |
| Análisis secundarios del conjunto ecuatoriano | Solo las 119 imágenes | Además, por observación y sin la observación 156329843 | Mismo motivo; fijados antes de evaluar |
| Medida de RAM | RAM pico por corrida (Fase 5) | RAM residente por arquitectura en un proceso aparte | La medida de la Fase 5 era acumulada (incidencia del 2026-11-24) |

---

# FASE 6 — Pruebas y recolección de datos

_(Entradas a partir de S17)_

## Entrada 012 — Evaluación sobre el conjunto de prueba

**Fecha:** 2026-11-25 (ejecución completa de 9,8 min, tras un primer intento interrumpido; ver incidencias)
**Script:** `fase6/evaluar.py`
**Protocolos aplicados:** `protocolo_evaluacion.md` v1.0 (SHA-256 `0a88a99c…cb31`) y `protocolo_evaluacion_anexo_fase6.md` v1.0 (`e1695322…ebfe`), ambas huellas verificadas por el script al iniciar
**Salidas:** `fase6/log_evaluacion.txt`, `fase6/reporte_fase6.json`, `fase6/metricas_prueba.json`, `fase6/metricas_por_estrato.csv`, `fase6/estabilidad_5_corridas.csv`, `fase6/control_dryad_ong.json`, `fase6/atajo_plataforma.csv`, `fase6/gradcam_*`, `fase6/eficiencia.json`, `fase6/figuras/` y `fase6/predicciones/` (20 modelos)

El conjunto de prueba (589 imágenes, 369 grupos) se usó aquí por primera vez. Todas las cifras son de la **corrida mediana** de cada arquitectura (EfficientNet-B0 46, MobileNetV3-Large 43, ResNet50 44, ViT-S/16 46), salvo las de estabilidad. Los IC 95 % son por *bootstrap* de grupos estratificado (2.000 réplicas, semilla 42).

### Parte A — Métricas globales

| Arquitectura | Exactitud | Exact. balanceada | Macro F1 (IC 95 %) | F1 *aegypti* | F1 *albopictus* | Sens. *aegypti* | Sens. *albopictus* |
|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | 0,8251 | 0,8251 | 0,8251 (0,786–0,859) | 0,8280 | 0,8221 | 0,8407 | 0,8095 |
| MobileNetV3-Large | 0,8081 | 0,8082 | 0,8081 (0,769–0,845) | 0,8068 | 0,8094 | 0,8000 | 0,8163 |
| ResNet50 | 0,8149 | 0,8150 | 0,8149 (0,773–0,856) | 0,8124 | 0,8174 | 0,8000 | 0,8299 |
| **ViT-S/16** | **0,8540** | **0,8540** | **0,8540 (0,813–0,893)** | **0,8562** | **0,8517** | **0,8678** | **0,8401** |

| Arquitectura | Matriz de confusión (filas: real *aegypti* / *albopictus*; columnas: predicción) | Precisión *aegypti* | Precisión *albopictus* |
|---|---|---|---|
| EfficientNet-B0 | [[248, 47], [56, 238]] | 0,8158 | 0,8351 |
| MobileNetV3-Large | [[236, 59], [54, 240]] | 0,8138 | 0,8027 |
| ResNet50 | [[236, 59], [50, 244]] | 0,8252 | 0,8053 |
| ViT-S/16 | [[256, 39], [47, 247]] | 0,8449 | 0,8636 |

En prueba la especificidad de una especie es la sensibilidad de la otra (clasificación binaria). Respecto a validación, el macro F1 de la corrida mediana baja entre 0,012 y 0,034 (ViT-S/16: 0,8745 → 0,8540; ResNet50: 0,8491 → 0,8149), lo que confirma el sesgo optimista previsto en la Entrada 011.

### Parte B — Estabilidad (nivel 1: 5 corridas en prueba)

| Arquitectura | Macro F1 (media ± DE) | Rango | Exactitud (media ± DE) | Sens. *aegypti* (media ± DE) | Macro F1 en control (media ± DE) |
|---|---|---|---|---|---|
| EfficientNet-B0 | 0,8242 ± 0,0276 | 0,7802–0,8540 | 0,8244 ± 0,0272 | 0,8265 ± 0,0322 | 0,8594 ± 0,0444 |
| MobileNetV3-Large | 0,8126 ± 0,0096 | 0,7979–0,8211 | 0,8129 ± 0,0098 | 0,8346 ± 0,0428 | 0,8441 ± 0,0172 |
| ResNet50 | 0,8192 ± 0,0088 | 0,8098–0,8315 | 0,8193 ± 0,0090 | 0,8264 ± 0,0360 | 0,8523 ± 0,0208 |
| ViT-S/16 | 0,8550 ± 0,0047 | 0,8472–0,8591 | 0,8550 ± 0,0047 | 0,8624 ± 0,0183 | 0,8984 ± 0,0102 |

ViT-S/16 es la mejor y la más estable: su peor corrida (0,8472) supera a la mejor de MobileNetV3-Large y de ResNet50. EfficientNet-B0 es la menos estable (DE 0,0276; de 0,7802 a 0,8540), como ya ocurrió en validación. Conforme al plan (3.8), estas cifras describen la variabilidad por inicialización y no se someten a pruebas de significancia.

### Parte C — Subconjunto de control y estratos (protocolo v1.0, apartados 2 y 3)

| Arquitectura | Prueba | Control (Dryad + Ong, 254) | Campo (335) | Dryad (180) | Ong (74) |
|---|---|---|---|---|---|
| EfficientNet-B0 | 0,825 | 0,902 (0,860–0,938) | 0,767 (0,711–0,820) | 0,894 | 0,919 |
| MobileNetV3-Large | 0,808 | 0,850 (0,783–0,907) | 0,776 (0,727–0,825) | 0,866 | 0,810 |
| ResNet50 | 0,815 | 0,842 (0,768–0,910) | 0,793 (0,744–0,842) | 0,872 | 0,769 |
| ViT-S/16 | 0,854 | 0,898 (0,828–0,954) | 0,821 (0,768–0,871) | 0,928 | 0,824 |

| Arquitectura | Δ_control (IC 95 %) | ¿Señal (Δ ≥ 0,05 e IC sin 0)? | Δ_campo = campo − control (IC 95 %) |
|---|---|---|---|
| EfficientNet-B0 | −0,0765 (−0,116 a −0,039) | No | −0,135 (−0,200 a −0,070) |
| MobileNetV3-Large | −0,0423 (−0,087 a +0,004) | No | −0,074 (−0,152 a +0,007) |
| ResNet50 | −0,0273 (−0,077 a +0,021) | No | −0,049 (−0,135 a +0,036) |
| ViT-S/16 | −0,0436 (−0,092 a +0,004) | No | −0,077 (−0,159 a +0,007) |

**Lectura (regla registrada de antemano).** Ninguna arquitectura muestra señal de dependencia del tipo de fotografía. Δ_control es negativo en las cuatro: los modelos rinden **más** en el control, donde el tipo de fotografía no puede delatar la especie, que en el conjunto completo. El rendimiento global no se apoya en reconocer el tipo de fotografía. La dificultad está en las fotografías de campo: ejemplares pequeños, fondos variados y menos resolución útil.

**Análisis de sensibilidad (orden de arquitecturas).**

| Subconjunto | Orden por macro F1 |
|---|---|
| Prueba | ViT-S/16 > EfficientNet-B0 > ResNet50 > MobileNetV3-Large |
| Control | EfficientNet-B0 > ViT-S/16 > MobileNetV3-Large > ResNet50 |

El orden **difiere**. Conforme al protocolo, la conclusión se formula con la salvedad de que el orden depende del tipo de fotografía, y se presentan ambos órdenes. En el control, EfficientNet-B0 (0,902) y ViT-S/16 (0,898) quedan prácticamente igualadas. La jerarquía de selección aplicada sobre el control también elige ViT-S/16 (Entrada 015), así que la salvedad afecta al orden de las demás, no a la mejor arquitectura.

### Parte D — Señales de atajo por plataforma (protocolo v1.0, apartado 4)

| Indicador | EfficientNet-B0 | MobileNetV3-Large | ResNet50 | ViT-S/16 |
|---|---|---|---|---|
| M-ae: *Ae. aegypti* de Mosquito Alert bien clasificados (8 imágenes, 4 reportes) | 4/8 | 6/8 | 6/8 | 6/8 |
| I-ae: sensibilidad *aegypti* en iNaturalist (160) | 0,806 | 0,769 | 0,731 | 0,831 |
| M-al: sensibilidad *albopictus* en Mosquito Alert (167) | 0,743 | 0,784 | 0,856 | 0,814 |

En ninguna arquitectura la mayoría de M-ae se clasifica como *Ae. albopictus*, de modo que no hay el patrón consistente con un atajo "plataforma → especie". EfficientNet-B0 queda en el límite (4 de 8). Con 8 imágenes de 4 reportes, el indicador es descriptivo. La lectura del conjunto ecuatoriano está en la Entrada 013.

### Parte E — Grad-CAM (protocolo v1.0, apartado 5)

64 imágenes de prueba (43 informativas), con las cajas trazadas antes de entrenar.

| Arquitectura | Picos fuera de la caja (total) | Mediana de concentración | Estratos con alerta (≥ 8 informativas) | Alertas |
|---|---|---|---|---|
| EfficientNet-B0 | 11/43 (25,6 %) | 1,451 | M-al (70 %) | Fondo (total, por 0,6 puntos) |
| MobileNetV3-Large | 15/43 (34,9 %) | 1,538 | M-al (30 %), O-ae (50 %), O-al (50 %) | Fondo |
| ResNet50 | 12/43 (27,9 %) | 2,171 | M-al (60 %) | Fondo |
| ViT-S/16 | 28/43 (65,1 %) | 0,934 | I-ae (88 %), M-al (70 %), O-ae (70 %), O-al (50 %) | Fondo y atención no localizada |

Picos fuera por estrato (informativas): I-ae 8, M-al 10, O-ae 10, O-al 10; Dryad y M-ae tienen menos de 8 y se describen sin criterio de alerta.

| Estrato | EfficientNet-B0 | MobileNetV3-Large | ResNet50 | ViT-S/16 |
|---|---|---|---|---|
| I-ae | 2/8 | 1/8 | 2/8 | 7/8 |
| M-al | 7/10 | 3/10 | 6/10 | 7/10 |
| O-ae | 1/10 | 5/10 | 2/10 | 7/10 |
| O-al | 1/10 | 5/10 | 1/10 | 5/10 |

**Lectura.** Las cuatro arquitecturas superan el umbral de alerta de fondo. La alerta común a todas está en **M-al**, las fotografías de *Ae. albopictus* de Mosquito Alert. En la lámina se ve que, en esas fotografías, el espécimen es diminuto y la caja muy pequeña. El mapa de las CNN tiene una resolución de 7 × 7 sobre la imagen de 224 px, de modo que el pico cae a menudo junto al mosquito pero fuera de su caja. En otras imágenes cae sobre la mano o el borde de la fotografía. La concentración sigue por encima de 1 en las tres CNN, es decir, la activación se acumula sobre el espécimen más de lo que correspondería por azar.

ViT-S/16 repite el patrón del piloto (Entrada 009): mapas en manchas repartidas por toda la imagen y concentración mediana por debajo de 1. Su Grad-CAM no es equivalente al de las CNN (protocolo, 5.3), y su rendimiento en el control (0,898) no muestra dependencia del tipo de fotografía. Aun así, se registra como limitación: con este procedimiento no puede afirmarse que ViT-S/16 base su decisión en el espécimen.

Conforme al protocolo, una alerta de Grad-CAM por sí sola no invalida un resultado. Como no hay señal de dependencia en el control (Parte C), la alerta **no refuerza** ninguna limitación del apartado 3. Se reporta como limitación de interpretabilidad, sobre todo en fotografías de campo con ejemplares pequeños.

### Parte F — Eficiencia computacional (anexo, 5)

Medida en un proceso aparte por arquitectura, con 6 hilos y el *checkpoint* de la corrida mediana.

| Arquitectura | Parámetros | Tamaño | ms/imagen (lote 1, mediana de 100) | ms/imagen (lote 64) | RAM residente tras inferencia | RAM del modelo e inferencia |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | 4,01 M | 15,57 MB | 17,61 | 18,13 | 500 MB | 298 MB |
| MobileNetV3-Large | 4,20 M | 16,23 MB | 12,48 | 10,97 | 525 MB | 322 MB |
| ResNet50 | 23,51 M | 89,99 MB | 50,43 | 48,40 | 591 MB | 388 MB |
| ViT-S/16 | 21,67 M | 82,70 MB | 43,74 | 38,41 | 624 MB | 422 MB |

La RAM base del proceso (Python, PyTorch y timm, sin modelo) es de 203 MB en los cuatro casos. ViT-S/16, la de mejor rendimiento, es unas 3,5 veces más lenta que MobileNetV3-Large y 2,5 veces más lenta que EfficientNet-B0. Aun así, 44 ms por imagen en CPU sigue siendo viable para uso interactivo. La medida de RAM corrige la de la Fase 5 (incidencia del 2026-11-24).

### Figuras

`fase6/figuras/`: curvas de pérdida y macro F1 por época de las 20 corridas (`curvas_<arquitectura>.png`), macro F1 en prueba, control y campo con IC (`macro_f1_subconjuntos.png`) y matrices de confusión (`matrices_confusion.png`). Láminas de Grad-CAM por estrato en `fase6/gradcam_laminas/`.

---

## Entrada 013 — Validación contextual (conjunto ecuatoriano)

**Fecha:** 2026-11-25
**Script:** `fase6/evaluar.py` (misma ejecución que la Entrada 012)
**Salidas:** `fase6/ecuador.json`, `fase6/ecuador_casos.csv`, `fase6/atajo_plataforma.csv`
**Datos:** `fase6/Dataset_Dengue_Aedes_Ecuador/` (127 imágenes de iNaturalist en Ecuador, `metadata.csv`)

**Comprobación de fuga:** 0 coincidencias de `observation_id` o `photo_id` con las 1.100 fotos de iNaturalist del entrenamiento.

### *Aedes aegypti* — 71 imágenes con identificación confirmada (19 observaciones)

| Arquitectura | Sensibilidad (71) (IC 95 % por observación) | Exactitud (75) | F1 *aegypti* (75) | Prueba: sens. *aegypti* global | Prueba: sens. I-ae |
|---|---|---|---|---|---|
| EfficientNet-B0 | 0,648 (0,493–0,878) | 0,640 | 0,773 | 0,841 | 0,806 |
| MobileNetV3-Large | 0,775 (0,488–0,907) | 0,773 | 0,866 | 0,800 | 0,769 |
| ResNet50 | 0,817 (0,521–0,939) | 0,827 | 0,899 | 0,800 | 0,731 |
| ViT-S/16 | 0,803 (0,512–0,935) | 0,813 | 0,891 | 0,868 | 0,831 |

En las 71 imágenes de *Ae. aegypti*, la exactitud coincide con la sensibilidad. La exactitud y el F1 se calculan sobre las 75 imágenes `research`, con las 4 *Ae. albopictus* como posibles falsos positivos (anexo, 6.2).

**Análisis secundarios fijados de antemano (anexo, 6.3):**

| Arquitectura | Por observación (media; obs. con mayoría de aciertos) | Sin la obs. 156329843 (39 imágenes) | Obs. 156329843 (32 vistas parciales) | 119 imágenes | Sens. en las 5 corridas (media ± DE) |
|---|---|---|---|---|---|
| EfficientNet-B0 | 0,768; 16 de 19 | 0,692 | 19/32 | 0,706 | 0,713 ± 0,042 |
| MobileNetV3-Large | 0,800; 16 de 19 | 0,692 | 28/32 | 0,765 | 0,766 ± 0,061 |
| ResNet50 | 0,742; 15 de 19 | 0,692 | 31/32 | 0,756 | 0,811 ± 0,064 |
| ViT-S/16 | 0,783; 15 de 19 | 0,692 | 30/32 | 0,756 | 0,809 ± 0,016 |

### *Aedes albopictus* — caso por caso (plan 3.8)

| Observación | Calidad | Lugar | EfficientNet-B0 | MobileNetV3-Large | ResNet50 | ViT-S/16 |
|---|---|---|---|---|---|---|
| 284272667 | research | Guayaquil | ✗ (*aegypti*) | ✗ (*aegypti*) | ✓ | ✓ |
| 288038488 | research | Guayaquil | ✓ | ✓ | ✓ | ✓ |
| 329126060 | research | Guayaquil | ✓ | ✓ | ✓ | ✓ |
| 337363912 | research | Guayaquil | ✗ (*aegypti*) | ✓ | ✓ | ✓ |
| 64071367 (foto 1) | needs_id | Urb. Laguna Club | ✓ | ✓ | ✓ | ✓ |
| 64071367 (foto 2) | needs_id | Urb. Laguna Club | ✓ | ✓ | ✓ | ✗ (*aegypti*) |
| 271247070 | needs_id | Vía Lago Agrio | ✓ | ✓ | ✗ (*aegypti*) | ✓ |
| 344733666 | needs_id | Sucumbíos | ✗ (*aegypti*) | ✓ | ✓ | ✗ (*aegypti*) |
| **Aciertos `research` / todas** | | | **2/4 · 5/8** | **3/4 · 7/8** | **4/4 · 7/8** | **4/4 · 6/8** |

**Lectura (exploratoria, plan 3.8).**

1. **Rendimiento comparable al de la prueba, con mucha incertidumbre.** ResNet50, ViT-S/16 y MobileNetV3-Large mantienen en Ecuador una sensibilidad similar a la de la prueba internacional (0,78–0,82 frente a 0,80–0,87). EfficientNet-B0 cae más (0,648 frente a 0,841). Los IC por observación son muy amplios (unos 0,4 de ancho), porque son solo 19 observaciones, y se solapan entre las cuatro arquitecturas.
2. **El peso de la observación 156329843 cambia la lectura.** Sin ella, las cuatro arquitecturas aciertan exactamente 27 de las 39 imágenes restantes (0,692), aunque no las mismas. Las vistas parciales al microscopio son las que más separan a las arquitecturas, y favorecen a ResNet50 y ViT-S/16 (31 y 30 de 32). En las fotografías de campo ecuatorianas, el rendimiento ronda el 0,69 en las cuatro, por debajo de la prueba.
3. **Errores sistemáticos.** La observación 31308558 (6 fotografías) falla con las cuatro arquitecturas.
4. **Atajo por plataforma (protocolo, 4).** Las *Ae. albopictus* ecuatorianas clasificadas como *Ae. aegypti* son consistentes con el atajo. EfficientNet-B0 lo hace en 2 de 4 `research` y en 3 de 8 en total. ResNet50 y ViT-S/16 aciertan las 4 `research`. El indicador es descriptivo: son 8 imágenes.
5. **ViT-S/16 es la más estable** entre corridas en Ecuador (DE 0,016 frente a 0,042–0,064).

Los resultados se presentan con carácter exploratorio, no como evidencia concluyente de generalización (plan 3.8).

---

# FASE 7 — Análisis, recomendaciones y redacción

_(Entradas a partir de S19)_

## Entrada 014 — Pruebas estadísticas

**Fecha:** 2026-11-25 (calculadas por `fase6/evaluar.py`; salida en `fase6/pruebas_estadisticas.json`)
**Datos:** acierto o error de la corrida mediana de cada arquitectura en las 589 imágenes de prueba (predicciones pareadas)

| Prueba | Estadístico | Valor p | Tamaño del efecto |
|---|---|---|---|
| Q de Cochran (global) | Q = 8,315 (gl = 3) | **0,0399** | W de Cochran = 0,0047 |
| McNemar — EfficientNet-B0 vs MobileNetV3-Large | b = 62, c = 52 | 0,399 (Bonf. 1,000) | +0,017 (−0,019 a +0,053) |
| McNemar — EfficientNet-B0 vs ResNet50 | b = 62, c = 56 | 0,646 (Bonf. 1,000) | +0,010 (−0,026 a +0,046) |
| McNemar — EfficientNet-B0 vs ViT-S/16 | b = 39, c = 56 | 0,100 (Bonf. 0,601) | −0,029 (−0,061 a +0,004) |
| McNemar — MobileNetV3-Large vs ResNet50 | b = 53, c = 57 | 0,775 (Bonf. 1,000) | −0,007 (−0,042 a +0,028) |
| McNemar — MobileNetV3-Large vs ViT-S/16 | b = 35, c = 62 | 0,0080 (**Bonf. 0,0478**) | −0,046 (−0,078 a −0,013) |
| McNemar — ResNet50 vs ViT-S/16 | b = 28, c = 51 | 0,0128 (Bonf. 0,0769) | −0,039 (−0,069 a −0,010) |

**Corrección aplicada:** Bonferroni (× 6)
**Nivel de significancia:** p < 0,05
b: imágenes que acierta solo la primera arquitectura; c: solo la segunda. Tamaño del efecto por pares: diferencia de exactitud (primera − segunda) con IC 95 % de Wald para proporciones pareadas. McNemar exacta (binomial).

**Análisis de sensibilidad (IC por *bootstrap* de grupos, anexo 2):** MobileNetV3-Large vs ViT-S/16 −0,046 (−0,081 a −0,010); ResNet50 vs ViT-S/16 −0,039 (−0,070 a −0,008); EfficientNet-B0 vs ViT-S/16 −0,029 (−0,067 a +0,011). Coincide con la lectura de Wald: tener en cuenta que varias imágenes son del mismo espécimen no cambia ninguna conclusión.

**Interpretación.**

1. La Q de Cochran es significativa (p = 0,040): las cuatro arquitecturas no tienen la misma proporción de aciertos. El tamaño del efecto global es muy pequeño (W = 0,005): las arquitecturas coinciden en la gran mayoría de las imágenes y difieren en pocas.
2. Tras la corrección de Bonferroni, solo es significativa la diferencia entre **ViT-S/16 y MobileNetV3-Large**: ViT-S/16 acierta 4,6 puntos porcentuales más (IC 1,3 a 7,8).
3. ViT-S/16 frente a ResNet50 (+3,9 puntos) es significativa sin corregir (p = 0,013), pero no tras Bonferroni (p = 0,077). Frente a EfficientNet-B0 (+2,9 puntos), la diferencia no es significativa.
4. Entre las tres CNN no hay diferencias significativas (diferencias de 0,7 a 1,7 puntos).
5. Validez: las conclusiones se refieren a este conjunto de prueba y a una partición fija (plan 3.8).

## Entrada 015 — Selección del mejor modelo

**Fecha:** 2026-11-25 (`fase6/seleccion_mejor_modelo.json`)
**Reglas:** plan 3.8 y anexo, 4 (empate si la diferencia es menor de 0,02)

| Nivel de la jerarquía | Criterio | Resultado en prueba | Resultado en control (sensibilidad) |
|---|---|---|---|
| 1.º | Macro F1 | ViT-S/16 0,8540; la siguiente, EfficientNet-B0, 0,8251 (dif. 0,029 ≥ 0,02): **sin empate** | Empate: EfficientNet-B0 0,9016 y ViT-S/16 0,8976 |
| 2.º | Sensibilidad en *Ae. aegypti* (si dif. < 0,02) | No se aplica | Empate: ViT-S/16 0,9213 y EfficientNet-B0 0,9055 (dif. 0,016) |
| 3.º | Estabilidad (DE del macro F1, 5 corridas) | No se aplica | ViT-S/16 0,0102 frente a EfficientNet-B0 0,0444: **ViT-S/16** |
| 4.º | Eficiencia computacional | No se aplica | No se aplica |

**Arquitectura seleccionada:** **ViT-S/16** (corrida mediana, semilla 46)

**Justificación:**

- **Rendimiento.** Obtiene el mayor macro F1 en prueba (0,854; IC 0,813–0,893) y la mayor sensibilidad en *Ae. aegypti* (0,868), el vector primario del dengue en el Ecuador.
- **Estabilidad.** Es la más estable entre corridas (DE 0,0047).
- **Robustez de la selección.** La selección se mantiene en el análisis de sensibilidad sobre el control, aunque allí el orden de las demás cambia.
- **Significancia.** Su ventaja es estadísticamente significativa frente a MobileNetV3-Large. Frente a ResNet50 y EfficientNet-B0 no lo es tras la corrección, de modo que la superioridad se afirma por la jerarquía de selección, no por significancia frente a todas.

**Salvedades:**

1. **Interpretabilidad.** Es la arquitectura con peor resultado en Grad-CAM (alerta de fondo y atención no localizada; Entrada 012, Parte E).
2. **Coste computacional.** Es 2,5 a 3,5 veces más lenta en inferencia que las arquitecturas ligeras. Para entornos con recursos muy limitados, MobileNetV3-Large (12,5 ms por imagen, 16 MB) es la alternativa más eficiente, con 4,6 puntos menos de exactitud.

---

## Entrada 016 — Redacción de resultados, discusión y conclusiones (borrador)

**Fecha:** 2026-11-25
**Documento:** `Batioja_Leandro_Articulo_completo.docx` (a partir de `Batioja_Leandro_Tercer_Avance_Articulo.docx`)

Se añadieron al artículo, a partir de las Entradas 011 a 015:

| Sección | Contenido |
|---|---|
| Abstract, Resumen y palabras clave | Unas 200 palabras cada uno, con los resultados principales |
| 3. Resultados | Ocho subsecciones; Tablas 6 a 13 y Figuras 2 a 5 (matrices de confusión, curvas por época, macro F1 por subconjunto y láminas de Grad-CAM) |
| 4. Discusión | Matriz comparativa de resultados (Tabla 14), comparación con la literatura citada, control y aprendizaje por atajos, salvedad del orden, Grad-CAM, conjunto ecuatoriano y limitaciones |
| 5. Conclusiones | Respuesta a la pregunta de investigación, recomendaciones técnicas y trabajo futuro |
| Referencias | Se añade [35] (statsmodels) |

El texto previo (Introducción y Materiales y métodos) no se modificó: sus correcciones y la actualización de tiempos verbales se harán junto con las observaciones del tutor. El investigador trasladará el contenido a la plantilla de la revista.

---

## Anexo — Inventario de archivos generados

> Todo archivo nuevo o modificado se registra aquí en el momento de su creación.

| Fecha | Fase | Archivo | Tipo | Propósito |
|---|---|---|---|---|
| 2026-09-19 | F2 | `bitacora_experimentos.md` | Instrumento 3.6.2.2 | Registro cronológico de todas las fases |
| 2026-09-19 | F2 | `fase2/requirements.txt` | Configuración | Dependencias con versiones fijadas |
| 2026-09-19 | F2 | `fase2/verificar_entorno.py` | Script | Verificación del entorno y las arquitecturas |
| 2026-09-19 | F2 | `fase2/benchmark_cpu.py` | Script | Medición de rendimiento y extrapolación de horas |
| 2026-09-19 | F2 | `fase2/README_FASE2.md` | Documentación | Instrucciones de uso de la Fase 2 |
| 2026-09-19 | F2 | `fase2/ficha_tecnica.md` | Instrumento 3.6.1.3 | Configuración experimental documentada |
| 2026-09-19 | F2 | `fase2/salidas/log_verificacion.txt` | Evidencia | Salida completa de la verificación |
| 2026-09-19 | F2 | `fase2/salidas/log_benchmark.txt` | Evidencia | Salida completa del benchmark |
| 2026-09-19 | F2 | `fase2/salidas/reporte_entorno.json` | Datos | Entorno y arquitecturas, estructurado |
| 2026-09-19 | F2 | `fase2/salidas/reporte_benchmark.json` | Datos | Rendimiento medido, estructurado |
| 2026-09-19 | F2 | `fase2/criterios_depuracion.md` | Protocolo | Criterios de calidad, etiquetado y selección |
| 2026-09-19 | F2 | `fase3/analizar_nitidez.py` | Script | Determinación empírica del umbral de desenfoque. Sobrescrita por la v2 el 2026-09-20; no se conserva |
| 2026-09-20 | F3 | `fase3/consolidar_conjunto.py` | Script | Consolidación de los cinco repositorios con trazabilidad de origen y espécimen |
| 2026-09-20 | F3 | `fase3/procedencia_conjunto_bruto.csv` | Instrumento 3.6.1.2 | Procedencia de cada imagen: origen, especie y espécimen |
| 2026-09-20 | F3 | `fase3/reporte_consolidacion.json` | Datos | Tabla cruzada especie × repositorio y verificación M1 |
| 2026-09-20 | F3 | `fase3/salidas/log_consolidacion_sim.txt` | Evidencia | Salida de la simulación previa a la consolidación |
| 2026-09-20 | F3 | `fase3/salidas/log_consolidacion.txt` | Evidencia | Salida completa de la consolidación |
| 2026-09-20 | F3 | `fase3/conjunto_bruto/` | Datos | 16.969 enlaces duros organizados por especie |
| 2026-09-20 | F3 | `fase3/analizar_nitidez.py` (v2) | Script | Desglose por repositorio, normalización a 512 px y muestra visual estratificada |
| 2026-09-20 | F3 | `fase3/salidas/log_nitidez.txt` | Evidencia | Salida completa del análisis de nitidez |
| 2026-09-20 | F3 | `fase3/analisis_nitidez/metricas_por_imagen.csv` | Datos | Varianza, ancho, alto y brillo de cada imagen |
| 2026-09-20 | F3 | `fase3/analisis_nitidez/reporte_nitidez.json` | Datos | Distribuciones, simulación de umbrales y techo de contención |
| 2026-09-20 | F3 | `fase3/analisis_nitidez/muestra_visual/` | Evidencia | Muestra estratificada por repositorio y percentil |
| 2026-09-20 | F3 | `fase3/analisis_nitidez/revision/` | Evidencia | Ocho imágenes de valores bajos inspeccionadas |
| 2026-09-20 | F3 | `fase3/analisis_nitidez/franja_5_10/` | Evidencia | Doce imágenes de la franja que elimina el umbral |
| 2026-09-24 | F3 | `fase3/depurar_conjunto.py` | Script | Aplicación de los criterios E1 a E9 con verificación de duplicados |
| 2026-09-24 | F3 | `fase3/salidas/log_depuracion_sim.txt` | Evidencia | Primera simulación, con falsos positivos en E4 y E8 |
| 2026-09-24 | F3 | `fase3/salidas/log_depuracion_sim2.txt` | Evidencia | Segunda simulación, tras corregir E4 y E8 |
| 2026-09-24 | F3 | `fase3/salidas/log_depuracion_sim3.txt` | Evidencia | Tercera simulación, tras reoperacionalizar E5 |
| 2026-09-24 | F3 | `fase3/salidas/log_depuracion.txt` | Evidencia | Ejecución definitiva con escritura del conjunto depurado |
| 2026-09-24 | F3 | `fase3/depuracion/registro_imagenes_eliminadas.csv` | Instrumento 3.6.2.1 | Registro de cada imagen eliminada con su criterio |
| 2026-09-24 | F3 | `fase3/depuracion/metricas_depuracion.csv` | Datos | MD5, pHash, brillo, rango dinámico y varianza de cada imagen |
| 2026-09-24 | F3 | `fase3/depuracion/reporte_depuracion.json` | Datos | Conteos por criterio, duplicados y tabla especie × repositorio |
| 2026-09-24 | F3 | `fase3/verificacion_dup/` | Evidencia | Pares señalados como duplicados, inspeccionados visualmente |
| 2026-09-24 | F3 | `fase3/verificacion_e5/` | Evidencia | Seis imágenes rechazadas por exposición, inspeccionadas |
| 2026-09-24 | F3 | `fase3/conjunto_depurado/` | Datos | 16.037 enlaces duros organizados por especie |
| 2026-09-20/24 ¹ | F3 | `fase2/criterios_depuracion.md` (v2) | Protocolo | Umbral de desenfoque fijado en 10; E1 sin tolerancia a truncamiento; revisión íntegra de Couret. Sobrescrita por la v3 |
| 2026-09-24 | F3 | `fase2/criterios_depuracion.md` (v3) | Protocolo | Reoperacionalización de E4 y E5; corrección de la nota sobre Couret |
| 2026-09-20 | F2 | `fase2/requirements.txt` (modificado, 09:03) | Configuración | Modificación al cierre de F2; el contenido del cambio no quedó registrado |
| 2026-09-20 | F2 | `fase2/ficha_tecnica.md` (modificado, 09:08) | Instrumento 3.6.1.3 | Modificación al cierre de F2; el contenido del cambio no quedó registrado |
| 2026-09-27 | F3 | `fase3/auditar_estructura.py` | Script | Contraste del inventario con el disco y regeneración del árbol |
| 2026-09-27 | F3 | `auditoria_estructura.txt` | Evidencia | Salida de la auditoría de estructura (Entrada 006 bis) |
| 2026-09-27 | F3 | `estructura_proyecto.txt` (regenerado) | Documentación | Árbol del proyecto con las carpetas de datos colapsadas y su conteo |
| ≤ 2026-09-27 | F3 | `fase3/analisis_nitidez/franja_5_10.zip` | Empaquetado | Copia comprimida de la carpeta homónima, para envío |
| ≤ 2026-09-27 | F3 | `fase3/analisis_nitidez/revision.zip` | Empaquetado | Copia comprimida de la carpeta homónima, para envío |
| ≤ 2026-09-27 | F3 | `fase3/verificacion_dup.zip` | Empaquetado | Copia comprimida de la carpeta homónima, para envío |
| ≤ 2026-09-27 | F3 | `fase3/verificacion_e5.zip` | Empaquetado | Copia comprimida de la carpeta homónima, para envío |
| 2026-10-01 | F3 | `descargar_bibliografia.py` | Script | Verificación de DOI, descarga y nombrado de la bibliografía; genera el índice |
| 2026-10-01 | F3 | `bibliografia/` | Bibliografía | 39 PDF nombrados por número de cita y título, `indice_bibliografia.md` y `verificacion_doi.md` |
| 2026-10-01 | F3 | `arbol_bibliografia.txt` | Documentación | Árbol de la carpeta `bibliografia/` |
| 2026-10-01 | F3 | `bibliografia.zip` | Empaquetado | Copia comprimida de `bibliografia/`, para envío |
| 2026-10-23 | F3 | `fase3/agrupar_especimenes.py` (v4) | Script | Agrupación por espécimen: reporte en Mosquito Alert, similitud y revisión manual en Ong. Versiones 1 a 3 sobrescritas |
| 2026-10-23 | F3 | `fase3/agrupacion/mosquito_alert_photos_clean.csv` | Metadatos de la fuente | Relación foto–reporte de Mosquito Alert, descargada del servidor público |
| 2026-10-23 | F3 | `fase3/agrupacion/especimenes.csv` | Datos | Grupo de cada imagen del conjunto depurado y si su reporte es verificable |
| 2026-10-23 | F3 | `fase3/agrupacion/reporte_agrupacion.json` | Evidencia | Resumen numérico y calibración de Mosquito Alert |
| 2026-10-23 | F3 | `fase3/agrupacion/ong_uniones_manuales.csv` | Evidencia | 149 uniones verificadas visualmente, con su motivo |
| 2026-10-23 | F3 | `fase3/agrupacion/ong_grupos.txt` | Evidencia | Grupos de Ong con los números de sus fotos |
| 2026-10-23 | F3 | `fase3/agrupacion/ong_grupos/` | Evidencia | Lámina de cada grupo de Ong, con el recorte comparado |
| 2026-10-23 | F3 | `fase3/agrupacion/ong_casi/` | Evidencia | Pares más parecidos que quedaron separados (última ronda) |
| 2026-10-23 | F3 | `fase3/agrupacion/ong_contacto/` | Evidencia | Hojas de contacto de Ong con el grupo de cada foto |
| 2026-10-23 | F3 | `fase3/agrupacion/ma_enlaces/` | Evidencia | Vacía: la unión por similitud en Mosquito Alert no se aplicó |
| 2026-10-23 | F3 | `fase3/agrupacion/cache_miniaturas_ma.npz` | Caché | Miniaturas de Mosquito Alert; regenerable, se excluye de la entrega |
| 2026-10-23 | F3 | `fase3/descargar_inaturalist.py` | Script | Descarga de *Ae. aegypti* de iNaturalist con criterios fijados y procedencia por foto |
| 2026-10-23 | F3 | `fase3/inaturalist/procedencia_inaturalist.csv` | Instrumento 3.6.1.2 | Observación, foto, autor, licencia, lugar y URL de cada imagen de iNaturalist |
| 2026-10-23 | F3 | `fase3/inaturalist/resumen_inaturalist.json` | Evidencia | Criterios, conteos y exclusiones de la descarga |
| 2026-10-23 | F3 | `fase3/consolidar_conjunto.py` (v2) | Script | Sexto repositorio (iNaturalist), fotografías `Phone_` de Dryad y sufijo de salida. Sobrescribió la v1 en la misma ruta |
| 2026-10-24 | F3 | `fase3/conjunto_bruto_v2/` | Datos | 18.467 enlaces duros organizados por especie |
| 2026-10-24 | F3 | `fase3/procedencia_conjunto_bruto_v2.csv` | Instrumento 3.6.1.2 | Procedencia de cada imagen del conjunto bruto v2 |
| 2026-10-24 | F3 | `fase3/reporte_consolidacion_v2.json` | Datos | Tabla cruzada especie × repositorio del conjunto bruto v2 |
| 2026-10-24 | F3 | `fase3/salidas/log_consolidacion_v2.txt` | Evidencia | Salida de la consolidación v2 |
| 2026-10-24 | F3 | `fase3/conjunto_depurado_v2/` | Datos | 17.527 enlaces duros organizados por especie |
| 2026-10-24 | F3 | `fase3/depuracion_v2/registro_imagenes_eliminadas.csv` | Instrumento 3.6.2.1 | Registro de cada imagen eliminada en la depuración v2 |
| 2026-10-24 | F3 | `fase3/depuracion_v2/metricas_depuracion.csv` | Datos | Métricas de cada imagen del conjunto bruto v2 |
| 2026-10-24 | F3 | `fase3/depuracion_v2/reporte_depuracion.json` | Datos | Conteos por criterio y tabla especie × repositorio v2 |
| 2026-10-24 | F3 | `fase3/salidas/log_depuracion_v2.txt` | Evidencia | Salida de la depuración v2 |
| 2026-10-23/24 | F3 | `fase3/verificacion_morfologica.py` (v3) | Script | Conjunto de trabajo, muestra a ciegas, exportación de la muestra y evaluación. Versiones 1 y 2 (láminas sin y con acercamiento) sobrescritas |
| 2026-10-23 | F3 | `fase3/conjunto_trabajo.csv` | Datos | Conjunto de trabajo de la opción A: especie, imagen, origen y grupo |
| 2026-10-23 | F3 | `fase3/verificacion_morfologica/preparacion.json` | Evidencia | Composición del conjunto de trabajo y reparto de la muestra |
| 2026-10-23 | F3 | `fase3/verificacion_morfologica/clave_verificacion.csv` | Datos | Etiqueta de cada número de la muestra (oculta durante la revisión) |
| 2026-10-23 | F3 | `fase3/verificacion_morfologica/laminas/` | Evidencia | Láminas v2 con recorte automático; descartadas para la revisión |
| 2026-10-24 | F3 | `fase3/verificacion_morfologica/muestra_ciega/` | Evidencia | Las 400 imágenes a 1.600 px, nombradas solo con su número |
| 2026-10-24 | F3 | `fase3/verificacion_morfologica/verificacion_morfologica.csv` | Instrumento | Veredicto, rasgo y nota de cada imagen, confirmado por el investigador |
| 2026-10-24 | F3 | `fase3/verificacion_morfologica/revision/` | Evidencia | Documentos de confirmación (dudosas, `otro` y con especie) |
| 2026-10-24 | F3 | `fase3/verificacion_morfologica/resultado_verificacion.json` | Evidencia | Resultado original de la evaluación |
| 2026-10-24 | F3 | `fase3/verificacion_morfologica/discrepancias/` | Evidencia | Láminas de las 65 discrepancias originales |
| 2026-10-24 | F3 | `fase3/corregir_verificacion.py` | Script | Reclasificación de laboratorio, exclusiones y filtro de estadio vital en iNaturalist |
| 2026-10-24 | F3 | `fase3/verificacion_morfologica/verificacion_morfologica_corregida.csv` | Datos | Veredictos tras la reclasificación de laboratorio |
| 2026-10-24 | F3 | `fase3/verificacion_morfologica/resultado_verificacion_corregido.json` | Evidencia | Resultado original y corregido, discrepancias y estadio vital |
| 2026-10-24 | F3 | `fase3/verificacion_morfologica/exclusiones_007.csv` | Datos | 30 imágenes excluidas del conjunto de trabajo, con su motivo |
| 2026-10-24 | F3 | `fase3/salidas/log_correccion_007.txt` | Evidencia | Salida de `corregir_verificacion.py` |
| 2026-10-24 | F3 | `fase3/revisar_estadio_inat.py` | Script | Hojas de miniaturas de iNaturalist sin anotación de estadio y aplicación de las exclusiones |
| 2026-10-24 | F3 | `fase3/verificacion_morfologica/estadio_inat/` | Evidencia | 14 hojas, índice, `no_adultos.txt` (vacío) y `resultado_estadio.json` |
| 2026-10-24 | F3 | `fase3/salidas/log_estadio_inat.txt` | Evidencia | Salida de la aplicación: 652 revisadas, 0 no adultos |
| 2026-10-24 | F3 | `fase3/particion_conjunto.py` | Script | Rebalanceo, partición agrupada y estratificada, y muestras de Grad-CAM y del piloto |
| 2026-10-24 | F3 | `fase3/particion/particion.csv` | Datos | Subconjunto y parte (control o campo) de cada una de las 3.916 imágenes |
| 2026-10-24 | F3 | `fase3/particion/rebalanceo_retiradas.csv` | Datos | 17 reportes de Mosquito Alert retirados al igualar |
| 2026-10-24 | F3 | `fase3/particion/gradcam_muestra.csv` | Datos | 64 imágenes de prueba para Grad-CAM |
| 2026-10-24 | F3 | `fase3/particion/piloto_entrenamiento.csv` | Datos | 400 imágenes del piloto |
| 2026-10-24 | F3 | `fase3/particion/piloto_gradcam.csv` | Datos | 20 imágenes de validación para Grad-CAM del piloto |
| 2026-10-24 | F3 | `fase3/particion/reporte_particion.json` | Evidencia | Conteos, verificaciones y M1 por subconjunto |
| 2026-10-24 | F3 | `fase3/conjunto_final/` | Datos | 3.916 enlaces duros en entrenamiento, validación y prueba, por especie |
| 2026-10-24 | F3 | `fase3/salidas/log_particion.txt` | Evidencia | Salida de la simulación posterior, con números idénticos a los de la ejecución (misma semilla) |
| 2026-10-24 | F3 | `fase3/anotar_cajas.py` | Script | Trazado de cajas del espécimen y hojas de control |
| 2026-10-24 | F3 | `fase3/particion/gradcam_cajas.csv` | Datos | 64 cajas trazadas antes de entrenar |
| 2026-10-24 | F3 | `fase3/particion/piloto_cajas.csv` | Datos | 20 cajas del piloto |
| 2026-10-24 | F3 | `fase3/particion/cajas_laminas/` | Evidencia | Hojas de control con las cajas dibujadas |
| 2026-10-24/27 | F4 | `fase4/entrenar.py` | Script | Entrenamiento, validación, Grad-CAM y criterios del piloto; modo de curva de aprendizaje. Se reutiliza en la Fase 5 |
| 2026-10-27 | F4 | `fase4/cache_512/` | Caché | Copias reducidas (≤ 512 px) con la orientación EXIF aplicada; regenerable |
| 2026-10-27 | F4 | `fase4/log_piloto.txt` | Evidencia | Salida completa del piloto |
| 2026-10-27 | F4 | `fase4/piloto/` | Datos y evidencia | Reporte, curvas, métricas, predicciones, *checkpoints* y Grad-CAM por arquitectura |
| 2026-11-16 | F4 | `fase4/log_curva.txt` | Evidencia | Salida de la curva de aprendizaje |
| 2026-11-16 | F4 | `fase4/curva/` | Datos y evidencia | Reporte, curvas y *checkpoints* de las cuatro fracciones |
| 2026-11-17 | F5 | `fase5/log_busqueda.txt` | Evidencia | Salida de la búsqueda de hiperparámetros |
| 2026-11-17 | F5 | `fase5/busqueda/` | Datos y evidencia | Reporte, configuración seleccionada y curvas de las 12 configuraciones |
| 2026-11-22 | F5 | `fase5/log_definitivas.txt` | Evidencia | Salida de las 20 corridas definitivas |
| 2026-11-22 | F5 | `fase5/definitivas/` | Datos y evidencia | `resumen_definitivas.json` y, por arquitectura y semilla, *checkpoint*, curvas, métricas y predicciones de validación |
| 2026-11-24 | F5 | `fase2/ficha_tecnica.md` (actualizada) | Instrumento 3.6.1.3 | Rendimiento de las corridas definitivas y alcance de la medida de RAM |
| 2026-11-24 | F6 | `fase2/protocolo_evaluacion_anexo_fase6.md` (v1.0) | Protocolo | Detalles de aplicación de la Fase 6, registrados antes de evaluar (SHA-256 en la Entrada 011 bis) |
| 2026-11-24 | F6 | `fase6/evaluar.py` | Script | Evaluación en prueba (niveles 1 y 2), control, pruebas estadísticas, jerarquía de selección, Grad-CAM, conjunto ecuatoriano, eficiencia y figuras |
| 2026-11-24 | F6 | `fase6/Dataset_Dengue_Aedes_Ecuador/` | Datos | Conjunto de validación contextual: 127 imágenes de iNaturalist en Ecuador, con `metadata.csv` y `README.md` |
| 2026-11-25 | F6 | `fase6/evaluar.py` (corregido) | Script | Identificación de cada imagen por especie y nombre (incidencia del 2026-11-25). Sobrescribió la versión del 2026-11-24 |
| 2026-11-25 | F6 | `fase6/log_evaluacion.txt` | Evidencia | Salida completa de la evaluación |
| 2026-11-25 | F6 | `fase6/predicciones/` | Datos | Predicción y probabilidad por imagen de los 20 modelos en prueba y en el conjunto ecuatoriano |
| 2026-11-25 | F6 | `fase6/reporte_fase6.json`, `metricas_prueba.json`, `metricas_por_estrato.csv`, `estabilidad_5_corridas.csv` | Datos | Métricas de prueba (niveles 1 y 2) por subconjunto, con IC |
| 2026-11-25 | F6 | `fase6/control_dryad_ong.json`, `atajo_plataforma.csv` | Datos | Δ_control, Δ_campo, orden de arquitecturas y casos de atajo por plataforma |
| 2026-11-25 | F6/F7 | `fase6/pruebas_estadisticas.json`, `seleccion_mejor_modelo.json` | Datos | Q de Cochran, McNemar y jerarquía de selección |
| 2026-11-25 | F6 | `fase6/gradcam_indicadores.csv`, `gradcam_resumen.json`, `gradcam_laminas/` | Evidencia | Indicadores de Grad-CAM por imagen y estrato; láminas con las cuatro arquitecturas |
| 2026-11-25 | F6 | `fase6/ecuador.json`, `ecuador_casos.csv` | Datos | Validación contextual: métricas, análisis secundarios y casos |
| 2026-11-25 | F6 | `fase6/eficiencia.json` | Datos | Tiempo de inferencia, RAM y tamaño por arquitectura |
| 2026-11-25 | F6 | `fase6/figuras/` | Evidencia | Curvas de entrenamiento, macro F1 por subconjunto y matrices de confusión |
| 2026-11-25 | F6 | `fase6/cache_ecuador_512/` | Caché | Copias reducidas del conjunto ecuatoriano; regenerable |
| 2026-10-27 | F2/F4 | `fase2/ficha_tecnica.md` (actualizada) | Instrumento 3.6.1.3 | Implementación del entrenamiento, parámetros entrenables y rendimiento del piloto |
| 2026-10-24 | F3 | `fase2/protocolo_evaluacion.md` (v1.0) | Protocolo | Evaluación por repositorio, subconjunto de control y Grad-CAM, registrada antes de la partición (SHA-256 en la Entrada 007 bis) |

¹ Fecha exacta no recuperable: la v2 del protocolo fue sobrescrita por la v3 el 2026-09-24. Consta que es posterior a la inspección de la franja 5–10 (2026-09-20 22:39).

Las fechas de las filas de F3 se reconstruyeron el 2026-09-27 a partir de la fecha de modificación de cada archivo (Entrada 006 bis).

---

## Anexo — Registro de incidencias

| Fecha | Fase | Incidencia | Solución | ¿Afectó resultados? |
|---|---|---|---|---|
| 2026-09-19 | F2 | Python 3.9.25 por defecto en AlmaLinux 9.8, incompatible con scipy 1.14.1 | Instalación de Python 3.11 desde repositorios oficiales y recreación del entorno | No |
| 2026-09-19 | F2 | Coexistencia de `opencv-python-headless 4.10.0.84` y `opencv-python 5.0.0.93` | Desinstalación de la versión headless; se fija `opencv-python 5.0.0.93` | No |
| ≤ 2026-09-20 | F3 | El conjunto descargado previamente mezclaba originales con imágenes derivadas (rotaciones, aumentadas y transformadas) | Redescarga desde las fuentes seleccionando únicamente imágenes originales | No; detectado antes de depurar |
| 2026-09-20 | F3 | Primera ejecución del consolidador con ruta incorrecta (`TESIS` en lugar de `TESIS/DATASETS`) | Detectado en modo simulación; se corrigió la ruta sin escribir archivos | No |
| 2026-09-20 | F3 | Espacio en disco insuficiente para copiar el repositorio (29 GB frente a 21 GB libres) | Construcción del conjunto bruto mediante enlaces duros | No |
| 2026-09-20 | F3 | Los archivos JPEG truncados superaban la verificación de integridad al decodificarse parcialmente con relleno gris | E1 se aplicará sin tolerancia a truncamiento en el pipeline de depuración | No; detectado antes de depurar |
| 2026-09-24 | F3 | Imagen de Couret etiquetada como *Ae. aegypti* sin sus rasgos diagnósticos aparentes | Verificada: corresponde a la cepa *New Orleans* de *Ae. aegypti*; la tonalidad parda es propia de ejemplares de colonia secos. Sospecha descartada | No |
| 2026-09-24 | F3 | E4 con pHash de 64 bits agrupó especímenes distintos del mismo montaje de laboratorio; el encadenamiento transitivo arrastró 2.385 imágenes a E8 | pHash de 256 bits con verificación por correlación ≥ 0,98; unión solo de pares verificados; E8 restringido a duplicados exactos | No; detectado y corregido antes de escribir el conjunto depurado |
| 2026-09-24 | F3 | E5 por brillo medio descartó 93 fotografías de microscopio correctamente expuestas de Dryad | Reoperacionalización como rango dinámico (p99 − p1) con umbral 40 | No; detectado y corregido antes de escribir el conjunto depurado |
| 2026-09-27 | F3 | `estructura_proyecto.txt` omitía `fase3/conjunto_bruto/` y `fase3/conjunto_depurado/`, declaradas en el inventario, y listaba archivos comprimidos no inventariados. Detectado por el tutor | Auditoría con `auditar_estructura.py`: ambas carpetas existen con los conteos declarados; árbol regenerado e inventario completado (Entrada 006 bis) | No |
| 2026-09-27 | F3 | Pares 1 y 2 de `fase3/verificacion_dup/` conservan solo la imagen eliminada (A); falta la imagen con la que se emparejó (B) | **Abierta.** Recuperar la imagen B desde `conjunto_bruto/` o declarar la limitación de la evidencia | No afecta al conjunto; limita la verificación independiente de la primera ejecución |
| 2026-09-27 | F3 | Entrada 006 fechada el 24/09 aunque sus partes A a C se ejecutaron el 20/09 | Fecha corregida según la fecha de modificación de los archivos | No |
| 2026-09-27 | F3 | En Mosquito Alert, un reporte ciudadano puede contener varias fotografías del mismo ejemplar (mismo identificador de reporte, distinto identificador de imagen). `consolidar_conjunto.py` usa el identificador de imagen como espécimen, por lo que esas fotografías se tratan como independientes | Resuelta el 23/10/2026 (Entrada 006 quater): agrupación por reporte con los metadatos públicos; las 3.839 fotos sin reporte se excluyen del conjunto de trabajo | No; resuelta antes de la partición |
| 2026-10-01 | F3 | Tres PDF de la bibliografía no corresponden al documento completo citado: [09] solo resumen, [38] extracto de 3 páginas, [48] documento CD63/7 de 2026 | Resuelta para [38] y [48] (texto completo y documento CD55/16). [09] queda con el resumen, porque el texto completo es de suscripción | No afecta a los datos |
| 2026-10-01 | F3 | [16] Lapo-Talledo (2024) no disponible: revista de suscripción sin acceso en la biblioteca de la PUCE | Se mantiene en el índice con su DOI verificado, sin PDF | No afecta a los datos |
| 2026-10-23 | F3 | Ong no tiene identificador de espécimen; la similitud automática encadena mosquitos distintos por el fondo y no separa limpiamente los casos | Recorte centrado en el mosquito y dos rondas de revisión visual con 149 uniones documentadas (Entrada 006 quater) | No; resuelta antes de la partición |
| 2026-10-23 | F3 | Las versiones 1 a 3 de `agrupar_especimenes.py` se sobrescribieron, contra la medida de la Entrada 006 bis | Cambios descritos en el encabezado del script y en la Entrada 006 quater | No |
| 2026-10-23 | F3 | Error en la Entrada 005: `DeviceDataset` de Dryad se excluyó por suponer que sus clases eran dispositivos; `body` contiene fotos de celular por especie de los mismos especímenes | Incorporación de 199 + 199 fotos `Phone_` en la consolidación v2 (Entrada 006 quinquies) | Sí: amplía Dryad de 398 a 597 por especie |
| 2026-10-24 | F3 | La primera consolidación v2 se ejecutó con la versión anterior del script (sin fotos `Phone_`): el navegador no reemplazó el archivo descargado | Detectado porque Dryad seguía en 398; se borraron las carpetas `_v2` y se repitieron consolidación y depuración | No; corregido antes de agrupar |
| 2026-10-24 | F3 | Aviso `DecompressionBombWarning` de PIL al leer una imagen de Mosquito Alert de 108 megapíxeles | Solo es un aviso; la imagen se procesó normalmente | No |
| 2026-10-24 | F3 | Las láminas con recorte automático encuadraban patas, fondo o abdomen en lugar del tórax; no permitían ver el escudo | Exportación de la muestra completa a 1.600 px, sin etiqueta (Entrada 007) | No; detectado antes de registrar veredictos |
| 2026-10-24 | F3 | La verificación marcó como `otro` 59 ejemplares de Dryad (100 % de error en ambas especies) por ausencia de escamas blancas | Ejemplares de colonia secos; criterio no aplicable. Reclasificados a `indeterminable`, decisión posterior a la clave y declarada (Entrada 007, Parte C) | Sí: cambia la decisión sobre Dryad de "evaluar exclusión" a "revisión documental" |
| 2026-10-24 | F3 | Dos larvas entre las 110 imágenes de iNaturalist revisadas: la descarga no filtró por estadio (E7) | Filtro por anotación de estadio vital (26 larvas excluidas) y revisión visual de las 652 no anotadas (ninguna) | Sí: reduce iNaturalist a 1.064 antes del rebalanceo |
| 2026-10-24 | F3 | El log de la partición quedó vacío: la partición se ejecutó dos veces y la segunda se detuvo (por diseño) sin escribir en la salida estándar | Log regenerado con `--simular`, que reproduce los mismos números con la misma semilla; la partición escrita no se modificó | No |
| 2026-10-24 | F3 | matplotlib sin interfaz gráfica (backend `agg`) para trazar las cajas | Instalación de `python3.11-tkinter` 3.11.13 y `tk` 8.6.10 desde los repositorios de AlmaLinux; backend `TkAgg` | No. Registrar en la ficha técnica |
| 2026-10-24 | F4 | Grad-CAM falló en la prueba local con la versión más reciente de grad-cam, que exige indicar la clase objetivo | Se pasa explícitamente `ClassifierOutputTarget` de la clase predicha; probado también con grad-cam 1.5.4 | No; corregido antes de ejecutar el piloto |
| 2026-10-27 | F4 | Aviso `FutureWarning` de `torch.load` (`weights_only`) | Se carga con `weights_only=True` | No |
| 2026-10-27 | F4 | Alerta de Grad-CAM en ViT-S/16 en el piloto (61 % de picos fuera de la caja) | Documentada; se informa al tutor y se continúa sin medida correctiva (Entrada 009) | No en el piloto; se evalúa en la Fase 6 |
| 2026-11-17 | F5 | El log de la búsqueda tardaba en mostrar líneas: Python acumula la salida cuando se redirige a un archivo | Sin efecto en el entrenamiento; las corridas definitivas se lanzan con `python -u` | No |
| 2026-11-24 | F5 | La RAM pico se mide con `ru_maxrss`, que es el máximo acumulado del proceso, no el de cada corrida. Como las 20 corridas (y las cuatro del piloto) se ejecutaron en un solo proceso, cada valor arrastra el de las anteriores: solo es propio de una arquitectura cuando sube respecto al anterior (EfficientNet-B0 y ResNet50). Los 1.676 MB de MobileNetV3-Large y los 1.860 MB de ViT-S/16 son solo cotas superiores. Tampoco incluye los 2 procesos de lectura | Medir la RAM de cada arquitectura en un proceso aparte durante la Fase 6, cargando el *checkpoint* de la corrida mediana | No en los modelos ni en las métricas de clasificación. Sí en la medida de RAM del piloto y de la Fase 5, que no se usa para comparar |
| 2026-11-25 | F6 | El primer intento de `evaluar.py` se detuvo en Grad-CAM (`KeyError`): en Ong, las dos especies tienen archivos con el mismo nombre (p. ej., `ong__7.jpg`) y el script buscaba la imagen solo por su nombre | Clave por especie y nombre en Grad-CAM y en la lectura de predicciones guardadas. Se relanzó la evaluación completa, que recalculó las predicciones de los 20 modelos | No: las métricas ya se calculaban sobre la lista de prueba ordenada. La inferencia es determinista y los resultados registrados son los de la ejecución completa |
| 2026-11-25 | F6 | El "pico del proceso" de la medida de eficiencia es igual en las cuatro arquitecturas (1.497 MB): `ru_maxrss` conserva el máximo del proceso padre al lanzar el subproceso | Se usa la RAM residente (base y tras la inferencia), que sí es propia de cada arquitectura | No |
| 2026-10-24 | F3 | `verificacion_morfologica.py` v1 y v2 sobrescritas | Cambios descritos en el encabezado del script y en esta entrada | No |
| 2026-10-24 | F3 | `consolidar_conjunto.py` v2 sobrescribió la v1 en la misma ruta | La v1 se conserva en el paquete de entrega de las Fases 2 y 3 y en los archivos del proyecto | No |

---

## Anexo — Cambios al documento derivados de la experimentación

| Fecha | Sección modificada | Qué cambió | Motivo |
|---|---|---|---|
| 2026-09-19 | Tabla 2 | Sustitución de la estimación teórica por tiempos medidos | Benchmark empírico en el equipo de trabajo |
| 2026-09-19 | 3.7, Fase 5 | Reformulación del párrafo del presupuesto de cómputo | Base empírica en lugar de estimación |
| 2026-09-19 | 2.2.8 | Precisión sobre el conteo de parámetros | Discrepancia entre clasificador de 1000 y de 2 clases |
| 2026-09-19 | 3.3 | Registro de Python 3.11.13, AlmaLinux 9.8 y opencv 5.0.0.93 | Versiones reales del entorno |
| 2026-09-19 | 3.6.2.1 | Umbral de desenfoque pasa de valor fijo a determinación empírica | Un umbral de 3 no filtraba ninguna imagen; se formaliza el procedimiento de determinación |
| 2026-09-20/24 ¹ | Protocolo de criterios, sección 3 | Normalización a 512 px, contención por especie y por repositorio, muestra estratificada | La métrica depende de la resolución y del tipo de fondo; sin desglose por fuente, un umbral podía eliminar un repositorio completo |
| 2026-09-20/24 ¹ | Protocolo de criterios, E6 | Umbral de desenfoque fijado en 10 | Resultado de la determinación empírica (Entrada 006) |
| 2026-09-20/24 ¹ | Protocolo de criterios, E1 | Archivos truncados se tratan como corruptos | Superaban la verificación de integridad permisiva |
| 2026-09-20/24 ¹ | Protocolo de criterios, sección 4 | Revisión íntegra del repositorio Couret | Posible error de etiquetado detectado |
| 2026-09-24 | Documento, 3.6.2.1 | **Pendiente:** registrar el umbral adoptado (10) y la limitación de la métrica | Coherencia entre plan y ejecución |
| 2026-09-24 | Protocolo de criterios, E4 | pHash de 256 bits con verificación por correlación; sin encadenamiento transitivo | Falsos positivos entre especímenes del mismo montaje de laboratorio |
| 2026-09-24 | Protocolo de criterios, E5 | Rango dinámico (p99 − p1 < 40) en lugar del brillo medio | El promedio global mide el fondo, no la exposición del espécimen |
| 2026-09-24 | Protocolo de criterios, E8 | Restringido a duplicados exactos (MD5) entre especies | La similitud perceptual no basta para declarar la etiqueta ambigua |
| 2026-09-24 | Protocolo de criterios, sección 3 bis | Sección nueva con la operacionalización de E4 y E5 y el hallazgo metodológico | Documentar por qué fallaron las formulaciones iniciales |
| 2026-09-24 | Protocolo de criterios, sección 4 | Corrección de la nota sobre Couret | Sospecha de etiquetado descartada tras verificar la cepa *New Orleans* |
| 2026-09-24 | Documento, 3.4 | **Pendiente:** actualizar el tamaño del conjunto: 1.529 por especie tras depuración, frente a las ~2.700 previstas | Cifras reales de la construcción del conjunto |
| 2026-10-01 | Documento, referencias [3], [5], [20], [38], [39] | **Pendiente:** agregar el DOI de [3]; corregir el número de [5], la conferencia de [20] y el autor de [38] y [39] | Verificación de la bibliografía (Entrada 006 ter) |
| 2026-10-01 | Documento, referencia [48] | **Pendiente:** cambiar el título a *Strategy for Arboviral Disease Prevention and Control* (CD55/16), 2016 | El documento citado con el título anterior no es de 2016 |
| 2026-10-23 | Documento, protocolo de evaluación | **Pendiente:** añadir la evaluación sobre el subconjunto de control (Dryad + Ong) y las métricas por repositorio, con el criterio de decisión fijado antes de entrenar | Observación del tutor del 14/10/2026 sobre el sesgo por tipo de fotografía |
| 2026-10-24 | Documento, 3.4 | **Pendiente:** población y tamaño: seis repositorios (se añade iNaturalist fuera de Ecuador), Kaggle y Couret fuera del conjunto de trabajo, 1.987 imágenes por especie con composición igualada por tipo de fotografía | Entradas 006 quater y 006 quinquies; observación del tutor del 23/10/2026 |
| 2026-10-24 | Documento, 3.4 | **Pendiente:** sustituir "repositorios públicos adicionales distintos de iNaturalist" por la formulación que mantiene reservadas solo las observaciones ecuatorianas | Desviación declarada en la Entrada 006 quinquies |
| 2026-10-24 | Informe de la Fase 3 | **Pendiente:** actualizar repositorios, tablas de composición y la corrección sobre `DeviceDataset` | Entrada 006 quinquies |
| 2026-10-24 | Protocolo de criterios, sección 4 | **Pendiente:** categoría `otro`; `otro` no se asigna por ausencia de escamas en ejemplares de laboratorio; ejecución asistida por IA con confirmación del investigador | Entrada 007 |
| 2026-10-24 | Protocolo de criterios, E7 | **Pendiente:** en iNaturalist, E7 se aplica con la anotación comunitaria de estadio vital | Larvas encontradas en la verificación (Entrada 007) |
| 2026-10-24 | Documento, 3.6 / limitaciones | **Pendiente:** baja proporción de imágenes con rasgo diagnóstico visible (27,5 %) y ausencia de entomólogo en la verificación | Entrada 007 |
| 2026-10-24 | Documento, 2.2.9, 3.1, 3.6.3 y 3.7 (Fase 6) | **Pendiente:** Grad-CAM con muestra estratificada (≈ 68) e indicadores de pico y concentración con umbral de alerta | Protocolo de evaluación (Entrada 007 bis) |
| 2026-10-24 | Documento, 3.4 y 3.8 | **Pendiente:** subconjunto de control Dryad + Ong, criterio Δ ≥ 0,05 y análisis de sensibilidad del orden de arquitecturas | Protocolo de evaluación (Entrada 007 bis) |
| 2026-10-24 | Documento, 3.4 | **Pendiente:** tamaño final 1.958 por especie (3.916 en total; 2.737 / 590 / 589) y método de partición agrupada por asignación al mayor déficit | Entrada 008 |
| 2026-10-24 | Documento, 3.5 / 3.6 | **Pendiente:** con las cifras finales, el entrenamiento procesa ≈ 8.211 imágenes por época tras el aumento, frente a 10.800 en la Tabla 2 | Entrada 008; se confirmará en el piloto |
| 2026-10-27 | Documento, Tabla 2 y 3.7 (Fases 4 y 5) | **Pendiente:** sustituir la estimación por lo medido en el piloto: 1,20 / 0,70 / 2,78 / 2,27 h por corrida y 38,9 h en total frente a 240 h | Entrada 009 |
| 2026-10-27 | Documento, 3.2 / 3.3 | **Pendiente:** precisar los módulos entrenados en el ajuste fino parcial y la proporción de parámetros entrenables de cada arquitectura | Entrada 009 y ficha técnica |
| 2026-11-16 | Documento, 3.4 | **Pendiente:** resultado de la curva de aprendizaje (macro F1 de 0,7253 a 0,8220; mejora 75 → 100 % de +0,0169, inferior al criterio de 0,02) como verificación empírica del tamaño del conjunto | Entrada 009 |
| 2026-11-22 | Documento, Tabla 2 y 3.7 (Fase 5) | **Pendiente:** cómputo real de la Fase 5: 3,98 h de búsqueda + 21,03 h de corridas definitivas = 25,0 h frente a 240 h; 16 de 20 corridas con detención temprana | Entradas 010 y 011 |
| 2026-11-22 | Documento, 3.8 (eficiencia computacional) | **Pendiente:** precisar que la inferencia se mide por imagen en lotes de 64 y que la RAM se mide por arquitectura en un proceso aparte | Entrada 011 e incidencia del 2026-11-24 |
| 2026-11-25 | Documento, 3.8 | **Pendiente:** precisar las reglas de empate de la jerarquía (diferencia < 0,02 en macro F1 y en sensibilidad de *Ae. aegypti*) y el *bootstrap* por grupos como análisis de sensibilidad de las pruebas | Anexo del protocolo para la Fase 6 (Entrada 011 bis) |
| 2026-11-25 | Documento, 3.4 y limitaciones | **Pendiente:** el conjunto ecuatoriano tiene 71 imágenes `research` de *Ae. aegypti* de solo 19 observaciones, y una aporta 32 vistas parciales | Entradas 011 bis y 013 |
| 2026-11-25 | Documento, resultados y limitaciones | **Pendiente:** resultados de las Entradas 012 a 015; salvedad del orden en el control; alerta de Grad-CAM en las cuatro arquitecturas (sobre todo en M-al y en ViT-S/16) | Entradas 012 a 015 |

¹ Fecha exacta no recuperable (ver nota del inventario).
