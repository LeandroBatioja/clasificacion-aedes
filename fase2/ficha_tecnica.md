# Ficha técnica de configuración experimental

> **Instrumento 3.6.1.3** del plan de investigación
> **Proyecto:** Comparación de arquitecturas de aprendizaje profundo para la clasificación de mosquitos vectores del dengue mediante reconocimiento de imágenes
> **Autor:** Leandro Steven Batioja Mena — PUCE Esmeraldas 2026–2027
> **Fecha de registro:** 19 de septiembre de 2026
> **Actualización:** 27 de octubre de 2026 — implementación del entrenamiento y resultados de la prueba piloto (Fase 4)

---

## 1. Entorno de ejecución

| Campo | Valor |
|---|---|
| Fecha de verificación | 2026-09-19 23:09:16 |
| Sistema operativo | Linux 5.14.0-687.46.1.el9_8.x86_64 (AlmaLinux 9.8) |
| Versión de Python | 3.11.13 |
| Procesador (CPU) | Intel Core i5-10400 @ 2.90 GHz (6 núcleos / 12 hilos) |
| Memoria RAM | 23,0 GB |
| Aceleración por GPU | No (CUDA disponible: `False`) |
| Hilos de PyTorch | 6 |

---

## 2. Versiones de librerías

| Librería | Versión instalada |
|---|---|
| torch | 2.4.1+cpu |
| torchvision | 0.19.1+cpu |
| timm | 1.0.9 |
| scikit-learn | 1.5.2 |
| statsmodels | 0.14.4 |
| scipy | 1.14.1 |
| opencv-python | 5.0.0.93 |
| Pillow | 10.4.0 |
| ImageHash | 4.3.1 |
| numpy | 2.1.1 |
| pandas | 2.2.3 |
| matplotlib | 3.9.2 |
| grad-cam | 1.5.4 |

**Paquetes del sistema añadidos en la Fase 3:** `python3.11-tkinter` 3.11.13 y `tk` 8.6.10 (repositorios de AlmaLinux), necesarios para la interfaz gráfica de matplotlib (`TkAgg`) con la que se trazaron las cajas del espécimen.

---

## 3. Arquitecturas evaluadas

| Arquitectura | Identificador timm | Parámetros con 2 clases (M) | Parámetros originales ImageNet (M) |
|---|---|---|---|
| EfficientNet-B0 | `efficientnet_b0` | 4,01 | ~5,3 |
| MobileNetV3-Large | `mobilenetv3_large_100` | 4,20 | ~5,4 |
| ResNet50 | `resnet50` | 23,51 | ~25,6 |
| ViT-S/16 | `vit_small_patch16_224` | 21,67 | ~22,0 |

**Pesos preentrenados:** ImageNet-1k
**Resolución de entrada:** 224 × 224 px
**Clases:** 2 (*Aedes aegypti*, *Aedes albopictus*)

> La diferencia entre ambas columnas obedece a la sustitución de la capa clasificadora original de mil salidas por una de dos, requerida por la clasificación binaria de este estudio.

---

## 4. Parámetros de normalización

| Arquitectura | Media (RGB) | Desviación estándar (RGB) |
|---|---|---|
| EfficientNet-B0 | 0,485 · 0,456 · 0,406 | 0,229 · 0,224 · 0,225 |
| MobileNetV3-Large | 0,485 · 0,456 · 0,406 | 0,229 · 0,224 · 0,225 |
| ResNet50 | 0,485 · 0,456 · 0,406 | 0,229 · 0,224 · 0,225 |
| ViT-S/16 | 0,500 · 0,500 · 0,500 | 0,500 · 0,500 · 0,500 |

> ViT-S/16 emplea una normalización distinta a las tres arquitecturas convolucionales. Cada modelo debe preprocesarse con los valores que le corresponden.

---

## 5. Protocolo de entrenamiento

| Parámetro | Valor |
|---|---|
| Estrategia de ajuste fino | Parcial: congelamiento de capas iniciales, entrenamiento de últimos bloques y clasificador |
| Épocas máximas (definitivas) | 15 |
| Detención temprana | 5 épocas sin mejora |
| Épocas (búsqueda de hiperparámetros) | 10 |
| Submuestra para búsqueda | 30 % del conjunto de entrenamiento |
| Configuraciones por arquitectura | 3 |
| Corridas definitivas por arquitectura | 5 |
| Función de pérdida | CrossEntropyLoss |
| Optimizador | AdamW |
| Criterio de selección de checkpoint | Mayor macro F1 en validación |
| Weight decay de AdamW | 0,01 (valor por defecto de PyTorch; no forma parte de la búsqueda) |
| Programa | `fase4/entrenar.py` (mismo código para el piloto y la Fase 5) |

### Implementación (fijada en la Fase 4)

| Aspecto | Decisión |
|---|---|
| Carga de imágenes | PIL con `ImageOps.exif_transpose` (misma orientación que las cajas de Grad-CAM) |
| Copia reducida | Cada imagen se guarda una vez con su lado mayor ≤ 512 px (`fase4/cache_512`), para no decodificar fotografías de hasta 108 MP en cada época |
| Redimensionado | Directo a 224 × 224, bicúbico, sin recorte central (un recorte podría eliminar ejemplares situados cerca del borde) |
| Aumento ×2 | En cada época, cada imagen original se usa sin transformar y en dos versiones aumentadas: volteo horizontal, rotación ±15°, escala 0,9–1,1, brillo y contraste ±0,2. Imágenes por época: 3 × originales |
| Ajuste fino parcial | Se congela la red y se entrenan la última etapa y la cabeza. Las capas congeladas conservan BatchNorm en modo evaluación |
| Grad-CAM | Clase predicha. EfficientNet-B0: `bn2`; MobileNetV3-Large: `blocks[-1]` (en timm, `conv_head` actúa después del *pooling*); ResNet50: `layer4[-1]`; ViT-S/16: `blocks[-1].norm1` con transformación de *tokens* a malla 14 × 14 |
| Hilos y lectura | 6 hilos de PyTorch y 2 procesos de lectura |

| Arquitectura | Módulos entrenados | Parámetros entrenables (M) | % del total |
|---|---|---|---|
| EfficientNet-B0 | `blocks[5]`, `blocks[6]`, `conv_head`, `bn2`, `classifier` | 3,16 | 79 % |
| MobileNetV3-Large | `blocks[5]`, `blocks[6]`, `conv_head`, `norm_head`, `classifier` | 3,41 | 81 % |
| ResNet50 | `layer4`, `fc` | 14,97 | 64 % |
| ViT-S/16 | `blocks[9–11]`, `norm`, `fc_norm`, `head` | 5,32 | 25 % |

> El criterio es común (la última etapa de la red y la cabeza), pero la proporción de parámetros entrenados difiere porque las redes convolucionales concentran sus parámetros en las etapas finales. Se considera un factor explicativo en el análisis comparativo, junto con el tamaño de cada modelo.

### Espacio de búsqueda de hiperparámetros

| Hiperparámetro | Valores |
|---|---|
| Tasa de aprendizaje | 1e-3, 5e-4, 1e-4 |
| Tamaño de batch | 16, 32 |

### Configuración seleccionada por arquitectura

Búsqueda: tres configuraciones elegidas al azar (semilla 42) del espacio de seis, las mismas para las cuatro arquitecturas; 30 % del entrenamiento por grupos completos y estratificado; máximo 10 épocas con detención tras 5 sin mejora; se elige la de mayor macro F1 en validación (`fase5/busqueda/configuracion_seleccionada.json`). Configuraciones sorteadas: (1e-3, 16), (1e-4, 16) y (1e-4, 32). Búsqueda terminada el 17/11/2026 (bitácora, Entrada 010).

| Arquitectura | Tasa de aprendizaje | Batch | Macro F1 en validación (búsqueda, 30 %) |
|---|---|---|---|
| EfficientNet-B0 | 1e-3 | 16 | 0,7559 |
| MobileNetV3-Large | 1e-3 | 16 | 0,7423 |
| ResNet50 | 1e-3 | 16 | 0,7814 |
| ViT-S/16 | 1e-4 | 16 | 0,8488 |

---

## 6. División del conjunto de datos

| Parámetro | Valor |
|---|---|
| Proporción entrenamiento | 70 % |
| Proporción validación | 15 % |
| Proporción prueba | 15 % |
| Estratificación | Por especie y por repositorio de origen |
| Agrupamiento | Por espécimen (identificador de observación) |
| Factor de aumento de datos | ×2, solo en entrenamiento |
| Semillas aleatorias | 42 para la partición, el piloto, la curva de aprendizaje y la búsqueda; 42, 43, 44, 45 y 46 para las cinco corridas definitivas |

---

## 7. Rendimiento medido

Medición realizada el 19 de septiembre de 2026 con lotes de 16 imágenes, 5 lotes por arquitectura, reportándose la mediana. Ajuste fino restringido a la capa clasificadora.

| Arquitectura | Inferencia (img/s) | ms/img | Entrenamiento (img/s) | RAM pico (MB) | Horas/corrida |
|---|---|---|---|---|---|
| EfficientNet-B0 | 43,7 | 22,9 | 39,3 | 399 | 1,14 |
| MobileNetV3-Large | 74,6 | 13,4 | 66,1 | 434 | 0,68 |
| ResNet50 | 17,1 | 58,6 | 16,6 | 506 | 2,71 |
| ViT-S/16 | 24,3 | 41,1 | 25,9 | 525 | 1,74 |

### Presupuesto de cómputo

| Concepto | Horas |
|---|---|
| Búsqueda de hiperparámetros (12 entrenamientos abreviados) | 3,7 |
| Corridas definitivas (20 entrenamientos) | 31,3 |
| **Total requerido** | **35,1** |
| **Disponible (Fase 5)** | **240,0** |
| **Margen** | **204,9 (+85 %)** |

---

### Prueba piloto (Fase 4, 27/10/2026)

400 imágenes de entrenamiento (1.200 por época con el aumento), 5 épocas, lr 5e-4, batch 32, semilla 42; validación completa (590 imágenes). Incluye lectura de disco, aumento y entrenamiento de los últimos bloques.

| Arquitectura | Entrenamiento (img/s) | Factor frente a la Fase 2 | RAM pico acumulada del proceso (MB) | Horas por corrida completa (8.211 img/época, 15 épocas) |
|---|---|---|---|---|
| EfficientNet-B0 | 30,0 | 0,76 | 1.774 | 1,20 |
| MobileNetV3-Large | 52,0 | 0,79 | 1.774 | 0,70 |
| ResNet50 | 13,0 | 0,78 | 2.038 | 2,78 |
| ViT-S/16 | 15,8 | 0,61 | 2.136 | 2,27 |

| Concepto | Horas |
|---|---|
| Corridas definitivas (20) | 34,8 |
| Búsqueda de hiperparámetros (12) | 4,2 |
| **Total estimado (sin contar la detención temprana)** | **38,9** |
| Disponible (Fase 5) | 240,0 |

### Corridas definitivas (Fase 5, 21–22/11/2026)

20 corridas (5 semillas × 4 arquitecturas) con la configuración seleccionada; 8.211 imágenes por época; máximo 15 épocas con detención tras 5 sin mejora. Medido en el equipo de trabajo, con lectura desde la copia reducida.

| Arquitectura | Entrenamiento (img/s) | Inferencia (ms/img, mediana) | Horas por corrida (media) | Tamaño del *checkpoint* (MB) | Corrida mediana (semilla) |
|---|---|---|---|---|---|
| EfficientNet-B0 | ≈ 40 | 18,98 | 0,73 | 15,57 | 46 |
| MobileNetV3-Large | ≈ 64 | 11,86 | 0,41 | 16,23 | 43 |
| ResNet50 | ≈ 16 | 50,68 | 1,28 | 89,99 | 44 |
| ViT-S/16 | ≈ 17 | 39,83 | 1,79 | 82,70 | 46 |

| Concepto | Horas |
|---|---|
| Búsqueda de hiperparámetros (12 entrenamientos) | 3,98 |
| Corridas definitivas (20 entrenamientos, 221 épocas) | 21,03 |
| **Total de la Fase 5** | **25,0** |
| Disponible (Fase 5) | 240,0 |

La inferencia es el tiempo de validación de la mejor época dividido entre 590 imágenes, con lotes de 64 y 2 procesos de lectura; incluye la lectura y el redimensionado.

**RAM pico.** Los valores de RAM del piloto y de la Fase 5 se midieron con `ru_maxrss`, que es el máximo acumulado del proceso. Como todas las corridas se ejecutaron en un mismo proceso, cada valor incluye los de las arquitecturas anteriores y no sirve para compararlas. La RAM por arquitectura se mide en la Fase 6, cada una en un proceso aparte.

### Eficiencia en inferencia (Fase 6, 25/11/2026)

Medida con el *checkpoint* de la corrida mediana, cada arquitectura en un proceso aparte, con 6 hilos. La latencia es la mediana de 100 imágenes de prueba con lote de 1, con los tensores ya en memoria (anexo del protocolo, 5).

| Arquitectura | ms/imagen (lote 1) | ms/imagen (lote 64) | RAM residente tras inferencia | RAM del modelo e inferencia | Tamaño |
|---|---|---|---|---|---|
| EfficientNet-B0 | 17,61 | 18,13 | 500 MB | 298 MB | 15,57 MB |
| MobileNetV3-Large | 12,48 | 10,97 | 525 MB | 322 MB | 16,23 MB |
| ResNet50 | 50,43 | 48,40 | 591 MB | 388 MB | 89,99 MB |
| ViT-S/16 | 43,74 | 38,41 | 624 MB | 422 MB | 82,70 MB |

La RAM base del proceso, con Python, PyTorch y timm cargados y sin modelo, es de 203 MB.

## 8. Observaciones

**Versión de Python.** AlmaLinux 9.8 incorpora Python 3.9.25 por defecto, incompatible con las versiones de scipy y numpy declaradas en el plan. Se instaló Python 3.11 desde los repositorios oficiales de la distribución, conforme a lo especificado en la sección 3.3.

**Conflicto de OpenCV.** Se detectó la coexistencia de `opencv-python-headless 4.10.0.84` y `opencv-python 5.0.0.93`, este último instalado como dependencia transitiva de `grad-cam`. Se eliminó la versión headless para garantizar la resolución unívoca del módulo `cv2`.

**Alcance de la medición de rendimiento.** Los tiempos registrados corresponden a tensores sintéticos en memoria y a un ajuste fino restringido a la capa clasificadora. No incorporan el costo de lectura de disco, decodificación de imágenes ni aumento de datos en tiempo de ejecución, ni el mayor costo del entrenamiento de los últimos bloques previsto en la sección 3.2. Por tanto, el tiempo real esperado es superior al estimado. La prueba piloto de la Fase 4 permitirá cuantificar esta diferencia. Aun aplicando un factor de corrección conservador de tres, el requerimiento se mantendría en torno a las 105 horas, holgadamente por debajo de las 240 disponibles.
