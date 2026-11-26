# Protocolo de evaluación — registro previo a la partición

**Proyecto:** Comparación de arquitecturas de aprendizaje profundo para la clasificación de mosquitos vectores del dengue
**Autor:** Leandro Steven Batioja Mena — PUCE Esmeraldas
**Versión:** 1.0 — 2026-10-24
**Ubicación:** `fase2/protocolo_evaluacion.md`

---

## 0. Propósito y estatus

Este documento fija, **antes de dividir el conjunto y antes de entrenar**, cómo se evaluará si los modelos aprenden la morfología del mosquito o el tipo de fotografía. Atiende la observación del tutor del 23/10/2026: resultados aparte con Dryad y Ong, métricas por repositorio y revisión con Grad-CAM de si el modelo mira el mosquito o el fondo.

Complementa las secciones 3.4, 3.7 (Fases 4 y 6) y 3.8 del plan de investigación sin sustituirlas. La selección del mejor modelo, las pruebas estadísticas (Q de Cochran y McNemar con Bonferroni) y la jerarquía de criterios de la sección 3.8 no cambian.

**Regla de registro.** El archivo se registra en la bitácora (Entrada 007 bis) con su huella SHA-256. Ningún umbral, muestra ni criterio de este documento se modifica después de ver resultados de prueba. Todo cambio posterior se registra como desviación, con fecha y justificación, y se reportan ambos análisis: el previsto y el modificado.

---

## 1. Conjunto de evaluación

| Concepto | Valor |
|---|---|
| Conjunto de trabajo | 1.958 imágenes por especie, tras las exclusiones de la Entrada 007 y el rebalanceo de la Entrada 008 |
| Partición | 70 / 15 / 15, fija, agrupada por espécimen (`grupo`), estratificada por especie × repositorio, semilla 42 |
| Conjunto de prueba | Único para las cuatro arquitecturas; no interviene en la selección de hiperparámetros ni de *checkpoints* |
| Modelo evaluado en los análisis de este protocolo | El de la **corrida mediana** de cada arquitectura según macro F1 en validación (el mismo que usa la sección 3.8) |

Cada imagen de prueba conserva su repositorio, especie y grupo, de modo que todos los análisis siguientes son subconjuntos del mismo conjunto de prueba. No hay inferencia adicional.

### Estratos

| Estrato | Repositorio | Especie | Tipo de fotografía |
|---|---|---|---|
| D-ae | Dryad | *Ae. aegypti* | Laboratorio: macro, microscopio y celular del mismo espécimen |
| D-al | Dryad | *Ae. albopictus* | Ídem |
| O-ae | Ong | *Ae. aegypti* | Laboratorio, mismo montaje para ambas especies |
| O-al | Ong | *Ae. albopictus* | Ídem |
| I-ae | iNaturalist (fuera de Ecuador) | *Ae. aegypti* | Celular de campo |
| M-al | Mosquito Alert | *Ae. albopictus* | Celular de campo |
| M-ae | Mosquito Alert | *Ae. aegypti* | Celular de campo (≈ 8 imágenes de prueba) |

**Subconjunto de control** = D-ae + D-al + O-ae + O-al. Es la única parte del conjunto en la que las dos especies fueron fotografiadas con el mismo procedimiento, por lo que el tipo de fotografía no puede delatar la especie.

**Subconjunto de campo** = I-ae + M-al + M-ae.

---

## 2. Métricas por repositorio

Para cada arquitectura (corrida mediana) se calculan:

| Subconjunto | Métricas | Motivo |
|---|---|---|
| Prueba completo | Todas las de la sección 3.8 | Análisis principal |
| Control (Dryad + Ong) | Exactitud, exactitud balanceada, macro F1, sensibilidad y especificidad por especie, matriz de confusión | Contiene ambas especies |
| Dryad y Ong por separado | Ídem | Contienen ambas especies |
| Campo (iNaturalist + Mosquito Alert) | Ídem | Contiene ambas especies, aunque cada una procede casi solo de una plataforma |
| I-ae y M-al | Sensibilidad de su especie (proporción de aciertos) | Estratos de una sola especie: el macro F1 no está definido |
| M-ae | Aciertos y errores, de forma descriptiva | ≈ 8 imágenes |

**Intervalos de confianza.** IC del 95 % por *bootstrap* de percentiles **remuestreando grupos (especímenes), no imágenes**, con 2.000 réplicas y semilla 42. Así se respeta que varias fotografías pueden ser del mismo ejemplar.

---

## 3. Subconjunto de control (Dryad + Ong): criterio de decisión

Para cada arquitectura se calcula

  **Δ_control = macro F1 (prueba completo) − macro F1 (control)**

con su IC del 95 % por *bootstrap* pareado de grupos (las mismas réplicas para ambos términos).

| Resultado | Interpretación registrada de antemano |
|---|---|
| Δ_control ≥ 0,05 **y** el IC excluye 0 | **Señal de dependencia del tipo de fotografía** para esa arquitectura. Se reporta como limitación y se aplica el análisis de sensibilidad de abajo |
| Cualquier otro caso | Sin evidencia de que el rendimiento global se apoye en el tipo de fotografía |

Se reporta además **Δ_campo = macro F1 (campo) − macro F1 (control)**, que es descriptivo. Un Δ_campo alto indica que el modelo acierta más donde el tipo de fotografía coincide con la especie.

**Análisis de sensibilidad (siempre se reporta; es decisivo si hay señal).** La jerarquía de selección de la sección 3.8 se aplica también sobre el subconjunto de control:

- Si el orden de las arquitecturas por macro F1 coincide en ambos, la conclusión sobre la mejor arquitectura se mantiene.
- Si difiere, la conclusión principal se formula con la salvedad de que el orden depende del tipo de fotografía, y se presentan ambos órdenes.

El umbral de 0,05 es una convención operativa fijada antes de ver datos, no un valor teórico.

---

## 4. Señales de atajo por plataforma

En la parte de campo, casi todo *Ae. aegypti* procede de iNaturalist y casi todo *Ae. albopictus* de Mosquito Alert. Un modelo podría aprender "plataforma → especie". Se registran de antemano dos indicadores descriptivos:

1. **M-ae** (*Ae. aegypti* de Mosquito Alert en prueba). Si la mayoría se clasifica como *Ae. albopictus* mientras I-ae se clasifica bien, el resultado es consistente con un atajo por plataforma.
2. **Conjunto ecuatoriano** (validación contextual, también de iNaturalist). Sus *Ae. aegypti* comparten plataforma con el entrenamiento, lo que puede favorecer los aciertos. Sus *Ae. albopictus* clasificados como *Ae. aegypti* son consistentes con el mismo atajo. Esta lectura se añade a la interpretación exploratoria prevista en la sección 3.8.

Por su tamaño, ninguno de los dos indicadores sustenta inferencia. Se reportan caso por caso.

---

## 5. Grad-CAM: ¿mira el mosquito o el fondo?

El plan (2.2.9 y 3.7, Fase 6) prevé una muestra de 20 imágenes con lectura ilustrativa. Para responder a la observación del tutor, la muestra se amplía y se estratifica por repositorio, y se añade un indicador simple fijado de antemano. Grad-CAM sigue siendo **complementario**: no interviene en la selección del mejor modelo ni en las pruebas estadísticas.

### 5.1 Muestra

- Se selecciona del **conjunto de prueba** en el momento de la partición, con semilla 42, sin mirar predicciones. Es la misma para las cuatro arquitecturas.
- 10 imágenes por estrato D-ae, D-al, O-ae, O-al, I-ae y M-al, más todas las de M-ae, **con una sola imagen por grupo**. Son unas 68 imágenes.
- Incluye aciertos y errores según caigan; no se seleccionan por resultado.

### 5.2 Caja del espécimen (anotada antes de entrenar)

Para cada imagen de la muestra, el investigador traza **una caja que encierra el espécimen completo** (cabeza, tórax, abdomen y patas visibles). Las cajas se trazan **antes de que exista ningún modelo entrenado**, se guardan en `fase3/particion/gradcam_cajas.csv` y no se modifican después.

Si la caja ocupa más del 60 % del área de la imagen (primeros planos), el indicador no discrimina entre mosquito y fondo: la imagen se marca como **no informativa** y solo se describe.

### 5.3 Mapas

| Arquitecturas | Técnica |
|---|---|
| EfficientNet-B0, MobileNetV3-Large, ResNet50 | Grad-CAM sobre la última capa convolucional, para la clase predicha |
| ViT-S/16 | Adaptación de `pytorch-grad-cam` para transformadores (transformación de *tokens* a malla 14 × 14) sobre el último bloque, para la clase predicha |

La diferencia técnica se declara al presentar los resultados: las visualizaciones no son estrictamente equivalentes entre ambas familias.

### 5.4 Indicadores por mapa

Con el mapa normalizado a [0, 1] y redimensionado a la imagen:

1. **Pico dentro:** el píxel de máxima activación cae dentro de la caja (sí o no).
2. **Concentración:** (fracción de la activación total dentro de la caja) ÷ (fracción del área de la imagen que ocupa la caja). Un valor mayor que 1 indica que la activación se concentra en el espécimen más de lo que correspondería por azar.

### 5.5 Criterio de alerta

Por arquitectura, sobre las imágenes informativas:

| Resultado | Interpretación registrada de antemano |
|---|---|
| Más del 25 % de los mapas con el pico fuera de la caja, en el total o en cualquier estrato con al menos 8 imágenes informativas | **Alerta de atención al fondo** para esa arquitectura o estrato |
| Mediana de la concentración ≤ 1 | Alerta de atención difusa o no localizada en el espécimen |
| Ninguna de las anteriores | Sin evidencia de que la arquitectura base sus decisiones en el fondo |

Las alertas se reportan junto con los resultados de los apartados 3 y 4. Una alerta de Grad-CAM en el mismo estrato donde aparece una señal de dependencia del tipo de fotografía refuerza esa limitación. Por sí sola, una alerta no invalida un resultado.

### 5.6 Presentación

Por arquitectura y estrato se reportan el porcentaje de pico dentro y la mediana de la concentración. Se incluyen láminas con ejemplos: el mismo conjunto de imágenes para las cuatro arquitecturas, con la caja dibujada.

---

## 6. Aplicación en la prueba piloto (Fase 4)

La prueba piloto (400 imágenes del conjunto de entrenamiento, 5 épocas) ejecuta el mismo procedimiento de Grad-CAM sobre 20 imágenes de **validación** (10 por especie, semilla 42), con cajas trazadas antes del piloto. Su propósito es verificar que la herramienta funciona en las cuatro arquitecturas (criterio de aceptación del plan) y obtener una lectura temprana.

El resultado del piloto es **descriptivo**. Si mostrara más del 50 % de picos fuera del espécimen en alguna arquitectura, se documentará y se consultará con el tutor antes de iniciar la Fase 5. Cualquier medida correctiva se decidirá y registrará antes del entrenamiento definitivo y nunca a la vista del conjunto de prueba.

---

## 7. Resumen de reglas de decisión

| Análisis | Indicador | Umbral | Consecuencia |
|---|---|---|---|
| Control Dryad + Ong | Δ_control (macro F1) | ≥ 0,05 con IC 95 % que excluye 0 | Limitación declarada y jerarquía de selección repetida sobre el control |
| Orden de arquitecturas | Macro F1 en prueba frente a control | Orden distinto | Conclusión principal con salvedad; se presentan ambos órdenes |
| Plataforma | M-ae y *Ae. albopictus* ecuatorianos | Mayoría mal clasificada | Lectura descriptiva de posible atajo por plataforma |
| Grad-CAM | % de picos fuera de la caja | > 25 % (total o estrato con ≥ 8) | Alerta de atención al fondo |
| Grad-CAM | Mediana de la concentración | ≤ 1 | Alerta de atención no localizada |
| Piloto | % de picos fuera de la caja | > 50 % | Documentar y consultar antes de la Fase 5 |

---

## 8. Salidas previstas

| Archivo | Contenido |
|---|---|
| `fase3/particion/particion.csv` | Imagen, especie, repositorio, grupo, subconjunto y marca de control o campo |
| `fase3/particion/gradcam_muestra.csv` | Muestra de Grad-CAM, fijada en la partición |
| `fase3/particion/gradcam_cajas.csv` | Cajas del espécimen, trazadas antes de entrenar |
| `fase6/metricas_por_estrato.csv` | Métricas por arquitectura y subconjunto, con IC |
| `fase6/control_dryad_ong.json` | Δ_control, Δ_campo, IC y orden de arquitecturas |
| `fase6/gradcam_indicadores.csv` | Pico dentro y concentración por imagen y arquitectura |
| `fase6/gradcam_laminas/` | Láminas de ejemplo |
