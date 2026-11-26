# Protocolo de criterios de calidad, etiquetado y selección

> **Producto de la Fase 2** (Tabla 4) — Diseño metodológico
> **Proyecto:** Comparación de arquitecturas de aprendizaje profundo para la clasificación de mosquitos vectores del dengue mediante reconocimiento de imágenes
> **Autor:** Leandro Steven Batioja Mena — PUCE Esmeraldas 2026–2027
> **Secciones del plan que formaliza:** 3.4, 3.6.1.2, 3.6.2.1, 3.6.3.3 y Fase 3 de 3.7
> **Versión:** 3 — actualizada tras la depuración del conjunto bruto (Fase 3): reoperacionalización de E4 y E5

---

## Propósito

Este protocolo operacionaliza los criterios declarados en el plan de investigación, estableciendo el orden de aplicación, los umbrales, las reglas de decisión y el registro exigido para cada uno. Su cumplimiento garantiza que la construcción del conjunto de datos sea reproducible y auditable por un tercero.

**Principio rector:** los criterios se fijan **antes** de examinar los datos y no se modifican en función de los resultados obtenidos. Cualquier ajuste posterior debe justificarse y registrarse en la bitácora de experimentos.

---

## 1. Criterios de inclusión

Una imagen se admite en el conjunto de trabajo si cumple **todas** las condiciones siguientes:

| # | Criterio | Verificación |
|---|---|---|
| I1 | Corresponde a un espécimen **adulto** de *Aedes aegypti* o *Aedes albopictus* | Etiqueta del repositorio y revisión visual |
| I2 | Posee **etiqueta de especie verificable** | Campo de metadatos del repositorio |
| I3 | Resolución **igual o superior a 100 × 100 píxeles** | Automática |
| I4 | Licencia de **acceso abierto** compatible con uso académico | Campo de licencia del repositorio |
| I5 | El archivo **se abre y decodifica** sin error | Automática |

---

## 2. Criterios de exclusión

Una imagen se descarta si cumple **cualquiera** de las condiciones siguientes. El orden de aplicación es el indicado, por eficiencia computacional:

| Orden | Código | Criterio | Umbral | Método |
|---|---|---|---|---|
| 1 | E1 | Archivo corrupto, ilegible o **truncado** | — | `PIL.Image.load()` sin tolerancia a truncamiento |
| 2 | E2 | Resolución insuficiente | < 100 × 100 px | Lectura de dimensiones |
| 3 | E3 | Duplicado exacto | Hash idéntico | MD5 sobre bytes del archivo |
| 4 | E4 | Duplicado perceptual | Similitud > 90 % en pHash de 256 bits **y** correlación ≥ 0,98 | pHash (`ImageHash`) con verificación por píxeles |
| 5 | E5 | Exposición deficiente | Rango dinámico (p99 − p1) < 40 | Percentiles del canal de luminancia (escala 0–255) |
| 6 | E6 | Desenfoque | Varianza del Laplaciano **< 10**, medida a 512 px de lado mayor | Ver sección 3 |
| 7 | E7 | Estadio no adulto (larva, pupa, huevo) | — | Etiqueta y revisión visual |
| 8 | E8 | Etiqueta ausente o ambigua | Duplicado **exacto** (MD5) entre especies | Metadatos y contraste entre especies |
| 9 | E9 | Licencia incompatible | — | Campo de licencia |

**Registro obligatorio:** cada exclusión se anota en el *Registro de imágenes eliminadas* (instrumento 3.6.2.1) indicando el identificador de la imagen, el repositorio de origen y el código del criterio aplicado.

---

## 3. Determinación del umbral de desenfoque

El umbral del operador Laplaciano **no se fija de forma arbitraria**, dado que la varianza depende del contenido de la imagen, del tipo de fondo, del tamaño del espécimen en el encuadre y de la resolución.

### Procedimiento

| Paso | Acción |
|---|---|
| 1 | Reducir cada imagen a un lado mayor de 512 px (sin ampliar las menores), para que la métrica sea comparable entre repositorios de distinta resolución |
| 2 | Calcular la varianza del Laplaciano para **todas** las imágenes del conjunto bruto |
| 3 | Obtener la distribución por especie, por repositorio y por especie × repositorio |
| 4 | Simular el porcentaje descartado por cada umbral candidato, por especie y por repositorio |
| 5 | Extraer una muestra visual estratificada por repositorio y percentil |
| 6 | Inspeccionar la muestra, así como la franja de valores que el umbral candidato eliminaría |
| 7 | Fijar el umbral, registrarlo en la bitácora y aplicarlo de forma uniforme a todo el conjunto |

### Regla de contención

Ningún umbral puede descartar **más del 30 %** de ninguna especie **ni de ningún repositorio**. La regla se aplica por repositorio porque las fuentes difieren en condiciones de captura: un umbral aceptable por especie puede eliminar una fuente completa.

### Resultado

| Concepto | Valor |
|---|---|
| Techo de contención | 30 (a partir de 50, Kaggle pierde el 42 %) |
| **Umbral adoptado** | **10** |
| Descarte en *Ae. aegypti* | 0 % (mínimo observado: 11,9) |
| Descarte en Mosquito Alert (*Ae. albopictus*) | 0,4 % |
| Descarte en Couret, Dryad, Kaggle y Ong | 0 % |

**Justificación.** La inspección visual mostró que la franja de varianza entre 5 y 10 contiene mayoritariamente imágenes no aptas para la morfología fina, ya sea por desenfoque o por el reducido tamaño del espécimen en el encuadre, aunque incluye también alguna imagen utilizable. Se adoptó 10 porque su costo recae exclusivamente sobre Mosquito Alert, fuente con un excedente amplio de *Ae. albopictus*, y es nulo sobre *Ae. aegypti*, la clase limitante.

### Limitaciones de la métrica

La varianza del Laplaciano confunde desenfoque con **uniformidad del fondo** y con **tamaño del espécimen en el encuadre**: imágenes nítidas sobre fondo uniforme presentan valores bajos. Por ello se emplea únicamente como filtro de degradación severa y no como medida de calidad. El problema del espécimen demasiado pequeño en el encuadre no se resuelve con este criterio y se reconoce como limitación.

**Riesgo de atajo asociado.** Las fuentes de *Ae. aegypti* son mayoritariamente primeros planos, mientras que Mosquito Alert contiene numerosas fotografías con el espécimen diminuto, casi todas de *Ae. albopictus*. Un modelo podría asociar el tamaño aparente del espécimen con la especie. Este riesgo se mitiga en parte por la inclusión de primeros planos de *Ae. albopictus* procedentes de Dryad y Ong, y se vigila mediante las métricas desagregadas por repositorio (medida M4) y los mapas de calor Grad-CAM.

### Archivos truncados

Durante el análisis se comprobó que los archivos JPEG truncados pueden decodificarse parcialmente, rellenando la zona faltante con gris, y superar así una verificación de integridad permisiva. Se presentan como imágenes con varianza extremadamente baja. Por ello el criterio E1 se aplica **sin tolerancia a truncamiento**: todo archivo incompleto se descarta como corrupto, no como desenfocado.

---

## 3 bis. Detección de duplicados y exposición: operacionalización

Durante la depuración se comprobó que las formulaciones iniciales de E4 y E5 producían falsos positivos masivos sobre las fuentes de laboratorio. Ambas se reoperacionalizaron.

### E4 — Duplicado perceptual

| Etapa | Procedimiento |
|---|---|
| Candidatos | pHash de **256 bits** (`hash_size=16`); similitud > 90 %, es decir distancia de Hamming ≤ 25 |
| Verificación | Correlación de Pearson ≥ **0,98** entre miniaturas de 64 × 64 píxeles en escala de grises, normalizadas |
| Agrupamiento | Solo se unen los pares **verificados**, lo que evita el encadenamiento transitivo |
| Representante | Entre los miembros que superan E5 y E6, el de mayor resolución; se desempata por varianza y por nombre más corto |

**Motivo.** Con un pHash de 64 bits y umbral de distancia ≤ 6, dos especímenes distintos fotografiados en el mismo montaje de laboratorio —mismo fondo, encuadre e iluminación— resultaban indistinguibles. Al agrupar por componentes conectados, cadenas de falsos positivos arrastraban decenas de imágenes. La verificación por correlación separa "la misma fotografía" de "el mismo montaje": un duplicado real alcanza correlación 1,00, mientras que dos ejemplares distintos en igual pose quedan en 0,90.

### E5 — Exposición

Se evalúa el **rango dinámico**, definido como la diferencia entre el percentil 99 y el percentil 1 del canal de luminancia, y no el brillo medio.

**Motivo.** El promedio global mide sobre todo el fondo. Una fotografía de microscopio correctamente expuesta sobre fondo blanco supera 235 de brillo medio aunque el espécimen conserve todo su detalle, de modo que el criterio original descartaba imágenes válidas. Una imagen realmente sobreexpuesta o subexpuesta pierde el rango tonal: las fotografías de laboratorio verificadas presentaron rangos de 184 a 201, frente a 14 en casos de exposición deficiente.

### Hallazgo metodológico

Tres criterios fallaron por la misma causa: **medían el fondo en lugar del espécimen**.

| Criterio | Métrica inicial | Qué medía en realidad |
|---|---|---|
| E6 | Varianza del Laplaciano global | Uniformidad del fondo y tamaño del espécimen |
| E4 | pHash de 64 bits | Composición general de la escena |
| E5 | Brillo medio | Proporción de fondo claro |

El conjunto reúne fotografías de microscopio con fondo controlado y fotografías ciudadanas con fondo natural, por lo que cualquier métrica global queda dominada por esa diferencia. Los criterios de calidad de imagen de uso habitual no son directamente transferibles a conjuntos heterogéneos y requieren verificación sobre las propias fuentes antes de su aplicación.

---

## 4. Verificación morfológica de etiquetas

| Parámetro | Valor |
|---|---|
| Cobertura mínima | 10 % de las imágenes de cada especie |
| Muestreo | Aleatorio estratificado por repositorio de origen |
| Clave de referencia | Gyawali et al. — clave morfológica de vectores del Pacífico |
| Rasgos examinados | Escudo torácico, bandas en articulaciones de las patas, coloración abdominal |
| Ejecutor | Investigador principal |


**Nota sobre el repositorio Couret.** Durante el análisis de nitidez se observaron ejemplares de tonalidad parda y sin bandas blancas nítidas en las patas, lo que inicialmente hizo sospechar de un error de etiquetado. La verificación posterior descartó esa hipótesis: los archivos corresponden a la cepa *New Orleans*, colonia de laboratorio de *Aedes aegypti* establecida en 1970 y empleada como referencia susceptible a insecticidas. La apariencia parda es propia de ejemplares de colonia secos, que pierden escamas. Se mantiene como observación de calidad que en estos ejemplares los rasgos diagnósticos se aprecian con dificultad, lo que se considerará al interpretar los resultados por repositorio.

### Plan de validación escalonado

Cuando la revisión morfológica no permita confirmar la etiqueta con certeza, se aplica el siguiente orden:

| Nivel | Acción |
|---|---|
| 1 | Consulta a profesional con formación en entomología o biología |
| 2 | Si no es posible: doble revisión documental por el investigador en dos momentos distintos, contrastando con al menos dos fuentes bibliográficas independientes |
| 3 | Si persiste la ambigüedad: exclusión de la imagen, con registro del caso y su justificación |

### Regla de decisión ante error de etiquetado

| Tasa de error detectada | Acción |
|---|---|
| ≤ 15 % | Continuar; corregir las etiquetas discrepantes |
| 15 % – 25 % | Ampliar la verificación al 20 % de las imágenes de ese repositorio antes de continuar |
| > 25 % | Evaluar la exclusión parcial o total del repositorio, con reconstrucción de la muestra |

**Resolución de discrepancias:** ante conflicto entre la etiqueta del repositorio y el criterio de revisión, prevalece el criterio morfológico documentado. Cada caso se registra con su justificación y fuente.

**Limitación reconocida:** el investigador no cuenta con formación especializada en entomología, por lo que una proporción de errores podría no ser detectada. Esta limitación se declara explícitamente en la interpretación de los resultados.

---

## 5. Control del sesgo por repositorio

Las colecciones presentan condiciones de captura heterogéneas. Sin control, un modelo podría alcanzar alto rendimiento distinguiendo la **fuente** de la imagen en lugar de la **especie**.

| Medida | Procedimiento | Umbral |
|---|---|---|
| M1 | Construir tabla cruzada especie × repositorio | Ninguna especie debe provenir en más del **70 %** de una única fuente |
| M2 | Si se supera el umbral, aplicar submuestreo sobre la colección sobredimensionada | — |
| M3 | Estratificar la división por especie **y** por repositorio | — |
| M4 | Calcular métricas desagregadas por repositorio en la evaluación | Una diferencia sustancial entre fuentes se reporta como limitación |

---

## 6. Prevención de fuga de datos

| Mecanismo | Qué detecta | Qué NO detecta |
|---|---|---|
| Hash MD5 | Archivos idénticos byte a byte | Fotos distintas del mismo espécimen |
| Hash perceptual (pHash) | Recortes, redimensionados, casi-duplicados | Fotos distintas del mismo espécimen |
| **Partición por grupo** | **Agrupa todas las imágenes de un mismo espécimen** | — |

**Identificador de agrupamiento:** campo de observación disponible en los metadatos de cada repositorio (por ejemplo, `observation_id` en iNaturalist).

**Regla:** todas las imágenes asociadas a una misma observación permanecen íntegramente en un único subconjunto — entrenamiento, validación o prueba — y nunca se reparten entre ellos.

**Implementación:** funciones de partición agrupada y estratificada de `scikit-learn`.

---

## 7. Umbral mínimo de muestra

| Parámetro | Valor |
|---|---|
| Umbral de referencia | 1.000 imágenes por especie tras depuración |
| Naturaleza | Criterio operativo, no mínimo teórico absoluto |
| Origen | Tamaños de conjunto reportados en los antecedentes revisados |
| Verificación | Análisis de la curva de aprendizaje durante la prueba piloto (Fase 4) |

**Si no se alcanza el umbral:** ampliar la recopilación consultando repositorios públicos adicionales **distintos de iNaturalist**, cuyo conjunto ecuatoriano se reserva exclusivamente para validación contextual.

**Prohibición expresa:** no se flexibilizarán los criterios de depuración establecidos en este protocolo para alcanzar el volumen previsto. Si aun ampliando las fuentes no se alcanza, la limitación se reporta explícitamente y se ajusta la interpretación de los resultados.

---

## 8. Secuencia completa de procesamiento

```
CONJUNTO BRUTO
      │
      ├─ 1. Verificación de integridad          (E1)
      ├─ 2. Filtro de resolución                (E2)
      ├─ 3. Eliminación de duplicados MD5       (E3)
      ├─ 4. Eliminación de duplicados pHash     (E4)
      ├─ 5. Filtro de exposición                (E5)
      ├─ 6. Análisis de nitidez → umbral        (sección 3)
      ├─ 7. Filtro de desenfoque                (E6)
      ├─ 8. Filtro de estadio y etiqueta        (E7, E8, E9)
      │
CONJUNTO DEPURADO
      │
      ├─ 9. Verificación morfológica (10 %)     (sección 4)
      ├─ 10. Tabla cruzada especie × repositorio (M1, M2)
      │
CONJUNTO VALIDADO
      │
      ├─ 11. Balanceo de clases
      ├─ 12. Partición agrupada y estratificada (sección 6)
      │
      ├─ 13. Aumento de datos ×2 (solo entrenamiento)
      │
CONJUNTO FINAL → entrenamiento / validación / prueba
```

---

## 9. Registros exigidos

| Instrumento | Sección del plan | Contenido |
|---|---|---|
| Ficha de selección y caracterización del conjunto | 3.6.1.2 | Repositorio, URL, licencia, número de imágenes por especie, formato, condiciones de captura |
| Registro de imágenes eliminadas | 3.6.2.1 | Identificador, repositorio, código del criterio aplicado |
| Ficha de revisión morfológica de etiquetas | 3.6.1.4 | Identificador, especie original, especie confirmada, criterio, fuente |
| Bitácora de experimentos | 3.6.2.2 | Umbral de desenfoque adoptado, decisiones, incidencias |
