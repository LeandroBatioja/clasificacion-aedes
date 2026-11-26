# Fase 2 — Configuración del entorno experimental

**Proyecto:** Comparación de arquitecturas de aprendizaje profundo para la clasificación de mosquitos vectores del dengue mediante reconocimiento de imágenes
**Autor:** Leandro Steven Batioja Mena — PUCE Esmeraldas 2026–2027
**Semanas:** S4–S6 del cronograma

---

## Objetivo de esta etapa

Producir los dos entregables de la Fase 2 declarados en la Tabla 4:

1. Entorno PyTorch + timm configurado y verificado
2. Ficha técnica de configuración experimental completada

Adicionalmente, este benchmark anticipa parte de la verificación prevista para la **Fase 4 (prueba piloto)**: contrastar empíricamente la estimación de la Tabla 2 antes de comprometer el presupuesto de cómputo.

---

## 1. Instalación

```bash
# Crear el entorno virtual
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Instalar dependencias (build CPU de PyTorch)
pip install -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple
```

> El índice `download.pytorch.org/whl/cpu` evita descargar las bibliotecas CUDA (~2 GB innecesarios), ya que el plan declara ejecución exclusiva en CPU.

---

## 2. Verificar el entorno

```bash
python verificar_entorno.py
```

**Qué comprueba:**

| Verificación | Por qué importa |
|---|---|
| Versiones de librerías | Se registran en la ficha técnica (sección 3.6.1.3) |
| Carga de las 4 arquitecturas | Confirma que los IDs de timm declarados en 3.3 existen |
| Número de parámetros | Debe coincidir con lo afirmado en 2.2.8 (~5,3 M / ~5,4 M / ~25,6 M / ~22 M) |
| Media y desviación de normalización | Cada modelo tiene las suyas; se declaran en 3.3 |
| Ajuste fino parcial | Verifica la estrategia de congelamiento de 3.2 |
| Pasada hacia adelante | Confirma salida de 2 clases |

**Salida:** `reporte_entorno.json`

---

## 3. Medir el rendimiento real

```bash
python benchmark_cpu.py
```

Opciones:

```bash
python benchmark_cpu.py --batch 32 --lotes 10   # medición más robusta
```

**Qué mide y extrapola:**

- Imágenes por segundo en inferencia y en entrenamiento
- Memoria RAM máxima por arquitectura
- Horas por corrida completa y horas totales de las 32 corridas
- Comparación frente a las 240 h disponibles
- Desviación respecto a la estimación teórica de la Tabla 2 (12 y 5 img/s)

**Salida:** `reporte_benchmark.json`

---

## 4. Interpretar el resultado

El script emite un veredicto automático:

| Veredicto | Acción |
|---|---|
| Cierra con margen (>15 %) | La Tabla 2 se sostiene. Conservar los parámetros |
| Cierra ajustado (<15 %) | Considerar reducir épocas o factor de aumento |
| No cierra | Aplicar las medidas correctivas de la Fase 4 y **actualizar la Tabla 2 con los valores medidos** |

> **Importante:** si los valores medidos difieren de los estimados, la Tabla 2 del documento debe actualizarse con los datos reales. El plan ya declara que la estimación es preliminar y que la prueba piloto la verificaría — cumplir esa declaración es parte del rigor comprometido.

---

## 5. Parámetros del plan que usan estos scripts

| Parámetro | Valor | Sección |
|---|---|---|
| Arquitecturas | EfficientNet-B0, MobileNetV3-Large, ResNet50, ViT-S/16 | 3.3 |
| Resolución de entrada | 224 × 224 px | 3.3 |
| Clases | 2 (*Aedes aegypti*, *Aedes albopictus*) | 1.1 |
| Épocas definitivas | 15 con detención temprana | 3.2 |
| Épocas de búsqueda | 10 sobre 30 % del conjunto | 3.2 |
| Corridas por arquitectura | 5 | 3.2 |
| Configuraciones de hiperparámetros | 3 por arquitectura | 3.2 |
| Imágenes de entrenamiento | ~10.800 (tras aumento ×2) | 3.4 |
| Horas disponibles Fase 5 | 240 | 3.7 |

Si alguno cambia en el documento, actualizar las constantes al inicio de `benchmark_cpu.py`.

---

## Archivos

```
requirements.txt          Dependencias con versiones fijadas
verificar_entorno.py      Verificación del entorno y las arquitecturas
benchmark_cpu.py          Medición de rendimiento y extrapolación
ficha_tecnica.md          Plantilla del instrumento 3.6.1.3
reporte_entorno.json      (generado)
reporte_benchmark.json    (generado)
```
