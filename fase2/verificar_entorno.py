"""
============================================================
 FASE 2 - VERIFICACION DEL ENTORNO EXPERIMENTAL
============================================================
 Comparacion de arquitecturas de aprendizaje profundo para
 la clasificacion de mosquitos vectores del dengue
 Leandro Steven Batioja Mena - PUCE Esmeraldas 2026-2027

 Que hace este script:
   1. Registra las versiones exactas del entorno
   2. Carga las cuatro arquitecturas con pesos de ImageNet-1k
   3. Confirma el numero de parametros de cada una
   4. Adapta la capa clasificadora a 2 clases
   5. Verifica la estrategia de ajuste fino parcial
   6. Genera un reporte para la ficha tecnica

 Uso:
   python verificar_entorno.py
============================================================
"""

import sys
import platform
import json
from datetime import datetime

REPORTE = {}

def seccion(titulo):
    print()
    print("=" * 62)
    print(f"  {titulo}")
    print("=" * 62)


# ------------------------------------------------------------
# 1. ENTORNO
# ------------------------------------------------------------
seccion("1. ENTORNO DE EJECUCION")

REPORTE["fecha_verificacion"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
REPORTE["python"] = sys.version.split()[0]
REPORTE["sistema"] = f"{platform.system()} {platform.release()}"
REPORTE["procesador"] = platform.processor() or "no detectado"

print(f"  Fecha              : {REPORTE['fecha_verificacion']}")
print(f"  Python             : {REPORTE['python']}")
print(f"  Sistema operativo  : {REPORTE['sistema']}")

# CPU real desde /proc/cpuinfo (Linux)
try:
    with open("/proc/cpuinfo") as f:
        for linea in f:
            if "model name" in linea:
                REPORTE["cpu"] = linea.split(":")[1].strip()
                break
except Exception:
    REPORTE["cpu"] = REPORTE["procesador"]
print(f"  CPU                : {REPORTE.get('cpu','?')}")

# RAM
try:
    with open("/proc/meminfo") as f:
        for linea in f:
            if "MemTotal" in linea:
                kb = int(linea.split()[1])
                REPORTE["ram_gb"] = round(kb / 1024 / 1024, 1)
                break
    print(f"  RAM total          : {REPORTE['ram_gb']} GB")
except Exception:
    pass


# ------------------------------------------------------------
# 2. LIBRERIAS
# ------------------------------------------------------------
seccion("2. VERSIONES DE LIBRERIAS")

libs = {}
faltantes = []

def check(nombre, importar, attr="__version__"):
    try:
        mod = __import__(importar)
        v = getattr(mod, attr, "?")
        libs[nombre] = str(v)
        print(f"  [OK]    {nombre:<22} {v}")
        return mod
    except ImportError:
        faltantes.append(nombre)
        print(f"  [FALTA] {nombre:<22} no instalado")
        return None

torch   = check("torch",        "torch")
tv      = check("torchvision",  "torchvision")
timm    = check("timm",         "timm")
sk      = check("scikit-learn", "sklearn")
sm      = check("statsmodels",  "statsmodels")
scipy   = check("scipy",        "scipy")
cv2     = check("opencv",       "cv2")
np      = check("numpy",        "numpy")
pd      = check("pandas",       "pandas")
check("Pillow",     "PIL", "__version__")
check("ImageHash",  "imagehash")
check("matplotlib", "matplotlib")

REPORTE["librerias"] = libs

if faltantes:
    print()
    print(f"  ATENCION: faltan {len(faltantes)} librerias -> {', '.join(faltantes)}")
    print("  Ejecute: pip install -r requirements.txt")
    if torch is None or timm is None:
        print("\n  torch y timm son indispensables. Verificacion detenida.")
        sys.exit(1)


# ------------------------------------------------------------
# 3. CONFIGURACION DE TORCH EN CPU
# ------------------------------------------------------------
seccion("3. CONFIGURACION DE PYTORCH")

print(f"  Version de torch   : {torch.__version__}")
print(f"  CUDA disponible    : {torch.cuda.is_available()}  (se espera False)")
print(f"  Hilos de CPU       : {torch.get_num_threads()}")

REPORTE["cuda_disponible"] = torch.cuda.is_available()
REPORTE["hilos_cpu"] = torch.get_num_threads()

if torch.cuda.is_available():
    print("  NOTA: se detecto GPU. El plan declara ejecucion en CPU.")
    print("        Si desea forzar CPU, use device='cpu' en el entrenamiento.")


# ------------------------------------------------------------
# 4. LAS CUATRO ARQUITECTURAS
# ------------------------------------------------------------
seccion("4. CARGA DE LAS CUATRO ARQUITECTURAS")

# Identificadores timm declarados en la seccion 3.3 del plan
ARQUITECTURAS = {
    "EfficientNet-B0":   "efficientnet_b0",
    "MobileNetV3-Large": "mobilenetv3_large_100",
    "ResNet50":          "resnet50",
    "ViT-S/16":          "vit_small_patch16_224",
}

NUM_CLASES = 2  # Aedes aegypti / Aedes albopictus

print(f"  Clases objetivo    : {NUM_CLASES} (Aedes aegypti, Aedes albopictus)")
print(f"  Resolucion entrada : 224 x 224")
print()
print(f"  {'Arquitectura':<20} {'ID timm':<24} {'Params':>12}  Estado")
print("  " + "-" * 74)

resultados = {}
for nombre, tid in ARQUITECTURAS.items():
    try:
        modelo = timm.create_model(tid, pretrained=True, num_classes=NUM_CLASES)
        modelo.eval()
        total = sum(p.numel() for p in modelo.parameters())
        entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)

        # Configuracion de preprocesamiento que timm asocia al modelo
        cfg = modelo.default_cfg
        resultados[nombre] = {
            "id_timm": tid,
            "parametros_total": total,
            "parametros_entrenables": entrenables,
            "media_normalizacion": [round(x, 4) for x in cfg.get("mean", [])],
            "desv_normalizacion": [round(x, 4) for x in cfg.get("std", [])],
            "resolucion_cfg": cfg.get("input_size", None),
            "estado": "OK",
        }
        print(f"  {nombre:<20} {tid:<24} {total/1e6:>9.2f} M  OK")
        del modelo
    except Exception as e:
        resultados[nombre] = {"id_timm": tid, "estado": f"ERROR: {e}"}
        print(f"  {nombre:<20} {tid:<24} {'--':>12}  ERROR")
        print(f"       -> {str(e)[:110]}")

REPORTE["arquitecturas"] = resultados


# ------------------------------------------------------------
# 5. NORMALIZACION POR ARQUITECTURA
# ------------------------------------------------------------
seccion("5. PARAMETROS DE NORMALIZACION (declarados en 3.3)")

print(f"  {'Arquitectura':<20} {'Media (RGB)':<26} Desv. estandar (RGB)")
print("  " + "-" * 74)
for nombre, r in resultados.items():
    if r["estado"] == "OK":
        m = str(r["media_normalizacion"])
        s = str(r["desv_normalizacion"])
        print(f"  {nombre:<20} {m:<26} {s}")


# ------------------------------------------------------------
# 6. AJUSTE FINO PARCIAL
# ------------------------------------------------------------
seccion("6. VERIFICACION DEL AJUSTE FINO PARCIAL")

print("  Estrategia declarada en 3.2: congelar capas iniciales y")
print("  entrenar unicamente los ultimos bloques y el clasificador.")
print()
print(f"  {'Arquitectura':<20} {'Total':>10} {'Entrenables':>13} {'% entrenable':>14}")
print("  " + "-" * 74)

congelado = {}
for nombre, tid in ARQUITECTURAS.items():
    if resultados[nombre]["estado"] != "OK":
        continue
    try:
        modelo = timm.create_model(tid, pretrained=True, num_classes=NUM_CLASES)

        # Congelar todo
        for p in modelo.parameters():
            p.requires_grad = False

        # Descongelar solo el clasificador (via API de timm)
        clasificador = modelo.get_classifier()
        for p in clasificador.parameters():
            p.requires_grad = True

        total = sum(p.numel() for p in modelo.parameters())
        entren = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
        pct = 100 * entren / total

        congelado[nombre] = {
            "total": total,
            "entrenables_solo_clasificador": entren,
            "porcentaje": round(pct, 4),
        }
        print(f"  {nombre:<20} {total/1e6:>8.2f}M {entren:>11,} {pct:>13.3f}%")
        del modelo
    except Exception as e:
        print(f"  {nombre:<20} ERROR: {str(e)[:50]}")

REPORTE["ajuste_fino_parcial"] = congelado


# ------------------------------------------------------------
# 7. PRUEBA DE PASADA HACIA ADELANTE
# ------------------------------------------------------------
seccion("7. PRUEBA FUNCIONAL (pasada hacia adelante)")

print("  Se verifica que cada modelo procese un lote y devuelva")
print("  dos salidas por imagen (una por clase).")
print()

lote = torch.randn(2, 3, 224, 224)  # 2 imagenes RGB de 224x224
for nombre, tid in ARQUITECTURAS.items():
    if resultados[nombre]["estado"] != "OK":
        continue
    try:
        modelo = timm.create_model(tid, pretrained=True, num_classes=NUM_CLASES)
        modelo.eval()
        with torch.no_grad():
            salida = modelo(lote)
        forma = tuple(salida.shape)
        ok = forma == (2, NUM_CLASES)
        print(f"  {nombre:<20} salida {str(forma):<12} {'OK' if ok else 'INESPERADO'}")
        del modelo
    except Exception as e:
        print(f"  {nombre:<20} ERROR: {str(e)[:60]}")


# ------------------------------------------------------------
# 8. GUARDAR REPORTE
# ------------------------------------------------------------
seccion("8. REPORTE PARA LA FICHA TECNICA")

archivo = "reporte_entorno.json"
with open(archivo, "w", encoding="utf-8") as f:
    json.dump(REPORTE, f, indent=2, ensure_ascii=False)

ok_count = sum(1 for r in resultados.values() if r["estado"] == "OK")
print(f"  Arquitecturas verificadas : {ok_count} de {len(ARQUITECTURAS)}")
print(f"  Reporte guardado en       : {archivo}")
print()
if ok_count == len(ARQUITECTURAS) and not faltantes:
    print("  RESULTADO: entorno verificado correctamente.")
    print("  Siguiente paso: ejecutar benchmark_cpu.py")
else:
    print("  RESULTADO: hay elementos pendientes. Revise los avisos.")
print("=" * 62)
