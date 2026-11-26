"""
============================================================
 FASE 3 - REBALANCEO Y PARTICION DEL CONJUNTO (ENTRADA 008)
============================================================
 Comparacion de arquitecturas de aprendizaje profundo para
 la clasificacion de mosquitos vectores del dengue
 Leandro Steven Batioja Mena - PUCE Esmeraldas 2026-2027

 PROPOSITO
 ---------
 Produce la particion fija que usaran las cuatro arquitecturas
 y las muestras que exige el protocolo de evaluacion
 (fase2/protocolo_evaluacion.md, v1.0).

   1. Conjunto de trabajo menos exclusiones_007.csv
   2. Rebalanceo: iguala las especies retirando REPORTES
      COMPLETOS de Mosquito Alert de la especie mayoritaria
      (semilla 42). Las demas fuentes no se tocan.
   3. Particion 70/15/15 AGRUPADA por especimen (un grupo nunca
      se reparte entre subconjuntos) y ESTRATIFICADA por
      especie x repositorio. Dentro de cada estrato los grupos
      se ordenan por tamano (desempate aleatorio, semilla 42) y
      cada uno se asigna al subconjunto con mayor deficit
      respecto a su cuota.
   4. Marca control (Dryad + Ong) o campo (iNaturalist +
      Mosquito Alert).
   5. Muestra de Grad-CAM (protocolo, 5.1): 10 grupos de prueba
      por estrato D-ae, D-al, O-ae, O-al, I-ae y M-al, mas todos
      los grupos de M-ae; una imagen por grupo.
   6. Muestras del piloto (plan, Fase 4): 200 imagenes por
      especie del entrenamiento, proporcionales por repositorio y
      por grupos completos; 10 por especie de validacion para
      Grad-CAM (protocolo, 6).
   7. Verificaciones: ningun grupo en dos subconjuntos, ninguna
      exclusion presente, especies iguales, medida M1 por
      subconjunto.

 USO (desde ~/Imagenes/tesis)
 ---
   python fase3/particion_conjunto.py --simular   (solo informa)
   python fase3/particion_conjunto.py             (escribe)
   python fase3/particion_conjunto.py --enlaces   (ademas crea
       fase3/conjunto_final/{entrenamiento,validacion,prueba}/
       {aedes_aegypti,aedes_albopictus}/ con enlaces duros)

 SALIDAS (fase3/particion/)
 -------
   particion.csv              especie, nombre_destino, origen,
                              grupo, subconjunto, parte
   rebalanceo_retiradas.csv   reportes retirados al igualar
   gradcam_muestra.csv        muestra de Grad-CAM (prueba)
   piloto_entrenamiento.csv   400 imagenes del piloto
   piloto_gradcam.csv         20 imagenes de validacion
   reporte_particion.json     conteos, verificaciones y M1
============================================================
"""

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

SEMILLA = 42
PROPORCION = {"entrenamiento": 0.70, "validacion": 0.15, "prueba": 0.15}
SUBCONJUNTOS = list(PROPORCION)
CONTROL = {"dryad", "ong"}
ESPECIES = ("aedes_aegypti", "aedes_albopictus")
CORTO = {"aedes_aegypti": "ae", "aedes_albopictus": "al",
         "dryad": "D", "ong": "O", "inaturalist": "I", "mosquito_alert": "M"}
GRADCAM_POR_ESTRATO = 10
PILOTO_POR_ESPECIE = 200
PILOTO_GRADCAM_POR_ESPECIE = 10
UMBRAL_M1 = 70.0


def leer(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def estrato(f):
    return f"{CORTO[f['origen']]}-{CORTO[f['especie']]}"


def agrupar(filas):
    g = defaultdict(list)
    for f in filas:
        g[(f["especie"], f["grupo"])].append(f)
    return g


# ------------------------------------------------------------
def rebalancear(filas, rnd):
    n = Counter(f["especie"] for f in filas)
    mayor, menor = max(ESPECIES, key=lambda e: n[e]), min(ESPECIES, key=lambda e: n[e])
    exceso = n[mayor] - n[menor]
    retirados = []
    if exceso == 0:
        return filas, retirados, n[menor]
    grupos = agrupar([f for f in filas if f["especie"] == mayor and f["origen"] == "mosquito_alert"])
    claves = sorted(grupos)
    rnd.shuffle(claves)
    fuera = set()
    for k in claves:
        if exceso == 0:
            break
        if len(grupos[k]) <= exceso:
            fuera.add(k)
            exceso -= len(grupos[k])
            retirados.append({"especie": k[0], "grupo": k[1], "imagenes": len(grupos[k])})
    if exceso:
        # Ningun reporte restante cabe en el exceso: se retira el reporte
        # mas pequeno y la diferencia se compensa con observaciones de una
        # sola foto de la otra especie en la parte de campo.
        resto = sorted((k for k in claves if k not in fuera), key=lambda k: (len(grupos[k]), k))
        k = resto[0]
        fuera.add(k)
        retirados.append({"especie": k[0], "grupo": k[1], "imagenes": len(grupos[k])})
        deficit = len(grupos[k]) - exceso
        otros = agrupar([f for f in filas if f["especie"] == menor and f["origen"] not in CONTROL])
        unos = sorted(c for c, v in otros.items() if len(v) == 1)
        rnd.shuffle(unos)
        if len(unos) < deficit:
            raise SystemExit("ERROR: no se pudo igualar exactamente las especies")
        for c in unos[:deficit]:
            fuera.add(c)
            retirados.append({"especie": c[0], "grupo": c[1], "imagenes": 1})
        objetivo = n[menor] - deficit
    else:
        objetivo = n[menor]
    return [f for f in filas if (f["especie"], f["grupo"]) not in fuera], retirados, objetivo


def particionar(filas, rnd):
    por_estrato = defaultdict(list)
    for k, imgs in agrupar(filas).items():
        por_estrato[estrato(imgs[0])].append((k, imgs))
    asignacion = {}
    for est in sorted(por_estrato):
        grupos = por_estrato[est]
        rnd.shuffle(grupos)
        grupos.sort(key=lambda x: -len(x[1]))            # estable: conserva el desempate aleatorio
        total = sum(len(i) for _, i in grupos)
        cuota = {s: PROPORCION[s] * total for s in SUBCONJUNTOS}
        lleno = Counter()
        for k, imgs in grupos:
            s = max(SUBCONJUNTOS, key=lambda s: (cuota[s] - lleno[s]) / cuota[s])
            asignacion[k] = s
            lleno[s] += len(imgs)
    for f in filas:
        f["subconjunto"] = asignacion[(f["especie"], f["grupo"])]
        f["parte"] = "control" if f["origen"] in CONTROL else "campo"
    return filas


def muestra_por_grupos(filas, n_grupos, rnd):
    """Una imagen por grupo, n_grupos al azar (todos si hay menos)."""
    grupos = agrupar(filas)
    claves = sorted(grupos)
    rnd.shuffle(claves)
    elegidos = claves if n_grupos is None else claves[:n_grupos]
    return [rnd.choice(sorted(grupos[k], key=lambda f: f["nombre_destino"])) for k in elegidos]


def muestra_gradcam(filas, rnd):
    prueba = [f for f in filas if f["subconjunto"] == "prueba"]
    salida = []
    for est in ("D-ae", "D-al", "O-ae", "O-al", "I-ae", "M-al", "M-ae"):
        sub = [f for f in prueba if estrato(f) == est]
        n = None if est == "M-ae" else GRADCAM_POR_ESTRATO
        for f in muestra_por_grupos(sub, n, rnd):
            salida.append({**f, "estrato": est})
    return salida


def muestra_piloto(filas, rnd):
    entren = [f for f in filas if f["subconjunto"] == "entrenamiento"]
    salida = []
    for esp in ESPECIES:
        sub = [f for f in entren if f["especie"] == esp]
        por_origen = Counter(f["origen"] for f in sub)
        # cuotas proporcionales por repositorio (resto mayor)
        bruto = {o: PILOTO_POR_ESPECIE * c / len(sub) for o, c in por_origen.items()}
        cuota = {o: int(v) for o, v in bruto.items()}
        for o in sorted(bruto, key=lambda o: -(bruto[o] - cuota[o]))[:PILOTO_POR_ESPECIE - sum(cuota.values())]:
            cuota[o] += 1
        for o in sorted(cuota):
            grupos = agrupar([f for f in sub if f["origen"] == o])
            claves = sorted(grupos)
            rnd.shuffle(claves)
            tomadas = []
            for k in claves:
                if len(tomadas) >= cuota[o]:
                    break
                resto = cuota[o] - len(tomadas)
                imgs = sorted(grupos[k], key=lambda f: f["nombre_destino"])
                tomadas.extend(imgs if len(imgs) <= resto else rnd.sample(imgs, resto))
            salida.extend(tomadas)
    val = [f for f in filas if f["subconjunto"] == "validacion"]
    gradcam = []
    for esp in ESPECIES:
        gradcam.extend(muestra_por_grupos([f for f in val if f["especie"] == esp],
                                          PILOTO_GRADCAM_POR_ESPECIE, rnd))
    return salida, gradcam


# ------------------------------------------------------------
def escribir(p, filas, campos):
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", default="fase3/conjunto_trabajo.csv")
    ap.add_argument("--exclusiones", default="fase3/verificacion_morfologica/exclusiones_007.csv")
    ap.add_argument("--depurado", default="fase3/conjunto_depurado_v2")
    ap.add_argument("--salida", default="fase3/particion")
    ap.add_argument("--final", default="fase3/conjunto_final")
    ap.add_argument("--simular", action="store_true")
    ap.add_argument("--enlaces", action="store_true")
    args = ap.parse_args()
    sal = Path(args.salida)
    if not args.simular and (sal / "particion.csv").exists():
        raise SystemExit(f"ERROR: {sal / 'particion.csv'} ya existe. La particion es fija: "
                         "no se regenera. Si es imprescindible, renombre la carpeta y registrelo.")

    print("=" * 66)
    print("  REBALANCEO Y PARTICION DEL CONJUNTO (ENTRADA 008)")
    print("=" * 66)

    # 1. exclusiones
    trabajo = leer(args.conjunto)
    excl = {(r["especie"], r["nombre_destino"]) for r in leer(args.exclusiones)}
    filas = [f for f in trabajo if (f["especie"], f["nombre_destino"]) not in excl]
    print(f"  Conjunto de trabajo: {len(trabajo)}   exclusiones 007: {len(excl)}   quedan: {len(filas)}")

    # 2. rebalanceo
    rnd = random.Random(SEMILLA)
    filas, retirados, objetivo = rebalancear(filas, rnd)
    print(f"  Rebalanceo: {len(retirados)} grupos retirados "
          f"({sum(r['imagenes'] for r in retirados)} imagenes) -> {objetivo} por especie")

    # 3-4. particion
    filas = particionar(filas, random.Random(SEMILLA))

    # 5-6. muestras
    gradcam = muestra_gradcam(filas, random.Random(SEMILLA))
    piloto, piloto_gc = muestra_piloto(filas, random.Random(SEMILLA))

    # 7. verificaciones
    sub_de_grupo = defaultdict(set)
    for f in filas:
        sub_de_grupo[(f["especie"], f["grupo"])].add(f["subconjunto"])
    fugas = [k for k, v in sub_de_grupo.items() if len(v) > 1]
    presentes = [f for f in filas if (f["especie"], f["nombre_destino"]) in excl]
    n_esp = Counter(f["especie"] for f in filas)
    tabla = defaultdict(Counter)
    for f in filas:
        tabla[(f["subconjunto"], f["especie"])][f["origen"]] += 1

    print(f"\n  {'Subconjunto':<14}{'Especie':<18}" + "".join(f"{o:>15}" for o in
          ("dryad", "ong", "inaturalist", "mosquito_alert")) + f"{'Total':>8}{'M1 max':>9}")
    m1 = {}
    for s in SUBCONJUNTOS:
        for e in ESPECIES:
            c = tabla[(s, e)]
            tot = sum(c.values())
            mx = max(c.values()) * 100 / tot if tot else 0
            m1[f"{s}|{e}"] = round(mx, 1)
            print(f"  {s:<14}{e:<18}" + "".join(f"{c.get(o, 0):>15}" for o in
                  ("dryad", "ong", "inaturalist", "mosquito_alert")) + f"{tot:>8}{mx:>8.1f}%")
    tot_s = Counter(f["subconjunto"] for f in filas)
    print("\n  Proporciones: " + ", ".join(f"{s} {tot_s[s]} ({100 * tot_s[s] / len(filas):.1f} %)"
                                          for s in SUBCONJUNTOS))
    ctrl = Counter((f["subconjunto"], f["especie"]) for f in filas if f["parte"] == "control")
    print("  Control (Dryad+Ong) en prueba: " + ", ".join(
        f"{e} {ctrl[('prueba', e)]}" for e in ESPECIES))
    print(f"  Grad-CAM: {len(gradcam)} imagenes  " + str(dict(Counter(g['estrato'] for g in gradcam))))
    print(f"  Piloto: {len(piloto)} de entrenamiento " + str(dict(Counter(p['especie'] for p in piloto)))
          + f", {len(piloto_gc)} de validacion para Grad-CAM")

    errores = []
    if fugas:
        errores.append(f"{len(fugas)} grupos en mas de un subconjunto")
    if presentes:
        errores.append(f"{len(presentes)} imagenes excluidas presentes")
    if n_esp["aedes_aegypti"] != n_esp["aedes_albopictus"]:
        errores.append(f"especies desiguales: {dict(n_esp)}")
    m1_mal = {k: v for k, v in m1.items() if v > UMBRAL_M1}
    print(f"\n  Verificaciones: grupos repartidos {len(fugas)}, excluidas presentes {len(presentes)}, "
          f"especies {dict(n_esp)}, M1 > {UMBRAL_M1:.0f} %: {m1_mal or 'ninguno'}")
    if errores:
        raise SystemExit("ERROR: " + "; ".join(errores))

    if args.simular:
        print("\n  SIMULACION: no se escribio nada.")
        return

    sal.mkdir(parents=True, exist_ok=True)
    campos = ["especie", "nombre_destino", "origen", "grupo", "subconjunto", "parte"]
    escribir(sal / "particion.csv",
             sorted(filas, key=lambda f: (f["subconjunto"], f["especie"], f["origen"], f["nombre_destino"])),
             campos)
    escribir(sal / "rebalanceo_retiradas.csv", retirados, ["especie", "grupo", "imagenes"])
    escribir(sal / "gradcam_muestra.csv", gradcam, ["estrato"] + campos)
    escribir(sal / "piloto_entrenamiento.csv", piloto, campos)
    escribir(sal / "piloto_gradcam.csv", piloto_gc, campos)
    json.dump({
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "semilla": SEMILLA,
        "proporciones": PROPORCION, "protocolo": "fase2/protocolo_evaluacion.md v1.0",
        "imagenes_por_especie": objetivo, "exclusiones_007": len(excl),
        "rebalanceo": {"reportes_retirados": len(retirados),
                       "imagenes_retiradas": sum(r["imagenes"] for r in retirados)},
        "tabla": {f"{s}|{e}": dict(tabla[(s, e)]) for s in SUBCONJUNTOS for e in ESPECIES},
        "totales": dict(tot_s), "m1_max_por_subconjunto": m1,
        "grupos": len(sub_de_grupo), "grupos_repartidos": 0,
        "control_prueba": {e: ctrl[("prueba", e)] for e in ESPECIES},
        "gradcam": dict(Counter(g["estrato"] for g in gradcam)),
        "piloto": {"entrenamiento": dict(Counter(p["especie"] for p in piloto)),
                   "gradcam_validacion": len(piloto_gc)},
    }, open(sal / "reporte_particion.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n  Escrito en {sal}/")

    if args.enlaces:
        base, dest = Path(args.depurado), Path(args.final)
        if dest.exists() and any(dest.rglob("*.*")):
            raise SystemExit(f"ERROR: {dest} ya contiene archivos")
        for f in filas:
            d = dest / f["subconjunto"] / f["especie"]
            d.mkdir(parents=True, exist_ok=True)
            os.link(base / f["especie"] / f["nombre_destino"], d / f["nombre_destino"])
        print(f"  Enlaces duros creados en {dest}/ ({len(filas)})")


if __name__ == "__main__":
    main()
