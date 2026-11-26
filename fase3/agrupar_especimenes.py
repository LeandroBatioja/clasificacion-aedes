"""
============================================================
 FASE 3 - AGRUPACION POR ESPECIMEN ANTES DE LA PARTICION
============================================================
 Comparacion de arquitecturas de aprendizaje profundo para
 la clasificacion de mosquitos vectores del dengue
 Leandro Steven Batioja Mena - PUCE Esmeraldas 2026-2027

 Version 4. Cambios respecto a la v3:
   - Ong: uniones manuales. La similitud automatica no separa
     del todo los casos (mismo mosquito con r = 0,86; mosquitos
     distintos con r = 0,90 cuando el recorte falla). El script
     propone, el investigador revisa los pares dudosos y anota en
     fase3/agrupacion/ong_uniones_manuales.csv los que son el
     mismo mosquito; el script los une en la siguiente ejecucion.
   - Ong: hojas de contacto (ong_contacto/) con todas las fotos en
     orden y su grupo, para revisar la especie completa.
   - Ong: numero de pares "casi" configurable (40 por defecto).

 Version 3. Cambios respecto a la v2:
   - Ong: la similitud se mide sobre un recorte centrado en el
     mosquito, no sobre la foto completa. En la v2 el fondo de
     piel uniforme hacia parecidas fotos de mosquitos distintos
     (grupo de 34 imagenes en albopictus).
   - Ong: dos fotos muy parecidas (r >= umbral alto) se unen
     aunque esten lejos en la numeracion (el mismo mosquito
     fotografiado en otra sesion, p. ej. 107-108 y 184).
   - Mosquito Alert: si ningun umbral alcanza una precision de
     0,95, la union por similitud NO se aplica (en la v2 se
     aplicaba con umbral 0,99 sin efecto). Se anade la columna
     reporte_verificable para que la particion pueda excluir las
     fotos cuyo reporte no se conoce.

 Version 2. Cambios respecto a la v1:
   - Ong: dos fotos se unen solo si se parecen Y sus numeros
     estan cerca (rafagas). La v1 unia por similitud global y
     encadenaba mosquitos distintos por el fondo de piel.
   - Ong: las laminas "casi" muestran solo pares que quedaron
     en grupos distintos (en la v1 muchos ya estaban unidos
     por encadenamiento).
   - Mosquito Alert: las fotos sin reporte en el CSV publico se
     unen por similitud visual, con un umbral calibrado sobre
     las fotos que si tienen reporte (verdad de referencia).

 PROPOSITO
 ---------
 Asigna a cada imagen del conjunto depurado un identificador
 de grupo, de modo que todas las fotografias de un mismo
 mosquito queden juntas en la particion y no haya fuga de datos.

 FUENTE DEL GRUPO SEGUN REPOSITORIO
 ----------------------------------
   kaggle, dryad, couret  especimen del CSV de procedencia
   mosquito_alert         reporte ciudadano (photos_clean.csv o
                          imgRef.txt); las fotos sin reporte se
                          unen por similitud calibrada
   ong                    similitud del recorte del mosquito +
                          cercania de numeracion

 USO (desde la raiz del proyecto)
 --------------------------------
   python fase3/agrupar_especimenes.py \
       --ma-csv fase3/agrupacion/mosquito_alert_photos_clean.csv
   Parametros de Ong:  --umbral-ong 0.80  --ventana-ong 10  --umbral-ong-alto 0.92
   Parametro de MA:    --umbral-ma auto   (o un valor, p. ej. 0.90)
                       'auto' = menor umbral con el que el vecino mas parecido
                       pertenece al mismo reporte en al menos el 95 % de los casos

 La primera ejecucion lee las ~14.000 fotos de Mosquito Alert y
 guarda sus miniaturas en cache; las siguientes son rapidas.

 SALIDAS (fase3/agrupacion/)
 ---------------------------
   especimenes.csv          una fila por imagen con su grupo
   reporte_agrupacion.json  resumen numerico y calibracion
   ong_grupos/  ong_grupos.txt   grupos formados en Ong
   ong_casi/                pares mas parecidos que quedaron separados
                            (el recuadro rojo es el recorte comparado)
   ong_contacto/            todas las fotos de Ong en orden, con su grupo

 ENTRADA OPCIONAL
 ----------------
   fase3/agrupacion/ong_uniones_manuales.csv
       especie,foto_a,foto_b,nota
       aegypti,123,187,mismo mosquito en otra sesion
   Cada fila une dos fotos de Ong (por su numero) en el mismo grupo.
   ma_enlaces/              uniones por similitud en Mosquito Alert
                            mas cercanas al umbral (las mas dudosas)
============================================================
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

AEGYPTI = "aedes_aegypti"
ALBOPICTUS = "aedes_albopictus"
LADO = 64          # miniatura para la correlacion
LAMINA = 200       # tamano de cada imagen en las laminas


# ------------------------------------------------------------
# utilidades
# ------------------------------------------------------------
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def stem_archivo(nombre_destino):
    """mosquito_alert__9dfa8fb3-....jpg -> 9dfa8fb3-...."""
    return Path(nombre_destino.split("__")[-1]).stem.lower()


def num_ong(nombre):
    m = re.search(r"(\d+)\.\w+$", nombre)
    return int(m.group(1)) if m else -1


def miniatura(ruta):
    try:
        with Image.open(ruta) as im:
            im.draft("L", (LADO * 4, LADO * 4))
            g = np.asarray(im.convert("L").resize((LADO, LADO), Image.BILINEAR), dtype=np.float32)
    except Exception:
        return np.zeros(LADO * LADO, dtype=np.float32)
    g = g.ravel() - g.mean()
    n = np.linalg.norm(g)
    return g / n if n else g


def componentes(mask):
    """Componentes conexas (8 vecinos) de una mascara pequena."""
    pend = set(zip(*np.nonzero(mask)))
    salida = []
    while pend:
        pila = [pend.pop()]
        comp = []
        while pila:
            y, x = pila.pop()
            comp.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (y + dy, x + dx)
                    if q in pend:
                        pend.remove(q)
                        pila.append(q)
        ys, xs = np.array(comp).T
        salida.append((ys, xs))
    return salida


def recorte_mosquito(ruta, lado=256):
    """
    Localiza el mosquito como la mancha oscura y compacta mas
    marcada respecto a su entorno (sombrero negro), descarta
    bordes de fondo que cruzan la imagen y devuelve:
      - la miniatura normalizada del recorte (vector LADO*LADO)
      - el recuadro en fracciones (x0, y0, x1, y1) para las laminas
    Si no encuentra una mancha valida, usa la imagen completa.
    """
    from PIL import ImageFilter
    try:
        with Image.open(ruta) as im:
            im.draft("L", (lado * 2, lado * 2))
            g = im.convert("L")
            g.thumbnail((lado, lado))
            a = np.asarray(g, dtype=np.float32)
            fondo = np.asarray(g.filter(ImageFilter.GaussianBlur(10)), dtype=np.float32)
    except Exception:
        return np.zeros(LADO * LADO, dtype=np.float32), (0, 0, 1, 1)
    h, w = a.shape
    D = fondo - a
    mask = D >= max(float(np.percentile(D, 99.3)), 8.0)
    mejor = None
    for ys, xs in componentes(mask):
        if len(ys) < 6:
            continue
        bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        if bh > 0.6 * h or bw > 0.6 * w:
            continue                         # borde de fondo, no mosquito
        peso = float(D[ys, xs].sum())
        if mejor is None or peso > mejor[0]:
            mejor = (peso, ys, xs, bh, bw)
    if mejor:
        _, ys, xs, bh, bw = mejor
        cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
        m = int(np.clip(max(bh, bw) * 1.8, 32, min(h, w)))
        y0 = int(np.clip(cy - m / 2, 0, h - m))
        x0 = int(np.clip(cx - m / 2, 0, w - m))
        caja = (x0 / w, y0 / h, (x0 + m) / w, (y0 + m) / h)
        sub = Image.fromarray(a[y0:y0 + m, x0:x0 + m].astype(np.uint8))
    else:
        caja = (0, 0, 1, 1)
        sub = Image.fromarray(a.astype(np.uint8))
    v = np.asarray(sub.resize((LADO, LADO), Image.BILINEAR), dtype=np.float32).ravel()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return (v / n if n else v), caja


def miniaturas(rutas, cache=None):
    if cache and cache.exists():
        datos = np.load(cache, allow_pickle=False)
        nombres = [str(x) for x in datos["nombres"]]
        if nombres == [str(r) for r in rutas]:
            return datos["X"]
    X = np.zeros((len(rutas), LADO * LADO), dtype=np.float32)
    for i, r in enumerate(rutas):
        X[i] = miniatura(r)
        if (i + 1) % 1000 == 0:
            print(f"    miniaturas {i + 1}/{len(rutas)}")
    if cache:
        np.savez(cache, X=X, nombres=np.array([str(r) for r in rutas]))
    return X


def lamina(rutas, titulo, destino, etiquetas=None, cajas=None):
    cols = 4 if len(rutas) <= 16 else 8
    filas = (len(rutas) + cols - 1) // cols
    hoja = Image.new("RGB", (cols * LAMINA, filas * LAMINA + 28), "white")
    d = ImageDraw.Draw(hoja)
    d.text((6, 6), titulo, fill="black")
    for i, r in enumerate(rutas):
        x, y = (i % cols) * LAMINA, 28 + (i // cols) * LAMINA
        try:
            with Image.open(r) as im:
                im = im.convert("RGB")
                im.thumbnail((LAMINA - 6, LAMINA - 22))
                hoja.paste(im, (x + 3, y + 3))
                if cajas:
                    c = cajas[i]
                    W, H = im.size
                    d.rectangle([x + 3 + c[0] * W, y + 3 + c[1] * H,
                                 x + 3 + c[2] * W, y + 3 + c[3] * H], outline="red", width=2)
        except Exception:
            d.text((x + 10, y + 60), "no legible", fill="red")
        txt = etiquetas[i] if etiquetas else Path(r).name.split("__")[-1]
        d.text((x + 4, y + LAMINA - 18), txt[:30], fill="black")
    hoja.save(destino, quality=85)


def limpiar(d):
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.jpg"):
        f.unlink()


# ------------------------------------------------------------
# Mosquito Alert: mapa imagen -> reporte
# ------------------------------------------------------------
def cargar_mapa_ma(ruta):
    """
    Acepta photos_clean.csv (id, photo, report_id, hide, uuid)
    o imgRef.txt de BioStudies (imgNmb, imgId, rprtId, ...).
    Devuelve {clave_en_minusculas: report_id}.
    """
    ruta = Path(ruta).expanduser()
    with open(ruta, newline="", encoding="utf-8", errors="replace") as fh:
        muestra = fh.read(4096)
        fh.seek(0)
        try:
            dialecto = csv.Sniffer().sniff(muestra, delimiters=",;\t")
        except csv.Error:
            dialecto = csv.excel
        lector = csv.DictReader(fh, dialect=dialecto)
        cols = {c.lower().strip(): c for c in (lector.fieldnames or [])}
        col_rep = cols.get("report_id") or cols.get("rprtid")
        if not col_rep:
            raise SystemExit(f"ERROR: {ruta.name} no tiene columna report_id ni rprtId. "
                             f"Columnas: {list(cols.values())}")
        mapa = {}
        for fila in lector:
            rep = (fila.get(col_rep) or "").strip()
            if not rep:
                continue
            claves = []
            for c in ("uuid", "imgid", "id"):
                if c in cols:
                    claves.append(fila.get(cols[c]) or "")
            if "photo" in cols:
                claves.append(Path(fila.get(cols["photo"]) or "").stem)
            if "imgnmb" in cols:
                v = (fila.get(cols["imgnmb"]) or "").strip()
                if v.isdigit():
                    claves.append(v.zfill(6))
            for k in claves:
                k = k.strip().lower()
                if k:
                    mapa[k] = rep
    return mapa, list(cols.values())


def agrupar_ma(filas, depurado, mapa, umbral_arg, salida):
    """Grupo por reporte; las fotos sin reporte se unen por similitud calibrada."""
    info = {"por_especie": {}}
    dir_e = salida / "ma_enlaces"
    limpiar(dir_e)

    idx_all = [i for i, f in enumerate(filas) if f["origen"] == "mosquito_alert"]
    reps = {i: mapa.get(stem_archivo(filas[i]["nombre_destino"])) for i in idx_all}
    info["imagenes"] = len(idx_all)
    for i in idx_all:
        filas[i]["reporte_verificable"] = "si" if reps[i] else "no"
    info["con_reporte"] = sum(1 for r in reps.values() if r)

    # --- miniaturas de todo Mosquito Alert (con cache) ---
    rutas_all = [depurado / filas[i]["especie"] / filas[i]["nombre_destino"] for i in idx_all]
    print(f"  Mosquito Alert: leyendo {len(rutas_all)} miniaturas (cache en {salida.name}/)")
    X_all = miniaturas(rutas_all, salida / "cache_miniaturas_ma.npz")
    pos = {i: k for k, i in enumerate(idx_all)}

    # --- vecino mas parecido de cada foto, dentro de su especie ---
    especie_de = np.array([filas[i]["especie"] for i in idx_all])
    vecino = np.full(len(idx_all), -1)
    r_vecino = np.full(len(idx_all), -1.0, dtype=np.float32)
    for esp in (AEGYPTI, ALBOPICTUS):
        ks = np.where(especie_de == esp)[0]
        if len(ks) < 2:
            continue
        Xe = X_all[ks]
        for ini in range(0, len(ks), 500):
            C = Xe[ini:ini + 500] @ Xe.T
            for f_b in range(C.shape[0]):
                C[f_b, ini + f_b] = -2.0
            j = C.argmax(axis=1)
            vecino[ks[ini:ini + 500]] = ks[j]
            r_vecino[ks[ini:ini + 500]] = C[np.arange(C.shape[0]), j]

    # --- calibracion: precision del vecino en fotos con reporte conocido ---
    rep_k = [reps[i] for i in idx_all]
    tam_rep_k = Counter(r for r in rep_k if r)
    obs = [(float(r_vecino[k]), rep_k[k] == rep_k[vecino[k]], tam_rep_k[rep_k[k]] > 1)
           for k in range(len(idx_all)) if rep_k[k] and vecino[k] >= 0 and rep_k[vecino[k]]]
    r_obs = np.array([o[0] for o in obs])
    ok_obs = np.array([o[1] for o in obs])
    multi_obs = np.array([o[2] for o in obs])
    tabla = []
    for t in np.round(np.arange(0.70, 0.995, 0.01), 2):
        sel = r_obs >= t
        if sel.sum() >= 30:
            sm = sel & multi_obs
            tabla.append({"umbral": float(t), "fotos": int(sel.sum()),
                          "precision": round(float(ok_obs[sel].mean()), 3),
                          "precision_reportes_multiples": round(float(ok_obs[sm].mean()), 3) if sm.any() else None})
    if umbral_arg == "auto":
        validos = [x["umbral"] for x in tabla if x["precision"] >= 0.95]
        umbral = min(validos) if validos else None
    elif umbral_arg == "no":
        umbral = None
    else:
        umbral = float(umbral_arg)
    sel = r_obs >= (umbral if umbral is not None else 9.0)
    info["calibracion"] = {
        "metodo": "vecino mas parecido; precision = vecino del mismo reporte",
        "fotos_con_reporte_evaluadas": len(obs),
        "umbral": round(umbral, 3) if umbral is not None else None,
        "aplicada": umbral is not None,
        "modo_umbral": umbral_arg,
        "precision_en_umbral": round(float(ok_obs[sel].mean()), 3) if sel.any() else None,
        "fotos_que_superan_umbral": int(sel.sum()),
        "tabla_umbral_precision": tabla,
    }

    # --- union por especie: reporte + vecino de las fotos sin reporte ---
    grupos, enlaces = {}, []
    for esp in (AEGYPTI, ALBOPICTUS):
        ks = [k for k in range(len(idx_all)) if especie_de[k] == esp]
        if not ks:
            continue
        loc = {k: n for n, k in enumerate(ks)}
        uf = UnionFind(len(ks))
        primero = {}
        for k in ks:
            if rep_k[k]:
                if rep_k[k] in primero:
                    uf.union(loc[primero[rep_k[k]]], loc[k])
                else:
                    primero[rep_k[k]] = k
        sin = [k for k in ks if not rep_k[k]]
        n_enl = 0
        for k in sin:
            v = vecino[k]
            if umbral is not None and v >= 0 and r_vecino[k] >= umbral:
                if uf.find(loc[k]) != uf.find(loc[v]):
                    n_enl += 1
                    enlaces.append((float(r_vecino[k]), idx_all[k], idx_all[v]))
                uf.union(loc[k], loc[v])

        miembros = defaultdict(list)
        for k in ks:
            miembros[uf.find(loc[k])].append(k)
        corto = esp.replace("aedes_", "")
        n_vis, tam = 0, []
        for m in miembros.values():
            r_m = sorted({rep_k[k] for k in m if rep_k[k]})
            con_sin = any(not rep_k[k] for k in m)
            if len(m) == 1 and not r_m:
                continue                     # foto suelta: queda individual
            if r_m:
                gid = f"ma_rep_{r_m[0]}"
                fuente = "reporte_ma+similitud" if con_sin else "reporte_ma"
            else:
                n_vis += 1
                gid = f"ma_vis_{corto}_{n_vis:04d}"
                fuente = "similitud_ma"
            for k in m:
                grupos[idx_all[k]] = (gid, fuente)
            tam.append(len(m))
        info["por_especie"][esp] = {
            "imagenes": len(ks),
            "sin_reporte": len(sin),
            "uniones_por_similitud": n_enl,
            "grupos_finales": len(miembros),
            "mayor_grupo": max(tam, default=1),
        }

    # laminas de las uniones mas dudosas (las mas cercanas al umbral)
    enlaces.sort()
    for j, (r, a, b) in enumerate(enlaces[:20], 1):
        ra = depurado / filas[a]["especie"] / filas[a]["nombre_destino"]
        rb = depurado / filas[b]["especie"] / filas[b]["nombre_destino"]
        lamina([ra, rb], f"enlace {j:02d}  r = {r:.3f}  (umbral {umbral})",
               dir_e / f"enlace{j:02d}_r{r:.3f}.jpg",
               etiquetas=["sin reporte" if not reps[a] else "con reporte",
                          "sin reporte" if not reps[b] else "con reporte"])

    # conflictos de especie en un mismo reporte
    esp_rep = defaultdict(set)
    for i in idx_all:
        if reps[i]:
            esp_rep[reps[i]].add(filas[i]["especie"])
    info["reportes_con_ambas_especies"] = sorted(r for r, s in esp_rep.items() if len(s) > 1)
    tam_rep = Counter(reps[i] for i in idx_all if reps[i])
    multi = {r: n for r, n in tam_rep.items() if n > 1}
    info["reportes_con_varias_fotos"] = len(multi)
    info["imagenes_en_esos_reportes"] = sum(multi.values())
    info["mayor_reporte"] = max(multi.values(), default=1)
    return grupos, info


# ------------------------------------------------------------
# Ong: similitud visual + cercania de numeracion
# ------------------------------------------------------------
def leer_uniones_manuales(ruta):
    """{especie_corta: [(a, b, nota), ...]} desde el CSV de uniones manuales."""
    uniones = defaultdict(list)
    if not ruta.exists():
        return uniones
    with open(ruta, newline="", encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            esp = (fila.get("especie") or "").strip().lower().replace("aedes_", "").replace("aedes ", "")
            try:
                a, b = int(fila["foto_a"]), int(fila["foto_b"])
            except (KeyError, TypeError, ValueError):
                continue
            uniones[esp].append((a, b, (fila.get("nota") or "").strip()))
    return uniones


def agrupar_ong(filas, depurado, umbral, ventana, umbral_alto, salida, n_casi=40):
    grupos, stats, lineas = {}, {}, []
    dir_g, dir_c, dir_h = salida / "ong_grupos", salida / "ong_casi", salida / "ong_contacto"
    limpiar(dir_g)
    limpiar(dir_c)
    limpiar(dir_h)
    ruta_man = salida / "ong_uniones_manuales.csv"
    if not ruta_man.exists():
        ruta_man.write_text("especie,foto_a,foto_b,nota\n", encoding="utf-8")
    manuales = leer_uniones_manuales(ruta_man)

    for esp in (AEGYPTI, ALBOPICTUS):
        idx = [i for i, f in enumerate(filas) if f["origen"] == "ong" and f["especie"] == esp]
        if not idx:
            continue
        idx.sort(key=lambda i: num_ong(filas[i]["nombre_destino"]))
        rutas = [depurado / esp / filas[i]["nombre_destino"] for i in idx]
        nums = np.array([num_ong(filas[i]["nombre_destino"]) for i in idx])
        rec = [recorte_mosquito(r) for r in rutas]
        X = np.stack([v for v, _ in rec])
        cajas = [c for _, c in rec]
        sin_recorte = sum(1 for c in cajas if c == (0, 0, 1, 1))
        C = X @ X.T
        np.fill_diagonal(C, -1.0)
        cerca = np.abs(nums[:, None] - nums[None, :]) <= ventana
        unir = ((C >= umbral) & cerca) | (C >= umbral_alto)

        uf = UnionFind(len(idx))
        for a, b in np.argwhere(np.triu(unir, 1)):
            uf.union(a, b)
        pos_num = {int(n): k for k, n in enumerate(nums)}
        aplicadas, no_halladas = 0, []
        for a, b, _ in manuales.get(esp.replace("aedes_", ""), []):
            if a in pos_num and b in pos_num:
                uf.union(pos_num[a], pos_num[b])
                aplicadas += 1
            else:
                no_halladas.append(f"{a}-{b}")
        miembros = defaultdict(list)
        for k in range(len(idx)):
            miembros[uf.find(k)].append(k)

        corto = esp.replace("aedes_", "")
        multi = sorted((sorted(m) for m in miembros.values() if len(m) > 1),
                       key=lambda m: nums[m[0]])
        for g, m in enumerate(multi, 1):
            gid = f"ong_{corto}_g{g:03d}"
            for k in m:
                grupos[idx[k]] = gid
            lista = ", ".join(str(nums[k]) for k in m)
            lineas.append(f"{gid}  ({len(m)})  {lista}")
            lamina([rutas[k] for k in m[:64]],
                   f"{gid}  ({len(m)} imagenes)  r >= {umbral} y ventana {ventana}, o r >= {umbral_alto}",
                   dir_g / f"{gid}.jpg", cajas=[cajas[k] for k in m[:64]])

        # hojas de contacto: todas las fotos en orden con su grupo
        etiqueta_g = {}
        for g, m in enumerate(multi, 1):
            for k in m:
                etiqueta_g[k] = f"g{g:03d}"
        POR_HOJA = 36
        for h, ini in enumerate(range(0, len(idx), POR_HOJA), 1):
            ks = list(range(ini, min(ini + POR_HOJA, len(idx))))
            lamina([rutas[k] for k in ks],
                   f"{corto}  hoja {h}  fotos {nums[ks[0]]}-{nums[ks[-1]]}",
                   dir_h / f"{corto}_hoja{h:02d}.jpg",
                   etiquetas=[f"{nums[k]}  {etiqueta_g.get(k, '')}" for k in ks])

        # pares mas parecidos que quedaron en grupos DISTINTOS
        raiz = np.array([uf.find(k) for k in range(len(idx))])
        tri = np.triu_indices(len(idx), 1)
        sep = raiz[tri[0]] != raiz[tri[1]]
        vals = C[tri]
        cand = np.where(sep)[0]
        top = cand[np.argsort(-vals[cand])][:n_casi]
        for j, t in enumerate(top, 1):
            a, b = tri[0][t], tri[1][t]
            dn = abs(int(nums[a]) - int(nums[b]))
            lamina([rutas[a], rutas[b]], f"casi {j:02d}  r = {vals[t]:.3f}  distancia {dn}",
                   dir_c / f"{corto}_casi{j:02d}_r{vals[t]:.3f}_d{dn}.jpg",
                   cajas=[cajas[a], cajas[b]])

        stats[esp] = {
            "imagenes": len(idx),
            "grupos_multiples": len(multi),
            "imagenes_en_grupos": sum(len(m) for m in multi),
            "mayor_grupo": max((len(m) for m in multi), default=1),
            "especimenes_resultantes": len(miembros),
            "fotos_sin_recorte": sin_recorte,
            "uniones_manuales_aplicadas": aplicadas,
            "uniones_manuales_no_halladas": no_halladas,
            "r_vecino_percentiles": {p: round(float(np.percentile(C.max(axis=1), p)), 3)
                                     for p in (10, 25, 50, 75, 90)},
        }
    (salida / "ong_grupos.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return grupos, stats


# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procedencia", default="fase3/procedencia_conjunto_bruto.csv")
    ap.add_argument("--depurado", default="fase3/conjunto_depurado")
    ap.add_argument("--ma-csv", required=True,
                    help="photos_clean.csv o imgRef.txt de Mosquito Alert")
    ap.add_argument("--umbral-ma", default="auto",
                    help="correlacion minima con su vecina para unir una foto de MA sin reporte; "
                         "'auto' = precision >= 0,95 en las fotos con reporte")
    ap.add_argument("--umbral-ong", type=float, default=0.80)
    ap.add_argument("--umbral-ong-alto", type=float, default=0.92,
                    help="correlacion con la que dos fotos de Ong se unen a cualquier distancia")
    ap.add_argument("--ventana-ong", type=int, default=10,
                    help="diferencia maxima de numeracion para unir dos fotos de Ong")
    ap.add_argument("--n-casi", type=int, default=40,
                    help="pares 'casi' por especie que se dibujan para revision")
    ap.add_argument("--salida", default="fase3/agrupacion")
    args = ap.parse_args()

    depurado = Path(args.depurado).expanduser()
    salida = Path(args.salida).expanduser()
    salida.mkdir(parents=True, exist_ok=True)

    presentes = {(e, f.name) for e in (AEGYPTI, ALBOPICTUS)
                 for f in (depurado / e).iterdir() if f.is_file()}
    filas = []
    with open(args.procedencia, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if (r["especie"], r["nombre_destino"]) in presentes:
                filas.append(r)
    print(f"Imagenes del conjunto depurado: {len(filas)} (en disco: {len(presentes)})")
    if len(filas) != len(presentes):
        print("  AVISO: hay imagenes en disco sin fila en el CSV de procedencia")

    # grupo base: procedencia o individual (la especie va en el nombre)
    for f in filas:
        corto = f["especie"].replace("aedes_", "")
        if f["especimen"] and f["origen"] != "mosquito_alert":
            f["grupo"] = f"{f['origen']}_{f['especimen']}"
            f["fuente_grupo"] = "procedencia"
        else:
            f["grupo"] = f"{f['origen']}_{corto}_{stem_archivo(f['nombre_destino'])}"
            f["fuente_grupo"] = "individual"

    mapa, columnas = cargar_mapa_ma(args.ma_csv)
    print(f"Metadatos de Mosquito Alert: {len(mapa)} claves. Columnas: {columnas}")
    g_ma, info_ma = agrupar_ma(filas, depurado, mapa, args.umbral_ma, salida)
    for i, (gid, fuente) in g_ma.items():
        filas[i]["grupo"], filas[i]["fuente_grupo"] = gid, fuente
    for f in filas:
        if f["origen"] == "mosquito_alert" and f["fuente_grupo"] == "individual":
            f["fuente_grupo"] = "individual_sin_reporte"

    g_ong, stats_ong = agrupar_ong(filas, depurado, args.umbral_ong, args.ventana_ong,
                                   args.umbral_ong_alto, salida, args.n_casi)
    for i, gid in g_ong.items():
        filas[i]["grupo"], filas[i]["fuente_grupo"] = gid, "similitud_ong"

    with open(salida / "especimenes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["especie", "nombre_destino", "origen", "grupo", "fuente_grupo",
                    "reporte_verificable"])
        for f in filas:
            w.writerow([f["especie"], f["nombre_destino"], f["origen"], f["grupo"], f["fuente_grupo"],
                        f.get("reporte_verificable", "na")])

    resumen = defaultdict(dict)
    for (o, e), n in Counter((f["origen"], f["especie"]) for f in filas).items():
        grupos = {f["grupo"] for f in filas if f["origen"] == o and f["especie"] == e}
        resumen[o][e] = {"imagenes": n, "grupos": len(grupos)}

    reporte = {
        "version": 4,
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_imagenes": len(filas),
        "total_grupos": len({(f["especie"], f["grupo"]) for f in filas}),
        "por_repositorio": resumen,
        "fuente_del_grupo": dict(Counter(f["fuente_grupo"] for f in filas)),
        "mosquito_alert": {"archivo_metadatos": Path(args.ma_csv).name, **info_ma},
        "ong": {"medida": "correlacion del recorte centrado en el mosquito",
                "umbral_correlacion": args.umbral_ong, "ventana_numeracion": args.ventana_ong,
                "umbral_alto_sin_ventana": args.umbral_ong_alto,
                "por_especie": stats_ong},
    }
    with open(salida / "reporte_agrupacion.json", "w", encoding="utf-8") as fh:
        json.dump(reporte, fh, indent=2, ensure_ascii=False)

    # consola
    print("\n" + "=" * 62)
    print("  RESULTADO")
    print("=" * 62)
    print(f"  {'Repositorio':<16}{'Especie':<12}{'Imagenes':>10}{'Grupos':>10}")
    for o in sorted(resumen):
        for e in (AEGYPTI, ALBOPICTUS):
            if e in resumen[o]:
                r = resumen[o][e]
                print(f"  {o:<16}{e.replace('aedes_', ''):<12}{r['imagenes']:>10}{r['grupos']:>10}")
    m, cal = info_ma, info_ma["calibracion"]
    print(f"\n  Mosquito Alert: {m['con_reporte']}/{m['imagenes']} con reporte; "
          f"reportes con ambas especies: {len(m['reportes_con_ambas_especies'])}")
    if cal["aplicada"]:
        print(f"    union por similitud APLICADA: umbral {cal['umbral']}, precision {cal['precision_en_umbral']}")
    else:
        print("    union por similitud NO aplicada: ningun umbral alcanza precision 0,95.")
        print("    Las fotos sin reporte quedan marcadas reporte_verificable = no.")
    print("    umbral  fotos  precision  (solo reportes con varias fotos)")
    for x in cal["tabla_umbral_precision"][::3]:
        print(f"    {x['umbral']:.2f}  {x['fotos']:>6}  {x['precision']:.3f}      {x['precision_reportes_multiples']}")
    for e, s in m["por_especie"].items():
        print(f"    {e.replace('aedes_', '')}: {s['sin_reporte']} sin reporte, "
              f"{s['uniones_por_similitud']} uniones por similitud, mayor grupo {s['mayor_grupo']}")
    for e, s in stats_ong.items():
        print(f"\n  Ong {e.replace('aedes_', '')}: {s['grupos_multiples']} grupos, "
              f"{s['imagenes_en_grupos']} imagenes agrupadas, mayor grupo {s['mayor_grupo']}, "
              f"sin recorte {s['fotos_sin_recorte']}, uniones manuales {s['uniones_manuales_aplicadas']}")
        if s["uniones_manuales_no_halladas"]:
            print(f"    AVISO: uniones manuales con fotos inexistentes: {s['uniones_manuales_no_halladas']}")
        print(f"    r con la foto mas parecida (percentiles): {s['r_vecino_percentiles']}")
    print(f"\n  Grupos de Ong: {salida / 'ong_grupos.txt'}")
    print(f"  Laminas: {salida / 'ong_grupos'}, {salida / 'ong_casi'}, {salida / 'ma_enlaces'}")
    print(f"  Archivo: {salida / 'especimenes.csv'}")


if __name__ == "__main__":
    main()
