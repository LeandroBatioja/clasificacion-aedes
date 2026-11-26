"""
============================================================
 BIBLIOGRAFIA: VERIFICACION DE DOI, DESCARGA E INDICE
============================================================
 Comparacion de arquitecturas de aprendizaje profundo para
 la clasificacion de mosquitos vectores del dengue
 Leandro Steven Batioja Mena - PUCE Esmeraldas 2026-2027

 1. Verifica cada DOI del plan contra Crossref: comprueba que
    exista y que el titulo registrado coincida con el citado.
 2. Descarga los PDF de acceso abierto (Unpaywall, arXiv, PMLR, CVF).
 3. Nombra cada archivo con el numero de cita y el titulo:
       06_Potentials and limitations in the application of ....pdf
    Los archivos que ya existan con el prefijo del numero
    (06_Nolte_2025.pdf, 06 algo.pdf, ...) se renombran solos.
 4. Genera en bibliografia/:
       indice_bibliografia.md   estado de cada referencia
       verificacion_doi.md      resultado de la verificacion

 Se puede ejecutar varias veces. Para los PDF que bajes a mano
 basta con que el nombre empiece por el numero (p. ej. 09_x.pdf).

 USO (desde la raiz del proyecto)
   python descargar_bibliografia.py --email tu_correo@pucese.edu.ec
   python descargar_bibliografia.py --email ... --solo-verificar
============================================================
"""

import argparse
import difflib
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

# n, titulo citado en el plan, tipo, autor y fuente, doi, enlace, pdf directo, observacion
REFS = [
    (1, "Increasing transmission of dengue virus across ecologically diverse regions of Ecuador and associated risk factors", "art", "Katzelnick et al., 2024, PLoS Negl. Trop. Dis.", "10.1371/journal.pntd.0011408", "", "", ""),
    (2, "Spatial Modeling of the Potential Distribution of Dengue in the City of Manta, Ecuador", "art", "Lalangui-Vivanco et al., 2025, Int. J. Environ. Res. Public Health", "10.3390/ijerph22101521", "", "", ""),
    (3, "Early Detection of Dengue Outbreaks: Transmission Model Analysis of a Dengue Outbreak in a Remote Setting in Ecuador", "art", "Van Wyk et al., 2025, Epidemiology", "10.1097/EDE.0000000000001874", "", "", "Versión de autor (PMC). Agregar el DOI a la referencia del plan"),
    (4, "Variable performance of DNA barcoding and morphological characteristics for the identification of Arctic blacklegged Aedes (Diptera: Culicidae), with a focus on the Punctor subgroup", "art", "Villeneuve et al., 2024, Arthropod Syst. Phylogeny", "10.3897/ASP.82.E111985", "", "", ""),
    (5, "Automatic Classification of Mosquito Genera Using Transfer Learning", "art", "Pise, Patil y Pise, 2022, J. Theor. Appl. Inf. Technol.", "", "https://www.jatit.org", "", "Corregir en el plan: el artículo está en el vol. 100, n.º 6, no en el n.º 7"),
    (6, "Potentials and limitations in the application of Convolutional Neural Networks for mosquito species identification using wing images", "art", "Nolte et al., 2025, PLoS Comput. Biol.", "10.1371/journal.pcbi.1013435", "", "", ""),
    (7, "Deep Learning-Based Vector Mosquitoes Classification for Preventing Infectious Diseases Transmission", "art", "Asgari et al., 2022, Image Anal. Stereol.", "10.5566/ias.2804", "", "", ""),
    (8, "Advanced vision transformers and open-set learning for robust mosquito classification: A novel approach to entomological studies", "art", "Karim, Mahmud y Khan, 2024, PLoS Comput. Biol.", "10.1371/journal.pcbi.1012654", "", "", ""),
    (9, "Application of computer vision and deep learning models to automatically classify medically important mosquitoes in North Borneo, Malaysia", "art", "Ong et al., 2024, Bull. Entomol. Res.", "10.1017/S000748532400018X", "", "", ""),
    (10, "Augmenting community-driven vector surveillance with automated image classification: Lessons from the Artificial Intelligence Mosquito Alert (AIMA) system", "art", "Falk et al., 2025, Epidemics", "10.1016/j.epidem.2025.100863", "", "", ""),
    (11, "Kaggle: Your Machine Learning and Data Science Community", "web", "Kaggle Inc.", "", "https://www.kaggle.com", "", ""),
    (12, "Mosquito Alert", "web", "Mosquito Alert", "", "https://www.mosquitoalert.com", "", ""),
    (13, "Dryad", "web", "Dryad", "", "https://datadryad.org", "", ""),
    (14, "iNaturalist", "web", "iNaturalist", "", "https://www.inaturalist.org", "", ""),
    (15, "Analysis of inpatient data on dengue fever, malaria and leishmaniasis in Ecuador: A cross-sectional national study, 2015–2022", "art", "Acosta-España et al., 2024, New Microbes New Infect.", "10.1016/j.nmni.2024.101421", "", "", ""),
    (16, "Dengue hospitalizations and in-hospital mortality changes in trend in Ecuador: a nationwide study from 2015 to 2022", "art", "Lapo-Talledo, 2024, Infect. Dis.", "10.1080/23744235.2024.2341871", "", "", "NO DISPONIBLE: revista de suscripción; la biblioteca de la PUCE no tiene acceso"),
    (17, "A Deep Convolutional Neural Network for Classification of Aedes Albopictus Mosquitoes", "art", "Adhane, Dehshibi y Masip, 2021, IEEE Access", "10.1109/ACCESS.2021.3079700", "", "", ""),
    (18, "A Deep Learning-based Dengue Mosquito Detection Method Using Faster R-CNN and Image Processing Techniques", "art", "Siddiqua, Rahman y Uddin, 2021, Ann. Emerg. Technol. Comput.", "10.33166/AETiC.2021.03.002", "", "", ""),
    (19, "A Study on Intelligent Insect Image Recognition Based on Overlap Patch Relevance Extraction and Classification", "art", "Qian, 2025, Int. J. High Speed Electron. Syst.", "10.1142/S0129156425402803", "", "", ""),
    (20, "Dengue mosquito larvae identification using digital images", "art", "De Silva y Jayalal, 2020, SCSE", "10.1109/SCSE49731.2020.9313003", "", "", "Corregir en el plan: la conferencia es SCSE 2020, no ICITR"),
    (21, "Mosquito species identification accuracy of early deployed algorithms in IDX, A vector identification tool", "art", "Gupta et al., 2024, Acta Trop.", "10.1016/j.actatropica.2024.107392", "", "", ""),
    (22, "Aedes aegypti and Aedes albopictus (diptera: culicidae) in artificial breeding sites in rural settlements and indigenous villages", "art", "Almeida et al., 2025, Rev. Patol. Trop.", "10.5216/rpt.v53i4.78772", "", "", ""),
    (23, "A morphological identification key to the mosquito disease vectors of the Pacific", "art", "Gyawali et al., 2025, Aust. Entomol.", "10.1111/aen.70003", "", "", ""),
    (24, "Enhancing mosquito classification through self-supervised learning", "art", "Charoenpanyakul et al., 2024, Sci. Rep.", "10.1038/s41598-024-78260-2", "", "", ""),
    (25, "Gen-CNN: a framework for the automatic generation of CNNs for image classification", "art", "García-Aguirre et al., 2025, Neural Comput. Appl.", "10.1007/s00521-024-10398-6", "", "", ""),
    (26, "Analyzing Parameter-Efficient Convolutional Neural Network Architectures for Visual Classification", "art", "Shahadat y Maida, 2025, Sensors", "10.3390/s25247663", "", "", ""),
    (27, "ImageNet Large Scale Visual Recognition Challenge", "art", "Russakovsky et al., 2015, Int. J. Comput. Vis.", "10.1007/s11263-015-0816-y", "", "https://arxiv.org/pdf/1409.0575", "Versión de arXiv"),
    (28, "A review on the effectiveness of recent approaches to few-shot learning for image analysis", "art", "Asudani, Nagwani y Singh, 2025, Appl. Intell.", "10.1007/s10489-025-06676-y", "", "", ""),
    (29, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", "art", "Tan y Le, 2019, PMLR", "", "https://proceedings.mlr.press/v97/tan19a.html", "https://proceedings.mlr.press/v97/tan19a/tan19a.pdf", ""),
    (30, "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications", "art", "Howard et al., 2017, arXiv", "", "https://arxiv.org/abs/1704.04861", "https://arxiv.org/pdf/1704.04861", ""),
    (31, "Searching for MobileNetV3", "art", "Howard et al., 2019, arXiv", "", "https://arxiv.org/abs/1905.02244", "https://arxiv.org/pdf/1905.02244", ""),
    (32, "Deep Residual Learning for Image Recognition", "art", "He et al., 2016, CVPR", "10.1109/CVPR.2016.90", "", "https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf", "Versión de acceso abierto de CVF"),
    (33, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", "art", "Dosovitskiy et al., 2020, arXiv", "", "https://arxiv.org/abs/2010.11929", "https://arxiv.org/pdf/2010.11929", ""),
    (34, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization", "art", "Selvaraju et al., 2017, ICCV", "10.1109/ICCV.2017.74", "", "https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf", "Versión de acceso abierto de CVF"),
    (35, "TransembleNet: Enhancing vector mosquito species classification through transfer learning-based ensemble model", "art", "Al Maruf et al., 2025, PLoS One", "10.1371/journal.pone.0322171", "", "", ""),
    (36, "MosquitoSong+: A noise-robust deep learning model for mosquito classification from wingbeat sounds", "art", "Supratak et al., 2024, PLoS One", "10.1371/journal.pone.0310121", "", "", ""),
    (37, "A convolutional neural network to identify mosquito species (Diptera: Culicidae) of the genus Aedes by wing images", "art", "Sauer et al., 2024, Sci. Rep.", "10.1038/s41598-024-53631-x", "", "", ""),
    (38, "Constitución de la República del Ecuador", "norma", "Asamblea Constituyente, Montecristi, 2008", "", "https://corteconstitucional.gob.ec", "", ""),
    (39, "Ley Orgánica de Salud", "norma", "Congreso Nacional, 2006", "", "https://salud.gob.ec", "", ""),
    (40, "Ley Orgánica de Educación Superior", "norma", "Asamblea Nacional, 2010", "", "https://ces.gob.ec", "", ""),
    (41, "Ley Orgánica de Protección de Datos Personales", "norma", "Asamblea Nacional, 2021", "", "https://superintendenciadedatos.gob.ec", "", ""),
    (42, "Código Orgánico de la Economía Social de los Conocimientos, Creatividad e Innovación", "norma", "Asamblea Nacional, 2016", "", "https://senadi.gob.ec", "", ""),
    (43, "Acuerdo Ministerial 00001-2025 - Norma Técnica del Sistema Integrado de Vigilancia Epidemiológica (SIVE)", "norma", "Ministerio de Salud Pública, 2025", "", "https://salud.gob.ec", "", ""),
    (44, "ISO/IEC 5259-1:2024 - Data quality for analytics and machine learning - Part 1", "iso", "ISO, 2024", "", "https://www.iso.org/standard/81088.html", "", ""),
    (45, "ISO/IEC 25059:2023 - Quality model for AI systems", "iso", "ISO, 2023", "", "https://www.iso.org/standard/80655.html", "", ""),
    (46, "ISO/IEC 42001:2023 - Artificial intelligence management system", "iso", "ISO, 2023", "", "https://www.iso.org/standard/81230.html", "", ""),
    (47, "ISO/IEC 27001:2022 - Information security management systems", "iso", "ISO, 2022", "", "https://www.iso.org/standard/27001", "", ""),
    (48, "Integrated Management Strategy for the Prevention and Control of Arboviral Diseases", "norma", "OPS/PAHO, 2016", "", "https://www.paho.org", "", ""),
]

TIPO = {"art": "Artículo", "norma": "Normativa", "web": "Sitio web", "iso": "Norma ISO"}


def get(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def nombre_archivo(n, titulo, max_len=120):
    t = titulo.replace("/", "-").replace(":", " -")
    t = re.sub(r'[\\*?"<>|]', "", t)
    t = re.sub(r"\s+", " ", t).strip().rstrip(".")
    if len(t) > max_len:
        t = t[:max_len].rsplit(" ", 1)[0]
    t = t.rstrip(" ,;-–")
    return f"{n:02d}_{t}.pdf"


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


# ---------------- verificacion de DOI ----------------
def verificar_doi(doi, titulo, email):
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}?mailto={email}"
    try:
        msg = json.loads(get(url))["message"]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "NO EXISTE", "Crossref no registra este DOI", ""
        return "SIN VERIFICAR", f"Crossref respondió HTTP {e.code}", ""
    except Exception as e:
        return "SIN VERIFICAR", f"error de conexión ({type(e).__name__})", ""
    reg = (msg.get("title") or [""])[0]
    revista = (msg.get("container-title") or [""])[0]
    anio = ((msg.get("issued") or {}).get("date-parts") or [[None]])[0][0]
    detalle = f"{reg} — {revista} ({anio})"
    sim = difflib.SequenceMatcher(None, norm(titulo), norm(reg)).ratio()
    if sim >= 0.85:
        return "OK", detalle, reg
    return "TITULO DISTINTO", f"Crossref registra: {detalle} (similitud {sim:.2f})", reg


# ---------------- descarga ----------------
def bajar_pdf(url, destino):
    try:
        datos = get(url, timeout=90)
    except Exception as e:
        return f"error {type(e).__name__}"
    if not datos.startswith(b"%PDF"):
        return "la página no devolvió un PDF"
    destino.write_bytes(datos)
    return "ok"


def unpaywall(doi, email):
    try:
        info = json.loads(get(f"https://api.unpaywall.org/v2/{doi}?email={email}"))
    except Exception as e:
        return None, [], f"Unpaywall: {type(e).__name__}"
    urls = []
    for loc in [info.get("best_oa_location")] + (info.get("oa_locations") or []):
        if loc and loc.get("url_for_pdf") and loc["url_for_pdf"] not in urls:
            urls.append(loc["url_for_pdf"])
    landing = (info.get("best_oa_location") or {}).get("url") or f"https://doi.org/{doi}"
    return info.get("is_oa", False), urls, landing


def localizar_y_renombrar(carpeta, n, canonico):
    """Busca un PDF que empiece por el numero y lo renombra al nombre canonico."""
    destino = carpeta / canonico
    if destino.exists():
        return destino
    for p in sorted(carpeta.glob("*.pdf")):
        m = re.match(r"^(\d{1,2})[_\s.\-]", p.name)
        if m and int(m.group(1)) == n:
            p.rename(destino)
            print(f"      renombrado: {p.name} -> {canonico}")
            return destino
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True, help="correo para Crossref y Unpaywall")
    ap.add_argument("--carpeta", default="bibliografia")
    ap.add_argument("--solo-verificar", action="store_true")
    args = ap.parse_args()
    carpeta = Path(args.carpeta)
    carpeta.mkdir(exist_ok=True)

    # 1. Verificacion de DOI
    print("=== Verificación de DOI en Crossref ===")
    verif = {}
    for n, titulo, tipo, autor, doi, *_ in REFS:
        if not doi:
            continue
        estado, detalle, _ = verificar_doi(doi, titulo, args.email)
        verif[n] = (estado, detalle)
        print(f"[{n:02d}] {estado:<16} {doi}")
        time.sleep(0.5)
    escribir_verificacion(carpeta, verif)
    malos = [n for n, (e, _) in verif.items() if e != "OK"]
    print(f"\nDOI verificados: {len(verif)}   Con problema: {len(malos)} {malos if malos else ''}")
    if args.solo_verificar:
        return

    # 2. Descarga y renombrado
    print("\n=== Descarga ===")
    filas, pendientes = [], []
    for n, titulo, tipo, autor, doi, enlace, pdf, obs in REFS:
        canonico = nombre_archivo(n, titulo)
        enlace_ref = f"https://doi.org/{doi}" if doi else enlace
        ref = f"{autor.rstrip('.')}. *{titulo}*"
        estado_doi = verif.get(n, ("—", ""))[0]
        print(f"[{n:02d}] ", end="", flush=True)

        if tipo in ("web", "iso"):
            nota = "Se cita el sitio; no aplica PDF" if tipo == "web" else "Norma de pago; no se puede redistribuir"
            filas.append((n, ref, enlace_ref, TIPO[tipo], "—", estado_doi, nota))
            print(TIPO[tipo].lower())
            continue

        previo = localizar_y_renombrar(carpeta, n, canonico)
        if previo:
            filas.append((n, ref, enlace_ref, TIPO[tipo], previo.name, estado_doi, obs or "Incluido"))
            print("ya existe")
            continue

        if tipo == "norma":
            filas.append((n, ref, enlace_ref, TIPO[tipo], "Pendiente", estado_doi, "Descargar del sitio oficial"))
            pendientes.append((canonico, enlace_ref, "Documento oficial: descargar del sitio"))
            print("manual")
            continue

        if estado_doi in ("NO EXISTE", "TITULO DISTINTO") and not pdf:
            nota = "DOI erróneo: corregir en el plan y descargar a mano (ver verificacion_doi.md)"
            filas.append((n, ref, enlace_ref, TIPO[tipo], "Pendiente", estado_doi, nota))
            pendientes.append((canonico, enlace_ref, nota))
            print("DOI con problema")
            continue

        candidatos, motivo, landing = ([pdf] if pdf else []), "", enlace_ref
        if doi and not pdf:
            es_oa, urls, landing = unpaywall(doi, args.email)
            candidatos += urls
            if es_oa is None:
                motivo, landing = landing, enlace_ref
            elif not es_oa:
                motivo = "suscripción"
            time.sleep(1)

        destino = carpeta / canonico
        res = "sin enlace a PDF"
        for u in candidatos:
            res = bajar_pdf(u, destino)
            if res == "ok":
                break
            time.sleep(1)

        if res == "ok":
            filas.append((n, ref, enlace_ref, TIPO[tipo], canonico, estado_doi, obs or "Acceso abierto"))
            print("descargado")
            continue
        if obs.startswith("NO DISPONIBLE"):
            filas.append((n, ref, enlace_ref, TIPO[tipo], "No incluido", estado_doi, obs))
            print("no disponible (suscripción)")
            continue
        if not doi and not pdf:
            nota = "Sin DOI: descargar a mano desde el enlace"
        elif motivo == "suscripción":
            nota = "Suscripción: descargar con el acceso institucional de la PUCE"
        elif motivo:
            nota = f"No se pudo consultar ({motivo}); descargar a mano"
        else:
            nota = "Acceso abierto, pero el sitio bloquea la descarga automática; descargar a mano"
        if obs:
            nota += f". {obs}"
        filas.append((n, ref, enlace_ref, TIPO[tipo], "Pendiente", estado_doi, nota))
        pendientes.append((canonico, landing, nota))
        print(f"pendiente ({res})")

    escribir_indice(carpeta, filas, pendientes)
    incluidos = sum(1 for f in filas if f[4].endswith(".pdf"))
    print(f"\nIncluidos: {incluidos}   Pendientes: {len(pendientes)}")
    print(f"Índice: {carpeta / 'indice_bibliografia.md'}")
    print(f"Verificación: {carpeta / 'verificacion_doi.md'}")


def escribir_verificacion(carpeta, verif):
    L = ["# Verificación de los DOI de la bibliografía", "",
         "Cada DOI del plan se consultó en Crossref y se comparó el título registrado con el citado.", "",
         "| N.º | DOI | Estado | Detalle |", "|---|---|---|---|"]
    doi_de = {r[0]: r[4] for r in REFS}
    for n in sorted(verif):
        e, d = verif[n]
        L.append(f"| {n} | {doi_de[n]} | {e} | {d} |")
    (carpeta / "verificacion_doi.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def escribir_indice(carpeta, filas, pendientes):
    L = ["# Índice de la bibliografía del plan de investigación", "",
         "*Comparación de arquitecturas de aprendizaje profundo para la clasificación de mosquitos "
         "vectores del dengue mediante reconocimiento de imágenes. Leandro Steven Batioja Mena, "
         "PUCE Esmeraldas, 2026–2027.*", "",
         "Cada archivo lleva como prefijo el número de cita en el plan, seguido del título.", "",
         "| N.º | Referencia | DOI o enlace | Tipo | Archivo | DOI verificado | Observación |",
         "|---|---|---|---|---|---|---|"]
    for f in filas:
        L.append("| " + " | ".join(str(x) for x in f) + " |")
    if pendientes:
        L += ["", "## Pendientes de descarga manual", "",
              "Guardar cada PDF en esta carpeta con un nombre que empiece por su número "
              "(p. ej., `09_x.pdf`) y volver a ejecutar el script: lo renombra y actualiza el índice.", ""]
        for nombre, url, nota in pendientes:
            L.append(f"- **{nombre}** — {url} — {nota}")
    (carpeta / "indice_bibliografia.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
