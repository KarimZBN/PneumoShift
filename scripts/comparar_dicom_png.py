"""
Compara o DICOM bruto da RSNA com o PNG convertido e mede o aspect ratio das bases.

  (1) ORIENTACAO / CANTOS — reproduz em memoria a conversao do converter_rsna.py
      (min-max 0-255, TAMANHO ORIGINAL) e compara, pixel a pixel, com o PNG salvo em
      dados/test/rsna_pool. Igualdade indica que a conversao nao gira, espelha nem inverte
      a imagem, e que os cantos escuros vem do proprio dado (recorte do detector).
      Imprime tambem a intensidade media dos 4 cantos (bruto x PNG).

  (2) ASPECT RATIO — mede largura/altura das imagens da Kaggle (JPEG) e da RSNA
      (PNG) e reporta a fracao que foge de 1:1 (quadrado).

Uso:
    python src/comparar_dicom_png.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv
import random

import cv2
import numpy as np
import pydicom

from pneumoshift import paths

# --- Configuracao (alinhada ao converter_rsna.py) ---
SEED = 13
IMG_SIZE = (224, 224)     # (mantido por referencia; a conversao nao redimensiona)
N_COMPARAR = 200           # DICOMs para a verificacao de orientacao (bruto x PNG)
N_ASPECT = 400             # imagens por base para a medida de aspect ratio
CANTO = 20                 # tamanho (px) do quadrado de canto amostrado
TOL_QUADRADO = 0.02        # tolerancia p/ considerar "quadrado" (|AR-1| <= tol)

# --- Caminhos ---
DICOM_DIR = paths.DADOS / "raw" / "rsna" / "stage_2_train_images"
PNG_RSNA_DIR = paths.DADOS_TESTE / "rsna_pool"     # {NORMAL,PNEUMONIA}/*.png
KAGGLE_DIR = paths.DADOS_TESTE / "cxray"          # {NORMAL,PNEUMONIA}/*.jpeg
RESULTS_DIR = paths.RESULTADOS / "csv"


def converter_em_memoria(patient_id):
    """Reproduz a conversao do converter_rsna.py (min-max 0-255, TAMANHO ORIGINAL)."""
    ds = pydicom.dcmread(str(DICOM_DIR / f"{patient_id}.dcm"))
    arr = ds.pixel_array.astype(np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr = arr / arr.max() * 255.0
    return arr.astype(np.uint8)          # sem resize: geometria vai no pre-processamento


def intensidade_cantos(img):
    """Media de intensidade dos 4 cantos (superior-esq, sup-dir, inf-esq, inf-dir)."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    c = CANTO
    return {
        "sup_esq": float(g[:c, :c].mean()),
        "sup_dir": float(g[:c, -c:].mean()),
        "inf_esq": float(g[-c:, :c].mean()),
        "inf_dir": float(g[-c:, -c:].mean()),
    }


def achar_png(patient_id):
    """Localiza o PNG convertido do paciente em NORMAL ou PNEUMONIA."""
    for classe in ("NORMAL", "PNEUMONIA"):
        p = PNG_RSNA_DIR / classe / f"{patient_id}.png"
        if p.is_file():
            return p
    return None


def prova_orientacao():
    """Compara conversao-em-memoria x PNG salvo para a amostra que existe em ambos."""
    print("=" * 55)
    print("ORIENTACAO / CANTOS (DICOM bruto x PNG)")
    print("=" * 55)

    # Percorre os PNGs ja gerados e casa com o DICOM de origem.
    pngs = []
    for classe in ("NORMAL", "PNEUMONIA"):
        d = PNG_RSNA_DIR / classe
        if d.is_dir():
            pngs += [p.stem for p in d.iterdir() if p.suffix.lower() == ".png"]
    pngs = sorted(set(pngs))
    random.seed(SEED)
    random.shuffle(pngs)
    amostra = pngs[:N_COMPARAR]

    if not amostra:
        print("  Nenhum PNG encontrado em dados/test/rsna_pool — rode converter_rsna.py antes.")
        return

    identicos = divergentes = sem_dicom = 0
    somas = {k: [0.0, 0.0] for k in ("sup_esq", "sup_dir", "inf_esq", "inf_dir")}
    for pid in amostra:
        if not (DICOM_DIR / f"{pid}.dcm").is_file():
            sem_dicom += 1
            continue
        png = cv2.imread(str(achar_png(pid)), cv2.IMREAD_GRAYSCALE)
        memoria = converter_em_memoria(pid)
        if png.shape != memoria.shape:
            divergentes += 1
            continue
        if np.array_equal(png, memoria):
            identicos += 1
        else:
            divergentes += 1
        cb, cp = intensidade_cantos(memoria), intensidade_cantos(png)
        for k in somas:
            somas[k][0] += cb[k]
            somas[k][1] += cp[k]

    comparados = identicos + divergentes
    print(f"\n  Amostra: {len(amostra)} | comparados: {comparados} | sem DICOM: {sem_dicom}")
    print(f"  PNG identico a conversao-em-memoria: {identicos}/{comparados}")
    print(f"  Divergentes: {divergentes}/{comparados}")
    if comparados:
        print("\n  Intensidade media dos cantos (0=preto, 255=branco):")
        print(f"    {'canto':10s} {'DICOM bruto':>12s} {'PNG':>10s}")
        for k, (sb, sp) in somas.items():
            print(f"    {k:10s} {sb/comparados:12.1f} {sp/comparados:10.1f}")
    print("\n  100% identicos => o PNG e exatamente a saida do pipeline (sem giro/inversao);")
    print("  cantos escuros iguais em bruto e PNG => vem do dado.")


def _aspect_de(paths):
    """Devolve lista de aspect ratios (largura/altura) das imagens dadas."""
    ars = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape[:2]
        if h:
            ars.append(w / h)
    return ars


def _amostra_base(raiz, exts):
    arquivos = []
    for classe in ("NORMAL", "PNEUMONIA"):
        d = raiz / classe
        if d.is_dir():
            arquivos += [p for p in d.iterdir() if p.suffix.lower() in exts]
    arquivos = sorted(arquivos, key=lambda p: p.name)
    random.seed(SEED)
    random.shuffle(arquivos)
    return arquivos[:N_ASPECT]


def prova_aspect_ratio():
    print("\n" + "=" * 55)
    print("ASPECT RATIO DAS BASES")
    print("=" * 55)

    bases = {
        "Kaggle (Chest X-Ray)": _amostra_base(KAGGLE_DIR, (".jpeg", ".jpg", ".png")),
        "RSNA (PNG convertido)": _amostra_base(PNG_RSNA_DIR, (".png",)),
    }
    linhas = []
    for nome, paths in bases.items():
        ars = _aspect_de(paths)
        if not ars:
            print(f"\n  {nome}: nenhuma imagem encontrada.")
            continue
        ars = np.array(ars)
        nao_quadradas = int(np.sum(np.abs(ars - 1.0) > TOL_QUADRADO))
        pct = nao_quadradas / len(ars) * 100
        print(f"\n  {nome}  (n={len(ars)})")
        print(f"    AR medio: {ars.mean():.3f} | min: {ars.min():.3f} | max: {ars.max():.3f}")
        print(f"    Nao-quadradas (|AR-1| > {TOL_QUADRADO}): {nao_quadradas} ({pct:.1f}%)")
        linhas.append([nome, len(ars), f"{ars.mean():.4f}", f"{ars.min():.4f}",
                       f"{ars.max():.4f}", nao_quadradas, f"{pct:.2f}"])

    if linhas:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / "aspect_ratio_bases.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["base", "n", "ar_medio", "ar_min", "ar_max",
                        "nao_quadradas", "pct_nao_quadradas"])
            w.writerows(linhas)
        print(f"\n  Resumo salvo em: {out}")


def main():
    if not DICOM_DIR.is_dir():
        print(f"AVISO: pasta de DICOMs nao encontrada ({DICOM_DIR}). Prova 1 sera pulada.")
    else:
        prova_orientacao()
    prova_aspect_ratio()


if __name__ == "__main__":
    main()
