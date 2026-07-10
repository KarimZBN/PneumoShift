"""
Mede objetivamente ONDE a ativacao do Grad-CAM se concentra, sem classificacao visual.

Medida geometrica reprodutivel: a fracao da massa do Grad-CAM que cai na PERIFERIA da
imagem (moldura externa: bordas, cantos, ombros, marcadores) versus no MIOLO central.
O padding letterbox preto (fundo) e descartado. Agrega media por categoria VP/VN/FP/FN.

LIMITACAO: "periferia" e geometrica, nao anatomica; nao afirma "fora do pulmao", e sim
"na moldura externa da imagem".

Saida:
    resultados/<base>/<geometria>_<ts>_seed<N>/gradcam/foco.csv        (por imagem)
    resultados/<base>/<geometria>_<ts>_seed<N>/gradcam/foco_resumo.csv (por categoria)

Uso:
    python scripts/analise_foco.py      (ajuste BASE/GEOMETRIA no topo)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv

import cv2
import numpy as np
from keras.models import load_model

from pneumoshift import paths, gradcam
from pneumoshift.preprocess import redimensionar_por_geometria, preparar_entrada
from pneumoshift.data import selecionar
from pneumoshift.metrics import categoria, LIMIAR


# --- Configuracao ---
BASE = "cxray"                  # "cxray" ou "rsna"
GEOMETRIA = "padding"          # "padding" ou "esticar"
N_NORMAL = 50
N_PNEUMONIA = 50
MARGEM = 0.18                  # espessura da moldura de periferia (fracao de cada lado)

# PNGs/JPEGs no tamanho original em ambas as bases; a geometria e aplicada aqui
# (pipeline novo), identica para cxray e rsna. RSNA usa o pool de teste.
TEST_DIR = paths.pasta_dados(BASE)
PREPROC = GEOMETRIA

# Saida ISOLADA em resultados/gradcam/<base>_<geometria>_<timestamp>/ (mesmo caminho das
# imagens do gradcam_lote), separada das pastas de execucao do avaliar_lote.
from datetime import datetime as _dt
OUT_DIR = paths.RESULTADOS / "gradcam" / f"{BASE}_{GEOMETRIA}_{_dt.now():%Y%m%d-%H%M%S}"


def mascara_conteudo(img_gray):
    """Pixels do conteudo real (descarta o padding letterbox preto das bordas)."""
    return img_gray > 5


def mascara_periferia(shape):
    """Moldura externa (True) x miolo central (False), por geometria fixa."""
    h, w = shape
    m = np.ones((h, w), bool)
    my, mx = int(h * MARGEM), int(w * MARGEM)
    m[my:h - my, mx:w - mx] = False
    return m


def metricas_foco(heatmap, img_gray):
    """Fracao da massa do heatmap na periferia e no miolo, restrito ao conteudo real."""
    conteudo = mascara_conteudo(img_gray)
    hm = heatmap * conteudo
    total = float(hm.sum())
    if total <= 0:
        return 0.0, 0.0, 0
    perif_mask = mascara_periferia(heatmap.shape)
    frac_perif = float(hm[perif_mask & conteudo].sum()) / total
    frac_miolo = float(hm[(~perif_mask) & conteudo].sum()) / total
    py, px = np.unravel_index(int(np.argmax(hm)), hm.shape)
    return frac_perif, frac_miolo, int(perif_mask[py, px])


def main():
    model = load_model(str(paths.MODELO), compile=False)
    owner, conv, camadas = gradcam.preparar(model)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    linhas = []
    por_cat = {}

    metas = {"PNEUMONIA": N_PNEUMONIA, "NORMAL": N_NORMAL}
    for label in ("PNEUMONIA", "NORMAL"):
        folder = TEST_DIR / label
        real_pneu = (label == "PNEUMONIA")
        for nome in selecionar(folder, metas[label]):
            img = cv2.imread(str(folder / nome))
            if img is None:
                continue
            img224 = redimensionar_por_geometria(img, PREPROC)
            gray = cv2.cvtColor(img224, cv2.COLOR_BGR2GRAY)
            entrada = preparar_entrada(img224)

            pred = model.predict(entrada, verbose=0)[0]
            idx = int(np.argmax(pred))
            pred_pneu = (float(pred[0]) >= LIMIAR)
            cat = categoria(real_pneu, pred_pneu)

            heat = gradcam.gerar(entrada, idx, owner, conv, camadas)
            fp_, fm_, pk_ = metricas_foco(heat, gray)

            linhas.append([nome, label, cat, f"{fp_:.4f}", f"{fm_:.4f}", pk_])
            por_cat.setdefault(cat, []).append((fp_, fm_, pk_))

    with open(OUT_DIR / "foco.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "classe_real", "categoria",
                    "frac_periferia", "frac_miolo", "pico_na_periferia"])
        w.writerows(linhas)

    with open(OUT_DIR / "foco_resumo.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["categoria", "n", "media_frac_periferia", "media_frac_miolo",
                    "pct_pico_na_periferia"])
        print(f"\nExecucao: {BASE}_{GEOMETRIA}  (moldura de periferia = {MARGEM:.0%} de cada lado)")
        print(f"{'cat':4s} {'n':>4s} {'periferia':>10s} {'miolo':>8s} {'pico_perif':>11s}")
        for cat in ("VP", "VN", "FP", "FN"):
            vals = por_cat.get(cat, [])
            if not vals:
                continue
            arr = np.array(vals, float)
            mp, mm, pk = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean() * 100
            w.writerow([cat, len(vals), f"{mp:.4f}", f"{mm:.4f}", f"{pk:.1f}"])
            print(f"{cat:4s} {len(vals):4d} {mp:10.3f} {mm:8.3f} {pk:10.1f}%")

    print(f"\nSaida: {OUT_DIR}/foco.csv e foco_resumo.csv")
    print("Nota: 'periferia' e geometrica (moldura externa), nao anatomica.")


if __name__ == "__main__":
    main()
