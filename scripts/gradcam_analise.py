"""
Grad-CAM em lote + analise objetiva de foco.

Para a amostra pareada (mesma selecao de avaliar_lote), classifica cada imagem pela matriz
de confusao (VP/VN/FP/FN), gera o overlay Grad-CAM E mede onde a ativacao se concentra
(fracao na PERIFERIA da imagem vs no MIOLO central). Nao ha re-inferencia entre gerar a
figura e medir o foco: tudo na mesma passada.

Os overlays sao salvos com NOME SEQUENCIAL por categoria (VP_001.png, FP_001.png, ...),
e o indice.csv mapeia cada nome sequencial ao arquivo original,
a categoria, o score e as metricas de foco.

Medida de foco (geometrica, reprodutivel): fracao da massa do Grad-CAM na moldura externa
(MARGEM de cada lado) vs no miolo, descartando o padding letterbox preto. LIMITACAO:
"periferia" e geometrica (moldura da imagem), nao anatomica.

Saida (resultados/APOIO_explicabilidade/<base>_<geometria>/):
    <CATEGORIA>/<CAT>_<NNN>.png   overlays nomeados por categoria
    indice.csv                    nome_seq, arquivo_original, categoria, score, foco...
    foco_resumo.csv               media do foco por categoria (VP/VN/FP/FN)

Uso:
    python scripts/gradcam_analise.py     (ajuste BASE/GEOMETRIA e LIMITES no topo)
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
from pneumoshift.metrics import categoria, CLASS_NAMES, LIMIAR


# --- Configuracao ---
BASE = "rsna"                  # "cxray" ou "rsna"
GEOMETRIA = "padding"          # "padding" ou "esticar"
N_NORMAL = 234
N_PNEUMONIA = 390
# Quantos overlays gerar/analisar por categoria (0 = nenhum, None = todos).
LIMITES = {"FP": None, "FN": None, "VP": 15, "VN": 15}
MARGEM = 0.18                  # espessura da moldura de periferia (fracao de cada lado)

TEST_DIR = paths.pasta_dados(BASE)
PREPROC = GEOMETRIA
OUT_DIR = paths.RESULTADOS / "APOIO_explicabilidade" / f"{BASE}_{GEOMETRIA}"


def mascara_periferia(shape):
    """Moldura externa (True) x miolo central (False), por geometria fixa."""
    h, w = shape
    m = np.ones((h, w), bool)
    my, mx = int(h * MARGEM), int(w * MARGEM)
    m[my:h - my, mx:w - mx] = False
    return m


def metricas_foco(heatmap, img_gray):
    """Fracao da massa do heatmap na periferia e no miolo, restrito ao conteudo real."""
    conteudo = img_gray > 5                       # descarta padding letterbox preto
    hm = heatmap * conteudo
    total = float(hm.sum())
    if total <= 0:
        return 0.0, 0.0, 0
    perif = mascara_periferia(heatmap.shape)
    frac_perif = float(hm[perif & conteudo].sum()) / total
    frac_miolo = float(hm[(~perif) & conteudo].sum()) / total
    py, px = np.unravel_index(int(np.argmax(hm)), hm.shape)
    return frac_perif, frac_miolo, int(perif[py, px])


def main():
    model = load_model(str(paths.MODELO), compile=False)
    owner, conv, camadas = gradcam.preparar(model)

    for cat in ("VP", "VN", "FP", "FN"):
        (OUT_DIR / cat).mkdir(parents=True, exist_ok=True)

    gerados = {"VP": 0, "VN": 0, "FP": 0, "FN": 0}
    contagem = {"VP": 0, "VN": 0, "FP": 0, "FN": 0}
    por_cat = {"VP": [], "VN": [], "FP": [], "FN": []}
    linhas = []

    metas = {"PNEUMONIA": N_PNEUMONIA, "NORMAL": N_NORMAL}
    for label in ("PNEUMONIA", "NORMAL"):
        folder = TEST_DIR / label
        real_pneu = (label == "PNEUMONIA")
        selecionados = selecionar(folder, metas[label])
        print(f"  {label}: {len(selecionados)} imagens", flush=True)
        for nome in selecionados:
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
            contagem[cat] += 1

            limite = LIMITES.get(cat)
            if limite is not None and gerados[cat] >= limite:
                continue

            # Grad-CAM (uma vez) -> overlay + medida de foco, sincronizados
            heat = gradcam.gerar(entrada, idx, owner, conv, camadas)
            fp_, fm_, pico = metricas_foco(heat, gray)

            gerados[cat] += 1
            seq = f"{cat}_{gerados[cat]:03d}"
            overlay = gradcam.sobrepor(img224, heat)
            painel = np.hstack([img224, overlay])
            cv2.imwrite(str(OUT_DIR / cat / f"{seq}.png"), painel)

            por_cat[cat].append((fp_, fm_, pico))
            linhas.append([seq, nome, label, CLASS_NAMES[idx], f"{float(pred[0]):.4f}",
                           cat, f"{fp_:.4f}", f"{fm_:.4f}", pico])

    with open(OUT_DIR / "indice.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["nome_seq", "arquivo_original", "classe_real", "classe_predita",
                    "score_pneumonia", "categoria",
                    "frac_periferia", "frac_miolo", "pico_na_periferia"])
        w.writerows(linhas)

    with open(OUT_DIR / "foco_resumo.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["categoria", "n", "media_frac_periferia", "media_frac_miolo",
                    "pct_pico_na_periferia"])
        print(f"\nExecucao: {BASE}_{GEOMETRIA}  (moldura de periferia = {MARGEM:.0%} de cada lado)")
        print(f"{'cat':4s} {'n':>4s} {'periferia':>10s} {'miolo':>8s} {'pico_perif':>11s}")
        for cat in ("VP", "VN", "FP", "FN"):
            vals = por_cat[cat]
            if not vals:
                continue
            arr = np.array(vals, float)
            mp, mm, pk = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean() * 100
            w.writerow([cat, len(vals), f"{mp:.4f}", f"{mm:.4f}", f"{pk:.1f}"])
            print(f"{cat:4s} {len(vals):4d} {mp:10.3f} {mm:8.3f} {pk:10.1f}%")

    paths.escrever_leia(
        OUT_DIR, f"Explicabilidade (Grad-CAM + foco) — {BASE} {GEOMETRIA}",
        "Grad-CAM por categoria (VP/VN/FP/FN) e medida objetiva de onde a ativacao se concentra.",
        "Overlays nomeados por categoria (VP_001, FP_001, ...) e a fracao da ativacao na "
        "periferia da imagem vs no miolo. 'Periferia' e geometrica (moldura), nao anatomica.",
        {"<CAT>/<CAT>_NNN.png": "overlay Grad-CAM por categoria",
         "indice.csv": "mapa nome_seq -> arquivo original, categoria, score e foco",
         "foco_resumo.csv": "media do foco por categoria"})

    print(f"\nContagem por categoria: {contagem}")
    print(f"Overlays gerados:       {gerados}")
    print(f"Saida: {OUT_DIR}")


if __name__ == "__main__":
    main()
