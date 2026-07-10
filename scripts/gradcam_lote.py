"""
Gera Grad-CAM em lote sobre a amostra pareada, separando por categoria (VP/VN/FP/FN).

Reusa a mesma selecao de avaliar_lote.py (sorted + semente fixa, padding letterbox) e o
mesmo pre-processamento. Para cada imagem classifica a categoria pela matriz de confusao
e gera o overlay Grad-CAM; por padrao o foco e nos ERROS (FP/FN), com um numero
configuravel de acertos para comparacao.

Saida:
    resultados/<base>/<geometria>_<ts>_seed<N>/gradcam/<CATEGORIA>/<arquivo>.png
    resultados/<base>/<geometria>_<ts>_seed<N>/gradcam/indice.csv

Uso:
    python scripts/gradcam_lote.py      (ajuste BASE/GEOMETRIA e LIMITES no topo)
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
N_NORMAL = 100
N_PNEUMONIA = 100
# Quantos overlays gerar por categoria (0 = nenhum, None = todos). Foco nos erros.
LIMITES = {"FP": None, "FN": None, "VP": 15, "VN": 15}

# PNGs/JPEGs no tamanho original; a geometria e aplicada aqui (pipeline novo), identica
# para cxray e rsna. RSNA usa o pool de teste (pasta_dados resolve).
TEST_DIR = paths.pasta_dados(BASE)
PREPROC = GEOMETRIA

# Saida ISOLADA em resultados/gradcam/<base>_<geometria>_<timestamp>/, separada das pastas
# de execucao do avaliar_lote (que guardam so metricas/CSV/figuras leves). Cada rodada de
# Grad-CAM/foco tem a propria pasta e nao polui a analise.
from datetime import datetime as _dt
OUT_DIR = paths.RESULTADOS / "gradcam" / f"{BASE}_{GEOMETRIA}_{_dt.now():%Y%m%d-%H%M%S}"


def main():
    model = load_model(str(paths.MODELO), compile=False)
    owner, conv, camadas = gradcam.preparar(model)

    for cat in ("VP", "VN", "FP", "FN"):
        (OUT_DIR / cat).mkdir(parents=True, exist_ok=True)

    gerados = {"VP": 0, "VN": 0, "FP": 0, "FN": 0}
    contagem = {"VP": 0, "VN": 0, "FP": 0, "FN": 0}
    linhas = []

    metas = {"PNEUMONIA": N_PNEUMONIA, "NORMAL": N_NORMAL}
    for label in ("PNEUMONIA", "NORMAL"):
        folder = TEST_DIR / label
        real_pneu = (label == "PNEUMONIA")
        for nome in selecionar(folder, metas[label]):
            img = cv2.imread(str(folder / nome))
            if img is None:
                continue
            img224 = redimensionar_por_geometria(img, PREPROC)
            entrada = preparar_entrada(img224)

            pred = model.predict(entrada, verbose=0)[0]
            idx = int(np.argmax(pred))
            pred_pneu = (float(pred[0]) >= LIMIAR)
            cat = categoria(real_pneu, pred_pneu)
            contagem[cat] += 1
            linhas.append([nome, label, CLASS_NAMES[idx], f"{float(pred[0]):.4f}", cat])

            limite = LIMITES.get(cat)
            if limite is not None and gerados[cat] >= limite:
                continue
            heat = gradcam.gerar(entrada, idx, owner, conv, camadas)
            overlay = gradcam.sobrepor(img224, heat)
            painel = np.hstack([img224, overlay])
            cv2.imwrite(str(OUT_DIR / cat / f"{Path(nome).stem}.png"), painel)
            gerados[cat] += 1

    with open(OUT_DIR / "indice.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "classe_real", "classe_predita", "score_pneumonia", "categoria"])
        w.writerows(linhas)

    print(f"Execucao: {BASE}_{GEOMETRIA}")
    print(f"Contagem por categoria: {contagem}")
    print(f"Overlays gerados:       {gerados}")
    print(f"Saida: {OUT_DIR}")


if __name__ == "__main__":
    main()
