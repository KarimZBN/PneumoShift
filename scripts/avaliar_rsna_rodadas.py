"""
Avaliacao da RSNA com limiar definido em conjunto de validacao separado e repeticao
com multiplas sementes.

Fluxo (usa os splits versionados de split_rsna.py, convertidos por converter_rsna.py):

  1. Infere na VALIDACAO (rsna_validacao) e acha o limiar de Youden -> aplicado 1x.
  2. Infere no POOL de teste (rsna_pool) uma unica vez (scores por imagem).
  3. Faz N rodadas: cada seed sorteia uma amostra pareada 390/234 do pool e calcula as
     metricas com o limiar de Youden (e, para referencia, o limiar 0,5).
  4. Reporta MEDIA +- DESVIO entre as seeds (incerteza da ESCOLHA da amostra).

O limiar de Youden e otimo para a definicao de negativo da RSNA (Target=0 = ausencia
de anotacao de opacidade), nao um limiar de decisao clinica.

Saida:
    resultados/rsna/rodadas_<timestamp>/
        rodadas.csv        (uma linha por seed: metricas com Youden e com 0,5)
        resumo.json        (media +- desvio; limiar de Youden; n seeds)

Uso:
    python scripts/avaliar_rsna_rodadas.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv
import json
import random
from datetime import datetime

import numpy as np
from keras.models import load_model

from pneumoshift import paths
from pneumoshift import metrics as M
from pneumoshift.inferencia import inferir_pasta

# --- Configuracao ---
SEEDS = list(range(42, 52))     # 10 rodadas (42..51);
N_PNE, N_NORM = 390, 234        # amostra pareada por rodada (proporcao do teste)
GEOMETRIA = "padding"           # PNG no tamanho original; padding aplicado no pre-proc

VAL_DIR = paths.DADOS_TESTE / "rsna_validacao"
POOL_DIR = paths.DADOS_TESTE / "rsna_pool"


def amostra_pareada(indices_por_classe, seed):
    """Sorteia N_PNE indices de pneumonia + N_NORM de normal (reprodutivel por seed)."""
    rng = random.Random(seed)
    pne = sorted(indices_por_classe[1])
    norm = sorted(indices_por_classe[0])
    rng.shuffle(pne)
    rng.shuffle(norm)
    return pne[:N_PNE] + norm[:N_NORM]


def metricas_rodada(y_true, y_score, idx, limiar):
    yt, ys = y_true[idx], y_score[idx]
    return M.metricas_por_limiar(yt, ys, limiar), M.auc_sklearn(yt, ys)


def main():
    for d in (VAL_DIR, POOL_DIR):
        if not d.is_dir():
            raise FileNotFoundError(
                f"Pasta ausente: {d}. Rode split_rsna.py e converter_rsna.py antes.")

    model = load_model(str(paths.MODELO), compile=False)
    print(f"Modelo carregado de: {paths.MODELO}")

    # 1) VALIDACAO -> limiar de Youden
    print("\nInferindo na VALIDACAO...")
    _, yv_true, yv_score = inferir_pasta(model, VAL_DIR, GEOMETRIA)
    limiar_youden = M.youden(yv_true, yv_score)
    print(f"  Limiar de Youden (validacao): {limiar_youden:.4f}  (n={len(yv_true)})")

    # 2) POOL -> scores uma unica vez
    print("\nInferindo no POOL de teste...")
    _, yp_true, yp_score = inferir_pasta(model, POOL_DIR, GEOMETRIA)
    print(f"  Pool: {len(yp_true)} imagens "
          f"({int(yp_true.sum())} pneumonia + {int((yp_true==0).sum())} normal)")

    indices_por_classe = {1: np.where(yp_true == 1)[0].tolist(),
                          0: np.where(yp_true == 0)[0].tolist()}

    # 3) N rodadas
    linhas = []
    coletas = {"auc": [], "acuracia_y": [], "sensibilidade_y": [], "especificidade_y": [],
               "acuracia_05": [], "especificidade_05": []}
    print(f"\nRodando {len(SEEDS)} amostras pareadas ({N_PNE}/{N_NORM})...")
    for seed in SEEDS:
        idx = np.array(amostra_pareada(indices_por_classe, seed))
        m_y, auc = metricas_rodada(yp_true, yp_score, idx, limiar_youden)
        m_05, _ = metricas_rodada(yp_true, yp_score, idx, 0.5)
        coletas["auc"].append(auc)
        coletas["acuracia_y"].append(m_y["acuracia"])
        coletas["sensibilidade_y"].append(m_y["sensibilidade"])
        coletas["especificidade_y"].append(m_y["especificidade"])
        coletas["acuracia_05"].append(m_05["acuracia"])
        coletas["especificidade_05"].append(m_05["especificidade"])
        linhas.append([seed, f"{auc:.4f}",
                       f"{m_y['acuracia']:.4f}", f"{m_y['sensibilidade']:.4f}", f"{m_y['especificidade']:.4f}",
                       f"{m_05['acuracia']:.4f}", f"{m_05['especificidade']:.4f}"])
        print(f"  seed {seed}: AUC={auc:.4f} | Youden acc={m_y['acuracia']*100:.1f}% "
              f"espec={m_y['especificidade']*100:.1f}% | 0,5 espec={m_05['especificidade']*100:.1f}%")

    def med_dp(v):
        a = np.array(v)
        return float(a.mean()), float(a.std(ddof=1))

    resumo = {"limiar_youden": limiar_youden, "n_seeds": len(SEEDS),
              "seeds": SEEDS, "n_pne": N_PNE, "n_norm": N_NORM,
              "media_desvio": {k: {"media": med_dp(v)[0], "desvio": med_dp(v)[1]}
                               for k, v in coletas.items()}}

    out = paths.RESULTADOS / "rsna" / f"rodadas_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rodadas.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["seed", "auc", "acuracia_youden", "sensibilidade_youden",
                    "especificidade_youden", "acuracia_lim0.5", "especificidade_lim0.5"])
        w.writerows(linhas)
    with open(out / "resumo.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)

    print(f"\n=== RESUMO ({len(SEEDS)} seeds) ===")
    for k, v in coletas.items():
        m, dp = med_dp(v)
        print(f"  {k:18s}: {m*100:.2f}% +- {dp*100:.2f}" if "auc" not in k
              else f"  {k:18s}: {m:.4f} +- {dp:.4f}")
    print(f"\nSaida: {out}")


if __name__ == "__main__":
    main()
