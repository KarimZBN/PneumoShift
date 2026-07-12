"""
Avalia a RSNA balanceada 1:1 (todos os positivos + iguais negativos), repetida por semente.

Le as selecoes dados/splits/rsna_balanceado_seed<N>.csv e consulta o cache de scores
(processed/rsna/scores_por_id.csv, gerado por avaliar_rsna_completa.py) — sem reinferencia
nem copia de imagens. Limiar 0,5. Reporta media +- desvio e IC 95% entre as sementes.

Saida em resultados/APOIO_rsna-balanceado/: rodadas.csv, resumo.json, LEIA.md.

Uso:
    python scripts/avaliar_rsna_balanceado.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv
import json
from datetime import datetime

import numpy as np

from pneumoshift import paths
from pneumoshift import scores as SC

# --- Configuracao ---
PREFIXO = "rsna_balanceado"
SEEDS = list(range(42, 52))
LIMIAR = 0.5

CACHE = paths.RSNA_FULL / "scores_por_id.csv"
SPLITS_DIR = paths.DADOS / "splits"
ROTULO = "balanceado 1:1 (todos os positivos + iguais negativos)"


def media_ic(valores):
    a = np.array(valores, float)
    return {"media": float(a.mean()),
            "desvio": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "ic95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]}


def main():
    if not CACHE.is_file():
        raise FileNotFoundError(
            f"Cache ausente: {CACHE}. Rode avaliar_rsna_completa.py antes.")
    selecoes = {s: SPLITS_DIR / f"{PREFIXO}_seed{s}.csv" for s in SEEDS}
    faltando = [s for s, p in selecoes.items() if not p.is_file()]
    if faltando:
        raise FileNotFoundError(
            f"Selecoes ausentes para as seeds {faltando}. Rode split_rsna.py antes.")

    cache = SC.ler_cache(CACHE)
    print(f"Cache: {len(cache)} pacientes | {ROTULO} | limiar {LIMIAR} | {len(SEEDS)} seeds\n")

    coletas = {"auc": [], "sensibilidade": [], "especificidade": [],
               "acuracia": [], "f1": [], "brier": []}
    linhas = []
    n_por_seed = None

    for seed in SEEDS:
        ids = SC.ler_selecao(selecoes[seed])
        m = SC.metricas_subconjunto(cache, ids, LIMIAR)
        n_por_seed = m["n"]
        for k in coletas:
            coletas[k].append(m[k])
        linhas.append([seed, f"{m['auc']:.4f}", f"{m['sensibilidade']:.4f}",
                       f"{m['especificidade']:.4f}", f"{m['acuracia']:.4f}",
                       f"{m['f1']:.4f}", f"{m['brier']:.4f}",
                       m["tp"], m["tn"], m["fp"], m["fn"]])
        print(f"  seed {seed}: AUC={m['auc']:.4f} sens={m['sensibilidade']*100:.1f}% "
              f"espec={m['especificidade']*100:.1f}% Brier={m['brier']:.4f} (n={n_por_seed})")

    resumo = {"limiar": LIMIAR, "conjunto": PREFIXO, "n_seeds": len(SEEDS),
              "seeds": SEEDS, "n_por_seed": n_por_seed,
              "metricas": {k: media_ic(v) for k, v in coletas.items()}}

    out = paths.pasta_resultado("APOIO", "rsna-balanceado")
    with open(out / "rodadas.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["seed", "auc", "sensibilidade", "especificidade", "acuracia",
                    "f1", "brier", "tp", "tn", "fp", "fn"])
        w.writerows(linhas)
    with open(out / "resumo.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)

    m_auc = resumo["metricas"]["auc"]; m_sen = resumo["metricas"]["sensibilidade"]; m_esp = resumo["metricas"]["especificidade"]
    paths.escrever_leia(
        out, "RSNA — balanceado 1:1",
        "Amostra balanceada 1:1 (proporcao entre classes controlada), repetida por semente.",
        "Todos os positivos + igual numero de negativos sorteados, repetido em "
        f"{len(SEEDS)} sementes; limiar 0,5. Reporta media +- desvio e IC 95% entre seeds. "
        "O pareamento em quantidade nao torna as bases equivalentes (a classe negativa da "
        "RSNA e ausencia de opacidade anotada, nao exame normal).",
        {"rodadas.csv": "metricas por semente",
         "resumo.json": "media +- desvio + IC 95% entre as sementes"},
        {"n por seed": str(resumo['n_por_seed']),
         "AUC (media)": f"{m_auc['media']:.4f} +- {m_auc['desvio']:.4f}",
         "Sensibilidade (media)": f"{m_sen['media']*100:.2f}%",
         "Especificidade (media)": f"{m_esp['media']*100:.2f}%",
         "Seeds": str(len(SEEDS))})

    print(f"\n=== RESUMO ({len(SEEDS)} seeds, {ROTULO}, limiar {LIMIAR}) ===")
    for k, v in resumo["metricas"].items():
        m, dp, ic = v["media"], v["desvio"], v["ic95"]
        if k in ("auc", "brier"):
            print(f"  {k:14s}: {m:.4f} +- {dp:.4f}  IC95[{ic[0]:.4f}, {ic[1]:.4f}]")
        else:
            print(f"  {k:14s}: {m*100:.2f}% +- {dp*100:.2f}  IC95[{ic[0]*100:.1f}, {ic[1]*100:.1f}]")
    print(f"\nSaida: {out}")


if __name__ == "__main__":
    main()
