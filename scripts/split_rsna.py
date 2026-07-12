"""
Gera CSVs de selecao de subconjuntos da RSNA a partir da base inteira convertida.

Le os patientId disponiveis em dados/processed/rsna/{NORMAL,PNEUMONIA}/ e grava, por
semente, um CSV com os IDs selecionados de cada conjunto. NAO copia PNGs: os avaliadores
leem esses CSVs e consultam o cache de scores (scores_por_id.csv, gerado por
avaliar_rsna_completa.py), sem duplicar imagens em disco nem reinferir.

Dois conjuntos, cada um repetido em varias sementes (para media +- desvio):

  BALANCEADO 1:1 (controla a proporcao, tamanho grande):
    todos os positivos + igual numero de negativos sorteados.
    -> dados/splits/rsna_balanceado_seed<N>.csv

  PAREADO A CHEST X-RAY (mesmo tamanho e proporcao da base interna):
    390 pneumonia + 234 normal sorteados.
    -> dados/splits/rsna_pareado_seed<N>.csv

A base inteira (processed/rsna) e avaliada diretamente por avaliar_rsna_completa.

Determinismo: nomes ordenados antes do embaralhamento; cada semente gera a mesma selecao.

Uso:
    python scripts/split_rsna.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv
import random
from datetime import datetime

from pneumoshift import paths

# --- Configuracao ---
SEEDS = list(range(42, 52))     # 10 sementes (min. aceitavel = 5)
N_PAREADO_PNE, N_PAREADO_NORM = 390, 234   # mesmo tamanho/proporcao da Chest X-Ray

FULL_DIR = paths.RSNA_FULL
SPLITS_DIR = paths.DADOS / "splits"


def ids_da_pasta(classe):
    """Lista os patientId (stem dos PNG) de uma classe em processed/rsna, ordenados."""
    d = FULL_DIR / classe
    if not d.is_dir():
        raise FileNotFoundError(
            f"Pasta-fonte ausente: {d}. Rode antes: python scripts/converter_rsna.py")
    return sorted(p.stem for p in d.iterdir() if p.suffix.lower() == ".png")


def sortear(ids, n, seed):
    """Ordena, embaralha por semente e devolve os primeiros n (ou todos, se n is None)."""
    v = sorted(ids)
    random.Random(seed).shuffle(v)
    return v if n is None else v[:n]


def gravar_selecao(caminho, pos_sel, neg_sel):
    """Grava um CSV patientId,classe com os IDs selecionados (pneumonia + normal)."""
    with open(caminho, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["patientId", "classe"])
        for pid in pos_sel:
            w.writerow([pid, "PNEUMONIA"])
        for pid in neg_sel:
            w.writerow([pid, "NORMAL"])


def gerar_conjunto(prefixo, pos_ids, neg_ids, n_pos, n_norm):
    """Para cada seed, sorteia n_pos/n_norm e grava um CSV de selecao (sem copiar PNG)."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    composicao = []
    for seed in SEEDS:
        sel_pos = sortear(pos_ids, n_pos, seed)
        sel_neg = sortear(neg_ids, n_norm, seed + 1000)   # offset: nao correlacionar classes
        gravar_selecao(SPLITS_DIR / f"{prefixo}_seed{seed}.csv", sel_pos, sel_neg)
        composicao.append([seed, len(sel_pos), len(sel_neg), len(sel_pos) + len(sel_neg)])
        print(f"  seed {seed}: {len(sel_pos)} pneumonia + {len(sel_neg)} normal "
              f"-> {prefixo}_seed{seed}.csv")
    with open(SPLITS_DIR / f"{prefixo}_composicao.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["seed", "n_pneumonia", "n_normal", "total"])
        w.writerows(composicao)


def main():
    pos_ids = ids_da_pasta("PNEUMONIA")
    neg_ids = ids_da_pasta("NORMAL")
    print(f"Fonte processed/rsna -> PNEUMONIA: {len(pos_ids)} | NORMAL: {len(neg_ids)}")
    if len(neg_ids) < len(pos_ids):
        raise ValueError("Negativos insuficientes para balancear 1:1 com os positivos.")
    if len(pos_ids) < N_PAREADO_PNE or len(neg_ids) < N_PAREADO_NORM:
        raise ValueError("Positivos/negativos insuficientes para o conjunto pareado.")

    print(f"\n[1] BALANCEADO 1:1 (todos os {len(pos_ids)} positivos + iguais negativos):")
    gerar_conjunto("rsna_balanceado", pos_ids, neg_ids, None, len(pos_ids))

    print(f"\n[2] PAREADO A CHEST X-RAY ({N_PAREADO_PNE} pneumonia + {N_PAREADO_NORM} normal):")
    gerar_conjunto("rsna_pareado", pos_ids, neg_ids, N_PAREADO_PNE, N_PAREADO_NORM)

    with open(SPLITS_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(
            f"# Selecoes amostradas da RSNA (proveniencia)\n\n"
            f"Gerado por `scripts/split_rsna.py` em {datetime.now():%Y-%m-%d %H:%M}. "
            f"Sementes: {SEEDS}. Sao CSVs de patientId (sem copia de imagem); os avaliadores "
            f"consultam o cache `processed/rsna/scores_por_id.csv`.\n\n"
            f"## Balanceado 1:1\n"
            f"- Todos os {len(pos_ids)} positivos + {len(pos_ids)} negativos sorteados por semente.\n"
            f"- `rsna_balanceado_seed<N>.csv`. Composicao: `rsna_balanceado_composicao.csv`.\n"
            f"- Avaliado por `avaliar_rsna_balanceado.py`.\n\n"
            f"## Pareado a Chest X-Ray\n"
            f"- {N_PAREADO_PNE} pneumonia + {N_PAREADO_NORM} normal por semente "
            f"(mesmo n e proporcao da base interna).\n"
            f"- `rsna_pareado_seed<N>.csv`. Composicao: `rsna_pareado_composicao.csv`.\n"
            f"- Avaliado por `avaliar_rsna_pareado.py`.\n\n"
            f"## Base inteira\n"
            f"- `dados/processed/rsna/` ({len(pos_ids)} pneumonia + {len(neg_ids)} normal), "
            f"lida por `avaliar_rsna_completa.py` (que gera o cache de scores).\n"
        )

    print(f"\nConcluido. CSVs de selecao e proveniencia em {SPLITS_DIR}/")


if __name__ == "__main__":
    main()
