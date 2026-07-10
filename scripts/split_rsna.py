"""
Separacao deterministica da RSNA em VALIDACAO e POOL DE TESTE (roda uma vez).

Feita UMA vez, com semente mestre fixa, e salva em disco. Nenhum outro script sorteia
os IDs "na hora": todos leem de dados/splits/*.csv. Isso prova ausencia de vazamento
entre o conjunto de validacao (usado para calibrar o limiar) e o de teste.

  - validacao:  390 pneumonia + 234 normal  -> define o limiar (Youden)
  - pool_teste: 1500 pneumonia + 1500 normal -> as rodadas sorteiam 390/234 daqui

Os dois conjuntos sao DISJUNTOS (asserção no fim; ver tambem tests/test_split.py).

Saida:
    dados/splits/validacao.csv    (coluna patientId, classe)
    dados/splits/pool_teste.csv   (coluna patientId, classe)
    dados/splits/README.md        (log de proveniencia)

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
from pneumoshift.data import ler_rotulos_rsna

# --- Configuracao ---
SEED_SPLIT = 20260710          # semente mestre da separacao (fixa, versionada)
N_VAL_PNE, N_VAL_NORM = 390, 234       # validacao: mesma proporcao do teste
N_POOL_PNE, N_POOL_NORM = 1500, 1500   # pool de teste (as rodadas sorteiam 390/234 daqui)

LABELS_CSV = paths.DADOS / "raw" / "rsna" / "stage_2_train_labels.csv"
SPLITS_DIR = paths.DADOS / "splits"


def fatiar(ids, n_val, n_pool, rng):
    """Ordena, embaralha e fatia: primeiros n_val -> validacao; proximos n_pool -> pool."""
    ordenados = sorted(ids)          # sorted() antes do shuffle: determinismo entre SO
    rng.shuffle(ordenados)
    val = ordenados[:n_val]
    pool = ordenados[n_val:n_val + n_pool]
    return val, pool


def salvar(caminho, registros):
    """Grava (patientId, classe) num CSV."""
    with open(caminho, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["patientId", "classe"])
        w.writerows(registros)


def main():
    if not LABELS_CSV.is_file():
        raise FileNotFoundError(f"CSV de rotulos nao encontrado: {LABELS_CSV}")
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    pneu, norm, ambiguos = ler_rotulos_rsna(LABELS_CSV)
    print(f"Elegiveis -> pneumonia: {len(pneu)} | normal: {len(norm)} | ambiguos removidos: {len(ambiguos)}")

    for nome, need in (("pneumonia", N_VAL_PNE + N_POOL_PNE), ("normal", N_VAL_NORM + N_POOL_NORM)):
        disp = len(pneu) if nome == "pneumonia" else len(norm)
        if disp < need:
            raise ValueError(f"Classe {nome}: {disp} disponiveis < {need} necessarios.")

    rng = random.Random(SEED_SPLIT)          # gerador dedicado, isolado
    val_pne, pool_pne = fatiar(pneu, N_VAL_PNE, N_POOL_PNE, rng)
    val_norm, pool_norm = fatiar(norm, N_VAL_NORM, N_POOL_NORM, rng)

    validacao = [(i, "PNEUMONIA") for i in val_pne] + [(i, "NORMAL") for i in val_norm]
    pool_teste = [(i, "PNEUMONIA") for i in pool_pne] + [(i, "NORMAL") for i in pool_norm]

    # --- ASSERCAO DE ZERO VAZAMENTO ---
    ids_val = {i for i, _ in validacao}
    ids_pool = {i for i, _ in pool_teste}
    intersec = ids_val & ids_pool
    assert not intersec, f"VAZAMENTO: {len(intersec)} IDs em validacao E pool_teste!"

    salvar(SPLITS_DIR / "validacao.csv", validacao)
    salvar(SPLITS_DIR / "pool_teste.csv", pool_teste)

    with open(SPLITS_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(
            f"# Splits da RSNA (proveniencia)\n\n"
            f"Gerado por `scripts/split_rsna.py` em {datetime.now():%Y-%m-%d %H:%M} "
            f"com `SEED_SPLIT = {SEED_SPLIT}`.\n\n"
            f"- validacao.csv:  {N_VAL_PNE} pneumonia + {N_VAL_NORM} normal "
            f"(define o limiar)\n"
            f"- pool_teste.csv: {N_POOL_PNE} pneumonia + {N_POOL_NORM} normal "
            f"(as rodadas sorteiam 390/234 deste conjunto)\n"
            f"- Conjuntos DISJUNTOS (0 IDs em comum) — verificado por asserção e "
            f"tests/test_split.py.\n\n"
            f"Reproduzivel: rodar este script com a mesma seed gera exatamente os "
            f"mesmos arquivos.\n"
        )

    print(f"\nValidacao: {len(validacao)} | Pool teste: {len(pool_teste)} | intersecao: {len(intersec)}")
    print(f"Salvo em: {SPLITS_DIR}/ (validacao.csv, pool_teste.csv, README.md)")


if __name__ == "__main__":
    main()
