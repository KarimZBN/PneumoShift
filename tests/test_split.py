"""
Verifica que os splits da RSNA (validacao x pool_teste) sao DISJUNTOS — prova de que
nao ha vazamento entre o conjunto que calibra o limiar e o que avalia o modelo.

Uso:
    python tests/test_split.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneumoshift import paths
from pneumoshift.data import ler_split

SPLITS_DIR = paths.DADOS / "splits"


def main():
    val_path = SPLITS_DIR / "validacao.csv"
    pool_path = SPLITS_DIR / "pool_teste.csv"
    if not (val_path.is_file() and pool_path.is_file()):
        print("Splits nao encontrados. Rode antes: python scripts/split_rsna.py")
        sys.exit(1)

    val = ler_split(val_path)
    pool = ler_split(pool_path)
    sv, sp = set(val), set(pool)

    # unicidade dentro de cada split
    assert len(val) == len(sv), "IDs duplicados em validacao.csv"
    assert len(pool) == len(sp), "IDs duplicados em pool_teste.csv"
    # disjuncao entre os splits
    intersec = sv & sp
    assert not intersec, f"VAZAMENTO: {len(intersec)} IDs em validacao E pool_teste"

    print(f"validacao: {len(val)} | pool_teste: {len(pool)} | intersecao: {len(intersec)}")
    print(">> OK: splits disjuntos, sem vazamento.")


if __name__ == "__main__":
    main()
