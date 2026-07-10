"""Selecao reprodutivel de imagens das bases de teste."""
import random
from pathlib import Path

SEED = 42
N_NORMAL = 234
N_PNEUMONIA = 390
EXTS = (".jpeg", ".jpg", ".png")


def listar_imagens(folder):
    """Lista os arquivos de imagem de uma pasta (nomes)."""
    return [f.name for f in Path(folder).iterdir() if f.suffix.lower() in EXTS]


def selecionar(folder, n, seed=SEED):
    """Sorteia n arquivos de forma reprodutivel.

    A lista e ORDENADA (sorted) antes do embaralhamento: como iterdir() devolve os
    arquivos na ordem do sistema de arquivos (que varia entre maquinas/SO), sem a
    ordenacao previa o shuffle com semente fixa produziria selecoes diferentes em
    ambientes diferentes. Com sorted() a selecao passa a ser deterministica.
    """
    arquivos = sorted(listar_imagens(folder))
    random.seed(seed)
    random.shuffle(arquivos)
    if len(arquivos) < n:
        print(f"  ATENCAO: '{Path(folder).name}' tem {len(arquivos)} imagens "
              f"(pedido: {n}). Usando todas.")
        return arquivos
    return arquivos[:n]
