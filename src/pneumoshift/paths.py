"""Resolucao central de caminhos do projeto.

A raiz do projeto e localizada subindo a arvore de diretorios ate encontrar um
marcador (a pasta 'modelo' e o 'requirements.txt'), em vez de assumir uma
profundidade fixa. Assim os scripts funcionam a partir de qualquer subpasta
(scripts/, tests/) sem depender de '.parent.parent'.
"""
from pathlib import Path


def raiz_projeto(inicio: Path | None = None) -> Path:
    """Sobe a partir de `inicio` (ou deste arquivo) ate achar a raiz do projeto."""
    atual = (inicio or Path(__file__)).resolve()
    for pasta in [atual, *atual.parents]:
        if (pasta / "modelo").is_dir() and (pasta / "requirements.txt").is_file():
            return pasta
    # fallback: dois niveis acima de src/pneumoshift/
    return Path(__file__).resolve().parents[2]


RAIZ = raiz_projeto()
MODELO = RAIZ / "modelo" / "keras_model.h5"
DADOS = RAIZ / "dados"
DADOS_TESTE = DADOS / "test"
RESULTADOS = RAIZ / "resultados"
