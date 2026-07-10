"""Resolucao central de caminhos do projeto.

A raiz do projeto e localizada subindo a arvore de diretorios ate encontrar um
marcador (a pasta 'modelo' e o 'requirements.txt'), em vez de assumir uma
profundidade fixa. Assim os scripts funcionam a partir de qualquer subpasta.
"""
from datetime import datetime
from pathlib import Path

from .data import SEED


def raiz_projeto(inicio: Path | None = None) -> Path:
    """Sobe a partir de `inicio` (ou deste arquivo) ate achar a raiz do projeto."""
    atual = (inicio or Path(__file__)).resolve()
    for pasta in [atual, *atual.parents]:
        if (pasta / "modelo").is_dir() and (pasta / "requirements.txt").is_file():
            return pasta
    return Path(__file__).resolve().parents[2]


RAIZ = raiz_projeto()
MODELO = RAIZ / "modelo" / "keras_model.h5"
DADOS = RAIZ / "dados"
DADOS_TESTE = DADOS / "test"
RESULTADOS = RAIZ / "resultados"


def pasta_dados(base, geometria):
    """Pasta de teste de uma (base, geometria).

    Na RSNA a geometria ja esta gravada no PNG por dois conversores distintos:
      rsna_padding = converter_rsna_padding.py (padding letterbox)
      rsna         = converter_rsna.py         (resize direto / esticado)
    Na Chest X-Ray (cxray) sao os JPEGs originais (geometria aplicada no
    pre-processamento), entao a pasta e a mesma para as duas geometrias.
    """
    if base == "rsna":
        return DADOS_TESTE / ("rsna_padding" if geometria == "padding" else "rsna")
    if base == "cxray":
        return DADOS_TESTE / "cxray"
    raise ValueError(f"base invalida: {base!r} (use 'cxray' ou 'rsna').")


def nova_execucao(base, geometria, seed=SEED):
    """Cria e devolve a pasta UNICA de uma execucao (nunca sobrescreve).

    Layout: resultados/<base>/<geometria>_<AAAAMMDD-HHMMSS>_seed<N>/
    O timestamp garante unicidade absoluta e preserva o historico de execucoes.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    pasta = RESULTADOS / base / f"{geometria}_{ts}_seed{seed}"
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def execucao_mais_recente(base, geometria):
    """Devolve a pasta da execucao mais recente de (base, geometria), ou None.

    Procura resultados/<base>/<geometria>_*/, ordena por nome (o timestamp torna a
    ordem alfabetica = ordem cronologica) e devolve a ultima que tenha predicoes.csv.
    """
    raiz = RESULTADOS / base
    if not raiz.is_dir():
        return None
    candidatas = sorted(p for p in raiz.iterdir()
                        if p.is_dir() and p.name.startswith(f"{geometria}_"))
    for p in reversed(candidatas):
        if (p / "predicoes.csv").is_file():
            return p
    return None
