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


def pasta_dados(base, geometria=None):
    """Pasta de teste de uma base. A geometria NAO altera a pasta.

    Os PNGs sao gravados no TAMANHO ORIGINAL (converter_rsna.py) e os JPEGs da
    Chest X-Ray sao os originais; em ambos a geometria (padding/esticar) e aplicada
    no pre-processamento da inferencia, nao "assada" no disco. Por isso a mesma
    pasta serve as duas geometrias. O parametro `geometria` e aceito e ignorado
    (mantido por compatibilidade de chamada).

    RSNA: usa-se o POOL de teste versionado (rsna_pool) como conjunto de teste da
    base — a amostra pareada 234/390 e sorteada dele. A validacao (rsna_validacao)
    e usada so para a definicao do limiar.
    """
    if base == "rsna":
        return DADOS_TESTE / "rsna_pool"
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
