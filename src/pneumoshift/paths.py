"""Resolucao central de caminhos do projeto e organizacao das saidas.

A raiz do projeto e localizada subindo a arvore de diretorios ate encontrar um
marcador (a pasta 'modelo' e o 'requirements.txt'), em vez de assumir uma
profundidade fixa. Assim os scripts funcionam a partir de qualquer subpasta.

Resultados sao organizados por PAPEL, via prefixo no nome da pasta:
    PRINCIPAL_<nome>    resultado principal
    APOIO_<nome>        analise complementar (robustez, comparacao, verificacoes)
    DESCARTAVEL_<nome>  execucao auxiliar (ex: comparacao de geometria)
Cada avaliacao usa uma pasta unica e SOBRESCREVE a execucao anterior (so a ultima
vale), evitando acumulo de timestamps.
"""
import shutil
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
DADOS_TESTE = DADOS / "test"                # conjuntos de avaliacao (ex: cxray, JPEGs originais)
DADOS_PROCESSED = DADOS / "processed"       # dados derivados do raw (ex: RSNA convertida de DICOM)
RSNA_FULL = DADOS_PROCESSED / "rsna"        # base RSNA inteira convertida (fonte unica)
RESULTADOS = RAIZ / "resultados"

PREFIXOS = ("PRINCIPAL", "APOIO", "DESCARTAVEL")


def pasta_dados(base, geometria=None):
    """Pasta de teste de uma base. A geometria NAO altera a pasta.

    Os PNGs sao gravados no TAMANHO ORIGINAL (converter_rsna.py) e os JPEGs da
    Chest X-Ray sao os originais; em ambos a geometria (padding/esticar) e aplicada
    no pre-processamento da inferencia. A RSNA usa processed/rsna (base inteira).
    """
    if base == "rsna":
        return RSNA_FULL
    if base == "cxray":
        return DADOS_TESTE / "cxray"
    raise ValueError(f"base invalida: {base!r} (use 'cxray' ou 'rsna').")


def pasta_resultado(prefixo, nome, limpar=True):
    """Devolve (e cria) resultados/<PREFIXO>_<nome>/. Se limpar, remove o conteudo antigo.

    prefixo: "PRINCIPAL" | "APOIO" | "DESCARTAVEL".
    Sobrescreve a execucao anterior (limpar=True) para nao acumular saidas antigas.
    """
    if prefixo not in PREFIXOS:
        raise ValueError(f"prefixo invalido: {prefixo!r} (use {PREFIXOS}).")
    base = RESULTADOS / f"{prefixo}_{nome}"

    if base.exists():
        acao = _perguntar_sobrescrever(base) if limpar else "novo"
        if acao == "sobrescrever":
            shutil.rmtree(base)
            pasta = base
        else:  # cria variante com sufixo (1), (2), ... sem tocar na existente
            pasta = _proximo_sufixo(base)
    else:
        pasta = base

    (pasta / "imagens").mkdir(parents=True, exist_ok=True)
    return pasta


def _proximo_sufixo(base):
    """Devolve base_(1), base_(2)... — o primeiro nome livre a partir de `base`."""
    i = 1
    while True:
        cand = base.parent / f"{base.name}_({i})"
        if not cand.exists():
            return cand
        i += 1


def _perguntar_sobrescrever(base):
    """Pergunta no terminal: sobrescrever a pasta existente ou criar uma nova com sufixo.

    Sem terminal interativo (ex: execucao agendada), opta por 'novo' (nao apaga nada).
    """
    import sys
    if not sys.stdin or not sys.stdin.isatty():
        return "novo"
    print(f"\n[!] Ja existe: {base.name}")
    resp = input("    [S] sobrescrever  |  [N] criar nova com sufixo (1),(2)  -> ").strip().lower()
    return "sobrescrever" if resp in ("s", "sim", "y") else "novo"


def imagens(pasta):
    """Subpasta 'imagens/' de uma pasta de resultado (figuras separadas dos dados)."""
    d = pasta / "imagens"
    d.mkdir(parents=True, exist_ok=True)
    return d


def escrever_leia(pasta, titulo, papel, descricao, arquivos, numeros=None):
    """Gera o LEIA.md da pasta: o que e, para que serve, numeros-chave e o que e cada arquivo.

    titulo: nome legivel da avaliacao.
    papel: uma frase sobre o proposito (por que esta pasta existe).
    descricao: texto curto de contexto.
    arquivos: dict {nome_arquivo: descricao}.
    numeros: dict opcional {metrica: valor_formatado} com os resultados-chave.
    """
    linhas = [f"# {titulo}", "", f"**Papel:** {papel}", "",
              f"_Gerado em {datetime.now():%Y-%m-%d %H:%M}._", "", descricao, ""]
    if numeros:
        linhas += ["## Numeros-chave", ""]
        linhas += [f"- **{k}:** {v}" for k, v in numeros.items()]
        linhas += [""]
    linhas += ["## Arquivos", ""]
    linhas += [f"- `{nome}` — {desc}" for nome, desc in arquivos.items()]
    linhas += [""]
    (pasta / "LEIA.md").write_text("\n".join(linhas), encoding="utf-8")
