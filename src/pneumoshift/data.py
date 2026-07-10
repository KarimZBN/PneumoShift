"""Selecao reprodutivel de imagens das bases de teste."""
import csv
import random
from pathlib import Path

SEED = 42
N_NORMAL = 234
N_PNEUMONIA = 390
EXTS = (".jpeg", ".jpg", ".png")


def ler_rotulos_rsna(labels_csv):
    """Le o CSV de rotulos da RSNA e devolve (pneumonia_ids, normal_ids, ambiguos).

    Target=1 -> pneumonia; Target=0 -> normal (ausencia de anotacao de opacidade, nao
    necessariamente exame normal). IDs presentes nas duas classes sao descartados.
    """
    pneumonia_ids, normal_ids = set(), set()
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row["Target"].strip()
            if t == "1":
                pneumonia_ids.add(row["patientId"])
            elif t == "0":
                normal_ids.add(row["patientId"])
    ambiguos = pneumonia_ids & normal_ids
    return pneumonia_ids - ambiguos, normal_ids - ambiguos, ambiguos


def ler_split(caminho):
    """Le um arquivo de split (uma coluna patientId) e devolve a lista de IDs, em ordem."""
    ids = []
    with open(caminho, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            pid = (row.get("patientId") or "").strip()
            if pid:
                ids.append(pid)
    return ids


def selecionar(pasta, n, seed=SEED):
    """Seleciona reprodutivelmente n arquivos de imagem de `pasta`.

    Ordena os nomes (torna a selecao independente da ordem do sistema de arquivos),
    embaralha com semente fixa e devolve os primeiros n. Se houver menos que n,
    devolve todos os disponiveis.
    """
    pasta = Path(pasta)
    arquivos = sorted(p.name for p in pasta.iterdir()
                      if p.is_file() and p.suffix.lower() in EXTS)
    random.Random(seed).shuffle(arquivos)
    return arquivos[:n]
