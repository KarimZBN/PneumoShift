"""
Cache de scores por paciente e avaliacao de subconjuntos sem reinferencia.

A inferencia do modelo sobre a base RSNA inteira e feita UMA vez e salva como um cache
(patientId -> classe, score cru). Qualquer subconjunto (balanceado, pareado a Chest
X-Ray) e avaliado consultando esse cache pelos IDs de uma selecao, sem rodar o modelo de
novo e sem copiar imagens.

Formatos:
    cache  (scores_por_id.csv): colunas patientId, classe, score
    selecao (rsna_<conj>_seed<N>.csv): colunas patientId, classe
"""
import csv

import numpy as np

from . import metrics as M


def salvar_cache(caminho, arquivos, y_true, y_score):
    """Grava o cache patientId,classe,score a partir da inferencia da base inteira."""
    with open(caminho, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["patientId", "classe", "score"])
        for nome, yt, ys in zip(arquivos, y_true, y_score):
            pid = nome.rsplit(".", 1)[0]                  # stem sem extensao
            classe = "PNEUMONIA" if int(yt) == 1 else "NORMAL"
            w.writerow([pid, classe, repr(float(ys))])    # score cru


def ler_cache(caminho):
    """Le o cache e devolve dict patientId -> (y_true, score)."""
    d = {}
    with open(caminho, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            pid = (row.get("patientId") or "").strip()
            classe = (row.get("classe") or "").strip().upper()
            if not pid or classe not in ("PNEUMONIA", "NORMAL"):
                continue
            d[pid] = (1 if classe == "PNEUMONIA" else 0, float(row["score"]))
    return d


def ler_selecao(caminho):
    """Le uma selecao (patientId, classe) e devolve a lista de patientId."""
    ids = []
    with open(caminho, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            pid = (row.get("patientId") or "").strip()
            if pid:
                ids.append(pid)
    return ids


def metricas_subconjunto(cache, ids, limiar):
    """Consulta o cache pelos IDs e calcula as metricas do subconjunto no limiar dado."""
    yt = np.array([cache[i][0] for i in ids if i in cache], dtype=int)
    ys = np.array([cache[i][1] for i in ids if i in cache], dtype=float)
    pto = M.metricas_por_limiar(yt, ys, limiar)
    return {
        "n": len(yt),
        "auc": M.auc_sklearn(yt, ys),
        "brier": M.brier(yt, ys),
        "sensibilidade": pto["sensibilidade"],
        "especificidade": pto["especificidade"],
        "acuracia": pto["acuracia"],
        "f1": pto["f1"],
        "tp": pto["tp"], "tn": pto["tn"], "fp": pto["fp"], "fn": pto["fn"],
    }
