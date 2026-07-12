"""
Nucleo estatistico da avaliacao da Chest X-Ray: AUC (sklearn), calibracao, intervalos de
confianca (bootstrap) e scores saturados, a partir do predicoes.csv de cada execucao.

Por execucao (funde na propria pasta, sem arquivo separado):
    - metricas.json          (acrescenta AUC, Brier, ECE, IC95 e saturados as metricas base)
    - imagens/roc.png        (curva ROC)
    - imagens/calibracao.png (curva de calibracao)

Comparacao de geometria em resultados/DESCARTAVEL_comparacao-geometria/:
    - roc_padding-vs-esticar_cxray.png

A comparacao de dominio (Chest X-Ray x RSNA) e feita a parte, pois a RSNA tem avaliacoes
proprias (avaliar_rsna_completa / _balanceado / _pareado).

Uso:
    python scripts/analise_estatistica.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve

from pneumoshift import paths
from pneumoshift import metrics as M

# --- Configuracao ---
SEED = 42
N_BOOT = 2000
LIMIAR = 0.5
N_BINS_ECE = 10
SATURADO = 1.0
GEOMETRIAS = ("padding", "esticar")


def ler_csv(caminho):
    """Le y_true, y_score (cru) e o rotulo acerto/erro; robusto a resumo no topo/fim."""
    with open(caminho, newline="", encoding="utf-8-sig") as f:
        todas = list(csv.reader(f))
    idx_header = next((i for i, row in enumerate(todas)
                       if "Arquivo" in row and "Classe Real" in row), None)
    if idx_header is None:
        raise ValueError(f"Cabecalho da tabela nao encontrado em {caminho}")
    header = todas[idx_header]
    col = {n: header.index(n) for n in ("Classe Real", "Score Pneumonia", "Resultado")}
    y_true, y_score, resultado = [], [], []
    for row in todas[idx_header + 1:]:
        if len(row) <= max(col.values()):
            continue
        classe = (row[col["Classe Real"]] or "").strip().upper()
        if classe not in ("PNEUMONIA", "NORMAL"):
            continue
        y_true.append(1 if classe == "PNEUMONIA" else 0)
        y_score.append(float(row[col["Score Pneumonia"]]))
        resultado.append((row[col["Resultado"]] or "").strip().upper())
    return np.array(y_true), np.array(y_score), resultado


def bootstrap_ic(y_true, y_score, limiar=LIMIAR, n=N_BOOT, seed=SEED):
    """IC 95% por reamostragem com reposicao para sens/espec/acuracia/AUC."""
    rng = np.random.default_rng(seed)
    N = len(y_true)
    coletas = {"sensibilidade": [], "especificidade": [], "acuracia": [], "auc": []}
    for _ in range(n):
        idx = rng.integers(0, N, N)
        yt, ys = y_true[idx], y_score[idx]
        if yt.min() == yt.max():
            continue
        m = M.metricas_por_limiar(yt, ys, limiar)
        coletas["sensibilidade"].append(m["sensibilidade"])
        coletas["especificidade"].append(m["especificidade"])
        coletas["acuracia"].append(m["acuracia"])
        coletas["auc"].append(M.auc_sklearn(yt, ys))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in coletas.items()}


def pasta_da_execucao(base, geometria):
    """Resolve a pasta de resultado de (base, geometria) no esquema por prefixo (cxray)."""
    if base != "cxray":
        return None
    p = (paths.RESULTADOS / "PRINCIPAL_cxray" if geometria == "padding"
         else paths.RESULTADOS / "DESCARTAVEL_cxray-esticar")
    return p if (p / "predicoes.csv").is_file() else None


def analisar_execucao(base, geometria):
    pasta = pasta_da_execucao(base, geometria)
    return analisar_pasta(base, geometria, pasta) if pasta else None


def analisar_pasta(base, geometria, pasta):
    """Analisa a execucao em `pasta`; funde a estatistica no metricas.json e gera figuras."""
    y_true, y_score, resultado = ler_csv(pasta / "predicoes.csv")
    n = len(y_true)
    auc_sk = M.auc_sklearn(y_true, y_score)
    auc_mw = M.auc_mann_whitney(list(y_true), list(y_score))
    brier = M.brier(y_true, y_score)
    ece = M.ece(y_true, y_score, N_BINS_ECE)
    pto = M.metricas_por_limiar(y_true, y_score, LIMIAR)
    ic = bootstrap_ic(y_true, y_score)

    sat = y_score >= SATURADO
    erros = np.array([r == "ERRO" for r in resultado]) if resultado else np.zeros(n, bool)
    d = {
        "base": base, "geometria": geometria, "n": n,
        "auc_sklearn": auc_sk, "auc_mann_whitney": auc_mw, "brier": brier, "ece": ece,
        "acuracia": pto["acuracia"], "sensibilidade": pto["sensibilidade"],
        "especificidade": pto["especificidade"], "f1": pto["f1"],
        "matriz": {k: pto[k] for k in ("tp", "tn", "fp", "fn")},
        "ic95": ic, "saturados_total": int(sat.sum()),
        "saturados_entre_erros": int((sat & erros).sum()), "n_erros": int(erros.sum()),
        "_y_true": y_true, "_y_score": y_score, "_pasta": pasta,
    }

    print(f"\n== {base}_{geometria} (n={n}) ==")
    print(f"  AUC sklearn {auc_sk:.4f} | mann-whitney {auc_mw:.4f} (dif {abs(auc_sk-auc_mw):.6f})")
    print(f"  Brier {brier:.4f} | ECE {ece:.4f} | saturados {int(sat.sum())}/{n}")

    img = paths.imagens(pasta)
    _fig_roc({f"{base}_{geometria}": d}, img / "roc.png", f"ROC — {base} {geometria}")
    _fig_calibracao({f"{base}_{geometria}": d}, img / "calibracao.png",
                    f"Calibracao — {base} {geometria}")

    # funde a estatistica no metricas.json existente (um unico arquivo por pasta)
    estat = {k: v for k, v in d.items() if not k.startswith("_")}
    mj = pasta / "metricas.json"
    base_json = json.load(open(mj, encoding="utf-8")) if mj.is_file() else {}
    base_json.update({k: estat[k] for k in (
        "auc_sklearn", "auc_mann_whitney", "brier", "ece", "ic95",
        "saturados_total", "saturados_entre_erros", "n_erros")})
    with open(mj, "w", encoding="utf-8") as f:
        json.dump(base_json, f, indent=2, ensure_ascii=False)
    return d


def _fig_roc(execucoes, destino, titulo):
    plt.figure(figsize=(6, 6))
    for rotulo, d in execucoes.items():
        fpr, tpr, _ = roc_curve(d["_y_true"], d["_y_score"])
        plt.plot(fpr, tpr, label=f"{rotulo} (AUC={d['auc_sklearn']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("1 - Especificidade (FPR)"); plt.ylabel("Sensibilidade (TPR)")
    plt.title(titulo); plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    destino.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(destino, dpi=150); plt.close()


def _fig_calibracao(execucoes, destino, titulo):
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfeita")
    for rotulo, d in execucoes.items():
        frac_pos, media_pred = calibration_curve(
            d["_y_true"], d["_y_score"], n_bins=N_BINS_ECE, strategy="uniform")
        plt.plot(media_pred, frac_pos, "o-",
                 label=f"{rotulo} (Brier={d['brier']:.3f}, ECE={d['ece']:.3f})")
    plt.xlabel("Score medio previsto"); plt.ylabel("Fracao real de positivos")
    plt.title(titulo); plt.legend(loc="upper left"); plt.grid(alpha=0.3)
    destino.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(destino, dpi=150); plt.close()


def main():
    dados = {}
    for geo in GEOMETRIAS:
        d = analisar_execucao("cxray", geo)
        if d is not None:
            dados[("cxray", geo)] = d

    if not dados:
        print("Nenhum predicoes.csv encontrado (PRINCIPAL_cxray / DESCARTAVEL_cxray-esticar).")
        return

    # comparacao de geometria: padding x esticar da Chest X-Ray
    geo = {g: dados.get(("cxray", g)) for g in GEOMETRIAS}
    geo = {g: d for g, d in geo.items() if d is not None}
    if len(geo) == 2:
        comp = paths.pasta_resultado("DESCARTAVEL", "comparacao-geometria")
        _fig_roc({g: d for g, d in geo.items()},
                 comp / "imagens" / "roc_padding-vs-esticar_cxray.png",
                 "ROC — padding vs esticar (Chest X-Ray)")
        paths.escrever_leia(
            comp, "Comparacao de geometria — Chest X-Ray",
            "Sensibilidade da metrica a geometria (padding x esticar) na Chest X-Ray.",
            "ROC das duas geometrias sobre a Chest X-Ray, para medir o efeito do "
            "redimensionamento.",
            {"imagens/roc_padding-vs-esticar_cxray.png": "ROC comparando as duas geometrias"})
        print("\n[comparacao geometria] gerada em", comp)

    print(f"\nExecucoes analisadas: {len(dados)}")


if __name__ == "__main__":
    main()
