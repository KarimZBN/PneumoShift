"""
Nucleo estatistico: AUC (sklearn), calibracao, intervalos de confianca (bootstrap) e
scores saturados, sobre os CSVs de predicao das execucoes.

Le resultados/<base>_<geometria>/predicoes.csv de cada execucao existente e produz:

  Por execucao (na propria pasta):
    - metricas_estat.json    (AUC, Brier, ECE, IC95, saturados)
    - roc.png                (ROC individual da execucao)
    - calibracao.png         (curva de calibracao individual)

  Comparativas (em resultados/_comparacoes/):
    - dominio/    cxray vs rsna na geometria canonica (padding) -> domain shift
    - geometria/  padding vs esticar por base -> sensibilidade da metrica a geometria

  Consolidado (em resultados/_comparacoes/):
    - resumo_geral.csv   (uma linha por execucao)

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
GEOMETRIA_CANONICA = "padding"          # base da comparacao de dominio

BASES = ("cxray", "rsna")
GEOMETRIAS = ("padding", "esticar")
COMP_DIR = paths.RESULTADOS / "_comparacoes"


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


def analisar_execucao(base, geometria):
    """Localiza a execucao mais recente de (base, geometria) e a analisa. Dict ou None."""
    pasta = paths.execucao_mais_recente(base, geometria)
    if pasta is None:
        return None
    return analisar_pasta(base, geometria, pasta)


def analisar_pasta(base, geometria, pasta):
    """Analisa a execucao em `pasta` (deve conter predicoes.csv). Devolve o dict de metricas.

    Gera, na propria pasta: metricas_estat.json, roc.png, calibracao.png. Reutilizavel
    tanto pelo main() (varre tudo) quanto pelo avaliar_lote (chama logo apos gravar o CSV).
    """
    csv_path = pasta / "predicoes.csv"

    y_true, y_score, resultado = ler_csv(csv_path)
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
        "ic95": ic,
        "saturados_total": int(sat.sum()),
        "saturados_entre_erros": int((sat & erros).sum()),
        "n_erros": int(erros.sum()),
        "_y_true": y_true, "_y_score": y_score, "_pasta": pasta,
    }

    print(f"\n== {base}_{geometria} (n={n}) ==")
    print(f"  AUC sklearn {auc_sk:.4f} | mann-whitney {auc_mw:.4f} (dif {abs(auc_sk-auc_mw):.6f})")
    print(f"  Brier {brier:.4f} | ECE {ece:.4f}")
    print(f"  Acc {pto['acuracia']*100:.2f}% IC[{ic['acuracia'][0]*100:.1f}-{ic['acuracia'][1]*100:.1f}]"
          f" | Espec {pto['especificidade']*100:.2f}% IC[{ic['especificidade'][0]*100:.1f}-{ic['especificidade'][1]*100:.1f}]")
    print(f"  AUC IC[{ic['auc'][0]:.4f}-{ic['auc'][1]:.4f}] | saturados {int(sat.sum())}/{n}"
          f" (entre erros {int((sat & erros).sum())}/{int(erros.sum())})")

    # figuras individuais na propria pasta
    _fig_roc({f"{base}_{geometria}": d}, pasta / "roc.png",
             f"ROC — {base} {geometria}")
    _fig_calibracao({f"{base}_{geometria}": d}, pasta / "calibracao.png",
                    f"Calibracao — {base} {geometria}")
    # json individual (sem arrays)
    with open(pasta / "metricas_estat.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in d.items() if not k.startswith("_")},
                  f, indent=2, ensure_ascii=False)
    return d


def _fig_roc(execucoes, destino, titulo):
    plt.figure(figsize=(6, 6))
    for rotulo, d in execucoes.items():
        fpr, tpr, _ = roc_curve(d["_y_true"], d["_y_score"])
        plt.plot(fpr, tpr, label=f"{rotulo} (AUC={d['auc_sklearn']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("1 - Especificidade (FPR)")
    plt.ylabel("Sensibilidade (TPR)")
    plt.title(titulo)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
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
    plt.xlabel("Score medio previsto")
    plt.ylabel("Fracao real de positivos")
    plt.title(titulo)
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    destino.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(destino, dpi=150); plt.close()


def _ref_execucoes(dados):
    """Linha por execucao (base, geometria -> pasta) para registrar o que entrou no run."""
    return {f"{b}_{g}": d["_pasta"].name for (b, g), d in sorted(dados.items())}


def comparativas(dados, run_dir):
    """Gera as figuras comparativas dentro de run_dir/{dominio,geometria}/.

    Nomes simples e fixos; o que distingue um run do outro e a PASTA (run_<timestamp>),
    nao o nome do arquivo. Assim cada execucao do analise_estatistica fica autocontida."""
    # --- DOMINIO: cxray vs rsna na geometria canonica ---
    dom = {b: dados.get((b, GEOMETRIA_CANONICA)) for b in BASES}
    dom = {b: d for b, d in dom.items() if d is not None}
    if len(dom) == 2:
        _fig_roc({b.upper(): d for b, d in dom.items()},
                 run_dir / "dominio" / f"roc_cxray-vs-rsna_{GEOMETRIA_CANONICA}.png",
                 f"ROC — Kaggle vs RSNA ({GEOMETRIA_CANONICA})")
        _fig_calibracao({b.upper(): d for b, d in dom.items()},
                        run_dir / "dominio" / f"calibracao_cxray-vs-rsna_{GEOMETRIA_CANONICA}.png",
                        f"Calibracao — Kaggle vs RSNA ({GEOMETRIA_CANONICA})")
        print(f"\n[comparacao dominio] gerada ({GEOMETRIA_CANONICA})")

    # --- GEOMETRIA: padding vs esticar por base ---
    for base in BASES:
        geo = {g: dados.get((base, g)) for g in GEOMETRIAS}
        geo = {g: d for g, d in geo.items() if d is not None}
        if len(geo) == 2:
            _fig_roc({g: d for g, d in geo.items()},
                     run_dir / "geometria" / f"roc_padding-vs-esticar_{base}.png",
                     f"ROC — padding vs esticar ({base})")
            print(f"[comparacao geometria] gerada ({base})")


def salvar_resumo(dados, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    # registro do que entrou neste run (rastreabilidade: qual execucao de cada base/geo)
    with open(run_dir / "execucoes.json", "w", encoding="utf-8") as f:
        json.dump(_ref_execucoes(dados), f, indent=2, ensure_ascii=False)
    with open(run_dir / "resumo_geral.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["base", "geometria", "n", "auc_sklearn", "auc_mann_whitney",
                    "brier", "ece", "acuracia", "acuracia_ic95", "sensibilidade",
                    "sensibilidade_ic95", "especificidade", "especificidade_ic95",
                    "auc_ic95", "saturados_total", "saturados_entre_erros", "n_erros"])
        for (base, geo), d in sorted(dados.items()):
            ic = d["ic95"]
            w.writerow([
                base, geo, d["n"], f"{d['auc_sklearn']:.4f}", f"{d['auc_mann_whitney']:.4f}",
                f"{d['brier']:.4f}", f"{d['ece']:.4f}",
                f"{d['acuracia']:.4f}", f"{ic['acuracia'][0]:.4f}-{ic['acuracia'][1]:.4f}",
                f"{d['sensibilidade']:.4f}", f"{ic['sensibilidade'][0]:.4f}-{ic['sensibilidade'][1]:.4f}",
                f"{d['especificidade']:.4f}", f"{ic['especificidade'][0]:.4f}-{ic['especificidade'][1]:.4f}",
                f"{ic['auc'][0]:.4f}-{ic['auc'][1]:.4f}",
                d["saturados_total"], d["saturados_entre_erros"], d["n_erros"],
            ])


def main():
    dados = {}
    for base in BASES:
        for geo in GEOMETRIAS:
            d = analisar_execucao(base, geo)
            if d is not None:
                dados[(base, geo)] = d

    if not dados:
        print("Nenhum predicoes.csv encontrado em resultados/<base>_<geometria>/.")
        return

    from datetime import datetime
    run_dir = COMP_DIR / f"run_{datetime.now():%Y%m%d-%H%M%S}"
    comparativas(dados, run_dir)
    salvar_resumo(dados, run_dir)
    print(f"\nExecucoes analisadas: {len(dados)}")
    print(f"Comparativas + resumo em: {run_dir}")


if __name__ == "__main__":
    main()
