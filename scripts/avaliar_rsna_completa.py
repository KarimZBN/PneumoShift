"""
Avalia a base RSNA inteira (todos os pacientes elegiveis, distribuicao real), limiar 0,5.

Le dados/processed/rsna/{NORMAL,PNEUMONIA}/ (1 paciente = 1 imagem). Metricas:
sensibilidade, especificidade, AUC, Brier, ECE e curva de calibracao; IC 95% por bootstrap
(unidade = paciente = imagem). As metricas da matriz de confusao sao recalculadas a mao
para conferir o scikit-learn.

Saida em resultados/PRINCIPAL_rsna-base-inteira/: predicoes.csv (score cru por paciente),
metricas.json (metricas + IC + conferencia), imagens/{roc,calibracao}.png.

Uso:
    python scripts/avaliar_rsna_completa.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv
import json
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve
from keras.models import load_model

from pneumoshift import paths
from pneumoshift import metrics as M
from pneumoshift.inferencia import inferir_pasta
from pneumoshift import scores as SC

# --- Configuracao ---
GEOMETRIA = "padding"
LIMIAR = 0.5                    # limiar oficial (padrao do modelo)
N_BOOT = 2000                   # reamostragens do bootstrap (unidade = paciente)
SEED_BOOT = 42
N_BINS = 10

FULL_DIR = paths.RSNA_FULL


def conferencia_manual(tp, tn, fp, fn):
    """Recalcula sens/espec/precisao direto da matriz, para conferir o scikit-learn."""
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    espec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    return {"sensibilidade_manual": sens, "especificidade_manual": espec,
            "precisao_manual": prec}


def bootstrap_ic(y_true, y_score, limiar, n=N_BOOT, seed=SEED_BOOT):
    """IC 95% por reamostragem com reposicao (unidade = paciente = imagem na RSNA)."""
    rng = np.random.default_rng(seed)
    N = len(y_true)
    col = {"sensibilidade": [], "especificidade": [], "acuracia": [], "auc": [], "brier": []}
    for _ in range(n):
        idx = rng.integers(0, N, N)
        yt, ys = y_true[idx], y_score[idx]
        if yt.min() == yt.max():
            continue
        m = M.metricas_por_limiar(yt, ys, limiar)
        col["sensibilidade"].append(m["sensibilidade"])
        col["especificidade"].append(m["especificidade"])
        col["acuracia"].append(m["acuracia"])
        col["auc"].append(M.auc_sklearn(yt, ys))
        col["brier"].append(M.brier(yt, ys))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in col.items()}


def fig_roc(y_true, y_score, auc, destino):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"RSNA base inteira (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("1 - Especificidade (FPR)"); plt.ylabel("Sensibilidade (TPR)")
    plt.title("ROC — RSNA base inteira (distribuicao real)")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(destino, dpi=150); plt.close()


def fig_calibracao(y_true, y_score, brier, ece, destino):
    frac, media = calibration_curve(y_true, y_score, n_bins=N_BINS, strategy="uniform")
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfeita")
    plt.plot(media, frac, "o-", label=f"RSNA (Brier={brier:.3f}, ECE={ece:.3f})")
    plt.xlabel("Score medio previsto"); plt.ylabel("Fracao real de positivos")
    plt.title("Calibracao — RSNA base inteira")
    plt.legend(loc="upper left"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(destino, dpi=150); plt.close()


def main():
    if not FULL_DIR.is_dir():
        raise FileNotFoundError(
            f"Pasta ausente: {FULL_DIR}. Rode converter_rsna.py antes.")

    model = load_model(str(paths.MODELO), compile=False)
    print(f"Modelo carregado de: {paths.MODELO}")
    print(f"Avaliando a base RSNA INTEIRA em {FULL_DIR} ...")

    arquivos, y_true, y_score = inferir_pasta(model, FULL_DIR, GEOMETRIA)
    n = len(y_true)
    n_pos = int((y_true == 1).sum()); n_neg = int((y_true == 0).sum())
    print(f"  Pacientes avaliados: {n} ({n_pos} pneumonia + {n_neg} normal; "
          f"prevalencia real {n_neg/max(n_pos,1):.2f}:1)")

    pto = M.metricas_por_limiar(y_true, y_score, LIMIAR)
    auc = M.auc_sklearn(y_true, y_score)
    brier = M.brier(y_true, y_score)
    ece = M.ece(y_true, y_score, N_BINS)
    precisao = pto["tp"] / (pto["tp"] + pto["fp"]) if (pto["tp"] + pto["fp"]) else 0.0
    conf = conferencia_manual(pto["tp"], pto["tn"], pto["fp"], pto["fn"])
    ic = bootstrap_ic(y_true, y_score, LIMIAR)
    sat = int((y_score >= 1.0).sum())

    dados = {
        "base": "rsna", "conjunto": "completa", "geometria": GEOMETRIA,
        "limiar": LIMIAR, "n": n, "n_pneumonia": n_pos, "n_normal": n_neg,
        "prevalencia_neg_por_pos": round(n_neg / max(n_pos, 1), 4),
        **{k: pto[k] for k in ("tp", "tn", "fp", "fn")},
        "sensibilidade": pto["sensibilidade"], "especificidade": pto["especificidade"],
        "acuracia": pto["acuracia"], "f1": pto["f1"],
        "precisao": precisao,
        "precisao_obs": "influenciada pela prevalencia (base desbalanceada)",
        "auc": auc, "brier": brier, "ece": ece,
        "saturados": sat,
        "ic95_bootstrap": ic,
        "bootstrap_unidade": "paciente (=imagem na RSNA)",
        "conferencia_manual": conf,
    }

    out = paths.pasta_resultado("PRINCIPAL", "rsna-base-inteira")
    img = paths.imagens(out)

    # cache de scores por paciente (base inteira): alimenta as avaliacoes de subconjunto
    # (balanceado, pareado) sem reinferencia. Fica na pasta-fonte, ao lado dos PNGs.
    SC.salvar_cache(FULL_DIR / "scores_por_id.csv", arquivos, y_true, y_score)
    print(f"Cache de scores salvo em {FULL_DIR / 'scores_por_id.csv'}")

    # predicoes por paciente (score cru) + resumo no topo
    with open(out / "predicoes.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([f"Resumo - RSNA base inteira (limiar {LIMIAR})"])
        w.writerow(["Metrica", "Valor"])
        w.writerow(["N", n]); w.writerow(["Pneumonia / Normal", f"{n_pos} / {n_neg}"])
        w.writerow(["Sensibilidade (%)", f"{pto['sensibilidade']*100:.2f}"])
        w.writerow(["Especificidade (%)", f"{pto['especificidade']*100:.2f}"])
        w.writerow(["AUC", f"{auc:.4f}"]); w.writerow(["Brier", f"{brier:.4f}"])
        w.writerow(["Precisao (%) [prevalencia]", f"{precisao*100:.2f}"])
        w.writerow(["TP/TN/FP/FN", f"{pto['tp']}/{pto['tn']}/{pto['fp']}/{pto['fn']}"])
        w.writerow(["Scores saturados (==1,0)", f"{sat} de {n}"])
        w.writerow([])
        w.writerow(["Arquivo", "Classe Real", "Score Pneumonia", "Resultado"])
        for nome, yt, ys in zip(arquivos, y_true, y_score):
            classe = "PNEUMONIA" if yt == 1 else "NORMAL"
            pred = "PNEUMONIA" if ys >= LIMIAR else "NORMAL"
            res = "ACERTO" if (pred == classe) else "ERRO"
            w.writerow([nome, classe, repr(float(ys)), res])

    with open(out / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)

    fig_roc(y_true, y_score, auc, img / "roc.png")
    fig_calibracao(y_true, y_score, brier, ece, img / "calibracao.png")

    paths.escrever_leia(
        out, "RSNA — base inteira",
        "Avaliacao sobre todos os pacientes elegiveis, na distribuicao real das classes.",
        "Avaliacao do modelo sobre TODA a base RSNA elegivel (agregada por paciente), "
        "com o limiar padrao 0,5. Metricas com IC 95% por bootstrap (unidade = paciente).",
        {"metricas.json": "todas as metricas + IC bootstrap + conferencia manual",
         "predicoes.csv": "score cru por paciente (resumo no topo)",
         "imagens/roc.png": "curva ROC",
         "imagens/calibracao.png": "curva de calibracao"},
        {"N": f"{n} ({n_pos} pneumonia / {n_neg} normal)",
         "AUC": f"{auc:.4f}",
         "Sensibilidade": f"{pto['sensibilidade']*100:.2f}%",
         "Especificidade": f"{pto['especificidade']*100:.2f}%",
         "Brier": f"{brier:.4f}", "Limiar": str(LIMIAR)})

    print("\n=== RESULTADO (base inteira, limiar 0,5) ===")
    print(f"  Sensibilidade: {pto['sensibilidade']*100:.2f}%  IC95[{ic['sensibilidade'][0]*100:.1f}, {ic['sensibilidade'][1]*100:.1f}]")
    print(f"  Especificidade:{pto['especificidade']*100:.2f}%  IC95[{ic['especificidade'][0]*100:.1f}, {ic['especificidade'][1]*100:.1f}]")
    print(f"  AUC:           {auc:.4f}  IC95[{ic['auc'][0]:.4f}, {ic['auc'][1]:.4f}]")
    print(f"  Acuracia:      {pto['acuracia']*100:.2f}%  IC95[{ic['acuracia'][0]*100:.1f}, {ic['acuracia'][1]*100:.1f}]")
    print(f"  Brier:         {brier:.4f} | ECE: {ece:.4f}")
    print(f"  Precisao:      {precisao*100:.2f}% (influenciada pela prevalencia)")
    print(f"  Conferencia manual (sens/espec/prec): "
          f"{conf['sensibilidade_manual']*100:.2f}% / {conf['especificidade_manual']*100:.2f}% / {conf['precisao_manual']*100:.2f}%")
    print(f"\nSaida: {out}")


if __name__ == "__main__":
    main()
