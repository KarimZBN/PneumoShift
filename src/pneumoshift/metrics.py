"""Metricas derivadas da matriz de confusao e rotulagem de categoria."""

CLASS_NAMES = ["pneumonia", "normal"]   # ordem da saida do modelo (indice 0 = pneumonia)
LIMIAR = 0.5                            # limiar de decisao (score da classe pneumonia)


def metricas_confusao(tp, tn, fp, fn):
    """Acuracia, precisao, sensibilidade, especificidade e F1 a partir da matriz."""
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0            # sensibilidade
    specificity = tn / (tn + fp) if (tn + fp) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return acc, precision, recall, specificity, f1


def auc_mann_whitney(y_true, y_score):
    """AUC pela estatistica de Mann-Whitney sobre os scores (area sob a ROC)."""
    pos = [s for t, s in zip(y_true, y_score) if t == 1]
    neg = [s for t, s in zip(y_true, y_score) if t == 0]
    if not pos or not neg:
        return 0.0
    wins = sum(1.0 if ps > ns else 0.5 if ps == ns else 0.0
               for ps in pos for ns in neg)
    return wins / (len(pos) * len(neg))


def categoria(real_pneumonia, pred_pneumonia):
    """Rotula o caso como VP/VN/FP/FN, tomando pneumonia como classe positiva."""
    if real_pneumonia and pred_pneumonia:
        return "VP"
    if real_pneumonia and not pred_pneumonia:
        return "FN"
    if not real_pneumonia and pred_pneumonia:
        return "FP"
    return "VN"


# --- Metricas via scikit-learn ---

def auc_sklearn(y_true, y_score):
    """AUC-ROC pela referencia auditavel do scikit-learn (roc_auc_score)."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


def brier(y_true, y_score):
    """Brier score (erro quadratico medio entre score e resultado; menor = melhor)."""
    from sklearn.metrics import brier_score_loss
    return float(brier_score_loss(y_true, y_score))


def ece(y_true, y_score, n_bins=10):
    """Expected Calibration Error por binning uniforme em [0, 1].

    Em cada bin: |media dos scores - fracao real de positivos|, ponderado pelo
    numero de amostras. Retorna o desvio medio ponderado (0 = perfeitamente calibrado).
    """
    import numpy as np
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    limites = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_score)
    erro = 0.0
    for i in range(n_bins):
        lo, hi = limites[i], limites[i + 1]
        # ultimo bin inclui o 1.0
        sel = (y_score >= lo) & (y_score < hi) if i < n_bins - 1 else (y_score >= lo) & (y_score <= hi)
        n = int(sel.sum())
        if n == 0:
            continue
        conf = y_score[sel].mean()          # confianca media declarada
        acc = y_true[sel].mean()            # fracao real de positivos
        erro += (n / total) * abs(conf - acc)
    return float(erro)


def metricas_por_limiar(y_true, y_score, limiar=LIMIAR):
    """Acuracia, sensibilidade, especificidade (e matriz) aplicando um limiar aos scores."""
    import numpy as np
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_score, dtype=float) >= limiar).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc, precision, recall, specificity, f1 = metricas_confusao(tp, tn, fp, fn)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "acuracia": acc, "sensibilidade": recall,
            "especificidade": specificity, "f1": f1}

