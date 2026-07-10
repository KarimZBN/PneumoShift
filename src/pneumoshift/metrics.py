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
