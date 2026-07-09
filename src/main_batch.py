"""
Avaliacao em lote do classificador de pneumonia (modelo Keras .h5).

Roda uma vez por base sobre o conjunto de teste completo e pareado:
    Kaggle (Chest X-Ray, pediatrico) -> 234 normais + 390 pneumonia
    RSNA   (adulto, convertido)      -> 234 normais + 390 pneumonia

O pareamento (mesma quantidade e proporcao nas duas bases) isola a variavel
"origem dos dados" e mantem a comparacao com Shao (2021), que avaliou o mesmo
conjunto de teste do Kaggle (234/390).

Para cada imagem grava-se o score da classe pneumonia, usado no calculo do AUC.
As metricas (acuracia, precisao, sensibilidade, especificidade, F1 e AUC) sao
derivadas da matriz de confusao.
"""
from pathlib import Path
import csv
import random

import cv2
import numpy as np
from keras.models import load_model

# --- Configuracao ---
FONTE = "rsna"                 # "kaggle" ou "rsna"
IMG_SIZE = (224, 224)
BATCH_SIZE = 100
SEED = 42
CLASS_NAMES = ["pneumonia", "normal"]   # ordem da saida do modelo (indice 0 = pneumonia)

# Amostra pareada entre as bases, seguindo Shao (2021).
N_NORMAL = 234
N_PNEUMONIA = 390
LIMIAR_ALT = 0.95              # limiar alternativo, apenas ilustrativo

# --- Caminhos (raiz do projeto = pasta acima de src/) ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "modelo" / "keras_model.h5"
RESULTS_DIR = BASE_DIR / "resultados" / "csv"
TEST_DIR = BASE_DIR / "dados" / "test" / FONTE
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modelo nao encontrado: {MODEL_PATH}")


def proximo_arquivo_saida():
    """Retorna um nome de CSV ainda inexistente (resultados_<fonte>_<n>.csv)."""
    n = 1
    while (RESULTS_DIR / f"resultados_{FONTE}_{n}.csv").exists():
        n += 1
    return RESULTS_DIR / f"resultados_{FONTE}_{n}.csv"


def listar_imagens(folder):
    """Lista os arquivos de imagem (jpeg/jpg/png) de uma pasta."""
    return [f.name for f in Path(folder).iterdir()
            if f.suffix.lower() in (".jpeg", ".jpg", ".png")]


def selecionar(folder, n):
    """Sorteia n arquivos de forma reprodutivel (semente fixa)."""
    arquivos = listar_imagens(folder)
    random.seed(SEED)
    random.shuffle(arquivos)
    if len(arquivos) < n:
        print(f"  ATENCAO: '{folder.name}' tem {len(arquivos)} imagens (pedido: {n}). Usando todas.")
        return arquivos
    return arquivos[:n]


def carregar_imagens(folder, file_names):
    """Le e pre-processa as imagens (resize 224x224, normalizacao para [-1, 1])."""
    images, nomes = [], []
    for file_name in file_names:
        img = cv2.imread(str(Path(folder) / file_name))
        if img is None:
            print(f"  Erro ao carregar {file_name}, pulando...")
            continue
        img_resized = cv2.resize(img, IMG_SIZE)
        img_array = (np.asarray(img_resized, dtype=np.float32) / 127.5) - 1
        images.append(img_array)
        nomes.append(file_name)
    return np.array(images), nomes


def metricas_confusao(tp, tn, fp, fn):
    """Calcula as metricas basicas a partir da matriz de confusao."""
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0            # sensibilidade
    specificity = tn / (tn + fp) if (tn + fp) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return acc, precision, recall, specificity, f1


def auc_mann_whitney(y_true, y_score):
    """AUC pela estatistica de Mann-Whitney sobre os scores (equivale a area sob a ROC)."""
    pos = [s for t, s in zip(y_true, y_score) if t == 1]
    neg = [s for t, s in zip(y_true, y_score) if t == 0]
    if not pos or not neg:
        return 0.0
    wins = sum(1.0 if ps > ns else 0.5 if ps == ns else 0.0
               for ps in pos for ns in neg)
    return wins / (len(pos) * len(neg))


def avaliar(model, writer):
    tp = tn = fp = fn = 0
    total_por_classe = {}
    y_true, y_score = [], []

    metas = {"PNEUMONIA": N_PNEUMONIA, "NORMAL": N_NORMAL}
    for label in ("PNEUMONIA", "NORMAL"):
        folder_path = TEST_DIR / label
        selecionados = selecionar(folder_path, metas[label])
        images, file_names = carregar_imagens(folder_path, selecionados)
        total_por_classe[label] = len(file_names)
        if len(images) == 0:
            continue

        print(f"\nTestando {len(images)} imagens da classe '{label}'...")
        for i in range(0, len(images), BATCH_SIZE):
            batch = images[i:i + BATCH_SIZE]
            batch_files = file_names[i:i + BATCH_SIZE]
            for file_name, pred in zip(batch_files, model.predict(batch)):
                index = int(np.argmax(pred))
                predicted_class = CLASS_NAMES[index]
                confidence = float(pred[index]) * 100
                pneumonia_score = float(pred[0])          # indice 0 = pneumonia
                real_class = label.lower()

                y_true.append(1 if real_class == "pneumonia" else 0)
                y_score.append(pneumonia_score)

                if real_class == "pneumonia":
                    tp += predicted_class == "pneumonia"
                    fn += predicted_class != "pneumonia"
                else:
                    tn += predicted_class == "normal"
                    fp += predicted_class != "normal"

                writer.writerow([
                    file_name, label, predicted_class,
                    f"{confidence:.2f}", f"{pneumonia_score:.4f}",
                    "ACERTO" if predicted_class == real_class else "ERRO",
                ])

    acc, precision, recall, specificity, f1 = metricas_confusao(tp, tn, fp, fn)
    auc = auc_mann_whitney(y_true, y_score)

    print("\nMetricas finais:")
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f" Acuracia:       {acc*100:.2f}%")
    print(f" Precisao:       {precision*100:.2f}%")
    print(f" Sensibilidade:  {recall*100:.2f}%")
    print(f" Especificidade: {specificity*100:.2f}%")
    print(f" F1-Score:       {f1*100:.2f}%")
    print(f" AUC:            {auc:.4f}")

    # Metricas para um limiar alternativo, reaproveitando os scores ja coletados.
    tp2 = sum(t == 1 and s >= LIMIAR_ALT for t, s in zip(y_true, y_score))
    fn2 = sum(t == 1 and s <  LIMIAR_ALT for t, s in zip(y_true, y_score))
    fp2 = sum(t == 0 and s >= LIMIAR_ALT for t, s in zip(y_true, y_score))
    tn2 = sum(t == 0 and s <  LIMIAR_ALT for t, s in zip(y_true, y_score))
    acc2, precision2, recall2, specificity2, f12 = metricas_confusao(tp2, tn2, fp2, fn2)

    print(f"\nAnalise de limiar (limiar = {LIMIAR_ALT}):")
    print(f"TP={tp2} TN={tn2} FP={fp2} FN={fn2}")
    print(f" Acuracia:       {acc2*100:.2f}%")
    print(f" Precisao:       {precision2*100:.2f}%")
    print(f" Sensibilidade:  {recall2*100:.2f}%")
    print(f" Especificidade: {specificity2*100:.2f}%")
    print(f" F1-Score:       {f12*100:.2f}%")

    # Resumo no CSV
    writer.writerow([])
    writer.writerow(["Resumo"])
    writer.writerow(["TP", "TN", "FP", "FN"])
    writer.writerow([tp, tn, fp, fn])
    writer.writerow([])
    writer.writerow(["Metrica", "Valor (%)"])
    writer.writerow(["Acuracia", f"{acc*100:.2f}"])
    writer.writerow(["Precisao (Pneumonia)", f"{precision*100:.2f}"])
    writer.writerow(["Sensibilidade (Recall)", f"{recall*100:.2f}"])
    writer.writerow(["Especificidade", f"{specificity*100:.2f}"])
    writer.writerow(["F1-Score", f"{f1*100:.2f}"])
    writer.writerow(["AUC", f"{auc:.4f}"])
    writer.writerow([])
    writer.writerow([f"Analise de limiar (limiar = {LIMIAR_ALT})"])
    writer.writerow(["TP", "TN", "FP", "FN"])
    writer.writerow([tp2, tn2, fp2, fn2])
    writer.writerow(["Metrica", "Valor (%)"])
    writer.writerow(["Acuracia", f"{acc2*100:.2f}"])
    writer.writerow(["Precisao (Pneumonia)", f"{precision2*100:.2f}"])
    writer.writerow(["Sensibilidade (Recall)", f"{recall2*100:.2f}"])
    writer.writerow(["Especificidade", f"{specificity2*100:.2f}"])
    writer.writerow(["F1-Score", f"{f12*100:.2f}"])
    writer.writerow([])
    writer.writerow(["Total de imagens por classe"])
    for label, n in total_por_classe.items():
        writer.writerow([label, n])
    return total_por_classe


def main():
    model = load_model(str(MODEL_PATH), compile=False)
    print(f"Modelo carregado de: {MODEL_PATH}")
    print(f"Fonte: {FONTE}  |  base: {TEST_DIR}")

    results_file = proximo_arquivo_saida()
    with open(results_file, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Arquivo", "Classe Real", "Classe Predita",
                         "Confianca (%)", "Score Pneumonia", "Resultado"])
        avaliar(model, writer)

    print(f"\nResultados salvos em: {results_file}")


if __name__ == "__main__":
    main()
