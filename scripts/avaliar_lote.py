"""
Avalia a Chest X-Ray sobre a amostra pareada (234 normais + 390 pneumonia), limiar 0,5.

Para cada imagem grava o score da classe pneumonia (usado no AUC); as metricas derivam da
matriz de confusao. A estatistica (IC bootstrap, calibracao) e acrescentada ao metricas.json
por analise_estatistica.py.

Uso:
    python scripts/avaliar_lote.py      (ajuste BASE e GEOMETRIA no topo)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv
import json

import cv2
import numpy as np
from keras.models import load_model

from pneumoshift import paths
from pneumoshift.preprocess import redimensionar_por_geometria, preparar_entrada
from pneumoshift.data import selecionar, N_NORMAL, N_PNEUMONIA
from pneumoshift import metrics as M
from pneumoshift.metrics import CLASS_NAMES, LIMIAR


# --- Configuracao ---
BASE = "cxray"                 # base interna (a RSNA tem scripts avaliar_rsna_*)
GEOMETRIA = "padding"          # "padding" (letterbox) ou "esticar" (resize direto)
BATCH_SIZE = 100
# Amostra pareada: 234 normais + 390 pneumonia (N_NORMAL/N_PNEUMONIA em pneumoshift/data.py).

# Pasta de teste (helper em paths). Os PNGs/JPEGs estao no tamanho original em ambas as
# bases; a geometria (padding/esticar) e aplicada aqui, no pre-processamento — de forma
# identica para cxray e rsna (sem geometria "assada" no disco).
TEST_DIR = paths.pasta_dados(BASE)
PREPROC = GEOMETRIA



def carregar_imagens(folder, file_names):
    """Le e pre-processa as imagens (geometria + normalizacao [-1, 1])."""
    images, nomes = [], []
    for file_name in file_names:
        img = cv2.imread(str(Path(folder) / file_name))
        if img is None:
            print(f"  Erro ao carregar {file_name}, pulando...")
            continue
        img_r = redimensionar_por_geometria(img, PREPROC)
        images.append(preparar_entrada(img_r)[0])
        nomes.append(file_name)
    return np.array(images), nomes


def coletar_predicoes(model):
    """Roda a inferencia sobre a amostra pareada e devolve as predicoes por imagem."""
    linhas = []                 # (arquivo, classe_real, classe_predita, confianca, score, resultado)
    y_true, y_score = [], []
    total_por_classe = {}

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
                score = float(pred[0])                     # indice 0 = pneumonia
                real_class = label.lower()

                y_true.append(1 if real_class == "pneumonia" else 0)
                y_score.append(score)
                linhas.append([
                    file_name, label, predicted_class,
                    f"{confidence:.2f}", repr(score),      # score CRU (sem arredondar)
                    "ACERTO" if predicted_class == real_class else "ERRO",
                ])
    return linhas, y_true, y_score, total_por_classe


def calcular_metricas(y_true, y_score, limiar=LIMIAR):
    """Metricas finais via scikit-learn (AUC, Brier, ECE) + matriz por limiar."""
    pto = M.metricas_por_limiar(y_true, y_score, limiar)
    return {
        **pto,
        "auc": M.auc_sklearn(y_true, y_score),
        "brier": M.brier(y_true, y_score),
        "ece": M.ece(y_true, y_score),
    }


def escrever_csv(caminho, met, linhas, total_por_classe, y_score):
    """Escreve o CSV com o RESUMO no topo, seguido da tabela de predicoes por imagem."""
    sat = sum(1 for s in y_score if s >= 1.0)
    with open(caminho, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        # --- RESUMO (topo) ---
        w.writerow([f"Resumo - {BASE} {GEOMETRIA} (limiar {LIMIAR})"])
        w.writerow(["Metrica", "Valor"])
        w.writerow(["Acuracia (%)", f"{met['acuracia']*100:.2f}"])
        w.writerow(["Sensibilidade (%)", f"{met['sensibilidade']*100:.2f}"])
        w.writerow(["Especificidade (%)", f"{met['especificidade']*100:.2f}"])
        w.writerow(["F1-Score (%)", f"{met['f1']*100:.2f}"])
        w.writerow(["AUC (sklearn)", f"{met['auc']:.4f}"])
        w.writerow(["Brier score", f"{met['brier']:.4f}"])
        w.writerow(["ECE (10 bins)", f"{met['ece']:.4f}"])
        w.writerow(["TP / TN / FP / FN", f"{met['tp']} / {met['tn']} / {met['fp']} / {met['fn']}"])
        w.writerow(["Scores saturados (==1,0)", f"{sat} de {len(y_score)}"])
        for label, n in total_por_classe.items():
            w.writerow([f"Total {label}", n])
        w.writerow([])
        # --- TABELA por imagem ---
        w.writerow(["Arquivo", "Classe Real", "Classe Predita",
                    "Confianca (%)", "Score Pneumonia", "Resultado"])
        w.writerows(linhas)


def escrever_json(caminho, met, total_por_classe, y_score):
    """Grava as metricas finais em JSON (leve, versionavel)."""
    sat = sum(1 for s in y_score if s >= 1.0)
    dados = {
        "base": BASE, "geometria": GEOMETRIA, "limiar": LIMIAR,
        "n": len(y_score), **{k: met[k] for k in ("tp", "tn", "fp", "fn")},
        "acuracia": met["acuracia"], "sensibilidade": met["sensibilidade"],
        "especificidade": met["especificidade"], "f1": met["f1"],
        "auc": met["auc"], "brier": met["brier"], "ece": met["ece"],
        "saturados": sat, "total_por_classe": total_por_classe,
    }
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)


def main():
    model = load_model(str(paths.MODELO), compile=False)
    print(f"Modelo carregado de: {paths.MODELO}")
    print(f"Execucao: {BASE}_{GEOMETRIA}  |  base lida: {TEST_DIR}")

    linhas, y_true, y_score, total = coletar_predicoes(model)
    met = calcular_metricas(y_true, y_score)

    print("\nMetricas finais (sklearn):")
    print(f"TP={met['tp']} TN={met['tn']} FP={met['fp']} FN={met['fn']}")
    print(f" Acuracia:       {met['acuracia']*100:.2f}%")
    print(f" Sensibilidade:  {met['sensibilidade']*100:.2f}%")
    print(f" Especificidade: {met['especificidade']*100:.2f}%")
    print(f" F1-Score:       {met['f1']*100:.2f}%")
    print(f" AUC:            {met['auc']:.4f}")
    print(f" Brier:          {met['brier']:.4f}")
    print(f" ECE:            {met['ece']:.4f}")

    # padding -> PRINCIPAL; esticar (comparacao de geometria) -> DESCARTAVEL.
    if GEOMETRIA == "esticar":
        prefixo, nome = "DESCARTAVEL", f"{BASE}-esticar"
    else:
        prefixo, nome = "PRINCIPAL", BASE
    OUT_DIR = paths.pasta_resultado(prefixo, nome)
    escrever_csv(OUT_DIR / "predicoes.csv", met, linhas, total, y_score)
    escrever_json(OUT_DIR / "metricas.json", met, total, y_score)
    print(f"\nResultados salvos em: {OUT_DIR}/")

    # Estatistica desta execucao: acrescenta AUC/Brier/ECE + IC bootstrap ao metricas.json
    # e gera imagens/roc.png e imagens/calibracao.png. Import tardio (matplotlib/sklearn so
    # quando roda); tolerante a falha para nao perder o CSV/JSON ja gravados.
    try:
        from analise_estatistica import analisar_pasta
        analisar_pasta(BASE, GEOMETRIA, OUT_DIR)
        print(f"Estatistica adicionada ao metricas.json; figuras em {OUT_DIR}/imagens/")
    except Exception as e:
        print(f"[aviso] analise estatistica nao rodou ({e}). "
              f"Rode 'python scripts/analise_estatistica.py' depois.")

    paths.escrever_leia(
        OUT_DIR,
        f"Chest X-Ray — {GEOMETRIA}" if BASE == "cxray" else f"{BASE} — {GEOMETRIA}",
        ("Chest X-Ray (base interna), geometria padding." if prefixo == "PRINCIPAL"
         else "Chest X-Ray com geometria esticar (comparacao de geometria)."),
        "Avaliacao sobre a amostra pareada (234 normal + 390 pneumonia), limiar 0,5. "
        "A estatistica (IC bootstrap, calibracao) esta no metricas.json e em imagens/.",
        {"metricas.json": "metricas + IC bootstrap + saturados",
         "predicoes.csv": "predicao por imagem (score cru)",
         "imagens/roc.png": "curva ROC", "imagens/calibracao.png": "curva de calibracao"},
        {"AUC": f"{met['auc']:.4f}",
         "Acuracia": f"{met['acuracia']*100:.2f}%",
         "Sensibilidade": f"{met['sensibilidade']*100:.2f}%",
         "Especificidade": f"{met['especificidade']*100:.2f}%",
         "Brier": f"{met['brier']:.4f}", "Geometria": GEOMETRIA})


if __name__ == "__main__":
    main()
