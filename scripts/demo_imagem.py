"""
Classifica uma unica imagem e sobrepoe o mapa Grad-CAM.

Sorteia uma imagem da base/classe configurada, roda a inferencia, exibe a predicao
sobre a radiografia e sobrepoe o Grad-CAM. A avaliacao em lote fica em avaliar_lote.py.

Uso:
    python scripts/demo_imagem.py       (ajuste BASE e CLASSE no topo)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import os
import random

import cv2
import numpy as np
from keras.models import load_model

from pneumoshift import paths, gradcam
from pneumoshift.preprocess import redimensionar, preparar_entrada
from pneumoshift.metrics import CLASS_NAMES

# --- Configuracao ---
BASE = "aleatorio"     # "aleatorio" | "kaggle" | "rsna"
CLASSE = "aleatorio"   # "aleatorio" | "PNEUMONIA" | "NORMAL"
BASES_VALIDAS = ("kaggle", "rsna")   # sorteaveis quando BASE == "aleatorio"
MOSTRAR_GRADCAM = True
SALVAR_SAIDA = True

GRADCAM_DIR = paths.RESULTADOS / "gradcam"


def main():
    model = load_model(str(paths.MODELO), compile=False)

    base = random.choice(BASES_VALIDAS) if BASE.lower() == "aleatorio" else BASE
    pasta_teste = paths.DADOS_TESTE / base
    if not pasta_teste.is_dir():
        raise FileNotFoundError(f"Pasta da base nao encontrada: {pasta_teste}")

    subpastas = [f for f in os.listdir(pasta_teste)
                 if (pasta_teste / f).is_dir()]
    if CLASSE.lower() == "aleatorio":
        pasta_escolhida = random.choice(subpastas)
    elif CLASSE in subpastas:
        pasta_escolhida = CLASSE
    else:
        raise ValueError(f"Classe '{CLASSE}' nao existe em {pasta_teste}. Opcoes: {subpastas}")

    imagens = os.listdir(pasta_teste / pasta_escolhida)
    imagem_escolhida = random.choice(imagens)
    caminho_imagem = pasta_teste / pasta_escolhida / imagem_escolhida
    print(f"Base: {base}  |  Classe real: {pasta_escolhida}  |  Arquivo: {imagem_escolhida}")

    img = cv2.imread(str(caminho_imagem))
    img_224 = redimensionar(img)
    entrada = preparar_entrada(img_224)

    prediction = model.predict(entrada)
    index = int(np.argmax(prediction))
    classe = CLASS_NAMES[index]
    score = float(prediction[0][index])
    print(f"Predicao: {classe}  |  score do modelo: {score*100:.1f}%")

    painel = img_224.copy()
    cv2.putText(painel, f"Real: {pasta_escolhida}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(painel, f"Pred: {classe} ({score*100:.0f}%)", (8, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    saida = painel
    if MOSTRAR_GRADCAM:
        try:
            owner, conv, camadas = gradcam.preparar(model)
            heat = gradcam.gerar(entrada, index, owner, conv, camadas)
            overlay = gradcam.sobrepor(img_224, heat)
            cv2.putText(overlay, "Grad-CAM", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            saida = np.hstack([painel, overlay])
        except Exception as exc:
            print(f"  [Grad-CAM] nao gerado: {exc}")

    if SALVAR_SAIDA:
        GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
        destino = GRADCAM_DIR / f"gradcam_{base}_{pasta_escolhida}_{Path(imagem_escolhida).stem}.png"
        cv2.imwrite(str(destino), saida)
        print(f"  Imagem salva em: {destino}")

    escala = 700 / saida.shape[0]
    display = cv2.resize(saida, (int(saida.shape[1] * escala), 700))
    cv2.imshow("PneumoShift - predicao + Grad-CAM", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
