"""
Demonstracao interativa: classifica uma unica imagem e mostra o resultado.

Sorteia uma imagem da base/classe configurada, roda a inferencia e exibe a
predicao sobre a radiografia. Util para inspecao visual; a avaliacao completa
das metricas fica em main_batch.py.
"""
from pathlib import Path
import os
import random

import cv2
import cvzone
import numpy as np
from keras.models import load_model

# --- Configuracao (o que avaliar) ---
BASE = "RSNA"        # "kaggle" (Chest X-Ray) ou "rsna"
CLASSE = "NORMAL"    # "aleatorio" | "PNEUMONIA" | "NORMAL"

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "modelo" / "keras_model.h5"
CLASS_NAMES = ["pneumonia", "normal"]

model = load_model(str(MODEL_PATH), compile=False)

# Pasta da base escolhida (dados/test/kaggle ou dados/test/rsna).
pasta_teste = str(BASE_DIR / "dados" / "test" / BASE)
if not os.path.isdir(pasta_teste):
    raise FileNotFoundError(f"Pasta da base nao encontrada: {pasta_teste}")

# Seleciona a subpasta (classe): aleatoria ou a definida em CLASSE.
subpastas = [f for f in os.listdir(pasta_teste) if os.path.isdir(os.path.join(pasta_teste, f))]
if CLASSE.lower() == "aleatorio":
    pasta_escolhida = random.choice(subpastas)
elif CLASSE in subpastas:
    pasta_escolhida = CLASSE
else:
    raise ValueError(f"Classe '{CLASSE}' nao existe em {pasta_teste}. Opcoes: {subpastas}")

# Sorteia uma imagem da classe escolhida.
imagens = os.listdir(os.path.join(pasta_teste, pasta_escolhida))
imagem_escolhida = random.choice(imagens)
caminho_imagem = os.path.join(pasta_teste, pasta_escolhida, imagem_escolhida)
print(f"Base: {BASE}  |  Classe real: {pasta_escolhida}  |  Arquivo: {imagem_escolhida}")

# Pre-processamento (mesmo pipeline do batch: resize 224x224, normalizacao [-1, 1]).
img = cv2.imread(caminho_imagem)
image = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
image = (image / 127.5) - 1

# Inferencia.
prediction = model.predict(image)
index = np.argmax(prediction)
classe = CLASS_NAMES[index]
confidence_score = prediction[0][index]

texto1 = f"Class: {classe}"
texto2 = f"Score: {str(np.round(confidence_score * 100))[:-2]} %"
print(texto1, texto2)

# Exibicao em tamanho fixo para o texto ficar sempre proporcional.
display = cv2.resize(img, (700, 700))
cvzone.putTextRect(display, texto1, (30, 50), scale=2, thickness=2)
cvzone.putTextRect(display, texto2, (30, 100), scale=2, thickness=2)

cv2.imshow("IMG", display)
cv2.waitKey(0)
cv2.destroyAllWindows()
