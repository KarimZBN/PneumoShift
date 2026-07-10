"""Inferencia do classificador sobre uma pasta {NORMAL,PNEUMONIA} (fonte unica)."""
import cv2
import numpy as np

from .preprocess import redimensionar_por_geometria, preparar_entrada


def inferir_pasta(model, pasta, geometria="padding", batch_size=100):
    """Roda o modelo sobre todas as imagens de pasta/{PNEUMONIA,NORMAL}.

    Devolve (arquivos, y_true, y_score) alinhados por indice.
    y_true = 1 se PNEUMONIA. y_score = score cru da classe pneumonia (indice 0).
    """
    arquivos, y_true, y_score = [], [], []
    for label in ("PNEUMONIA", "NORMAL"):
        d = pasta / label
        if not d.is_dir():
            continue
        nomes = sorted(p.name for p in d.iterdir()
                       if p.suffix.lower() in (".png", ".jpeg", ".jpg"))
        imgs = []
        validos = []
        for nome in nomes:
            img = cv2.imread(str(d / nome))
            if img is None:
                continue
            imgs.append(preparar_entrada(redimensionar_por_geometria(img, geometria))[0])
            validos.append(nome)
        if not imgs:
            continue
        imgs = np.array(imgs)
        for i in range(0, len(imgs), batch_size):
            preds = model.predict(imgs[i:i + batch_size], verbose=0)
            for nome, pred in zip(validos[i:i + batch_size], preds):
                arquivos.append(nome)
                y_true.append(1 if label == "PNEUMONIA" else 0)
                y_score.append(float(pred[0]))
    return arquivos, np.array(y_true), np.array(y_score)
