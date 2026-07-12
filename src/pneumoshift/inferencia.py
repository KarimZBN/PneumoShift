"""Inferencia do classificador sobre uma pasta {NORMAL,PNEUMONIA} (fonte unica)."""
import cv2
import numpy as np

from .preprocess import redimensionar_por_geometria, preparar_entrada


def inferir_pasta(model, pasta, geometria="padding", batch_size=100, progresso=True):
    """Roda o modelo sobre todas as imagens de pasta/{PNEUMONIA,NORMAL}, em streaming.

    Processa um lote de cada vez (carrega -> infere -> descarta), sem manter todas as
    imagens na memoria. Carregar a base RSNA inteira de uma vez custaria ~16 GB
    (26.684 x 224 x 224 x 3 x 4 bytes), inviavel em 16 GB de RAM; em streaming o pico cai
    para o tamanho de um lote (~0,06 GB). Imprime progresso a cada lote se `progresso`.

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
        total = len(nomes)
        if progresso and total:
            print(f"  {label}: {total} imagens", flush=True)

        feitas = 0
        for ini in range(0, total, batch_size):
            lote_nomes = nomes[ini:ini + batch_size]
            batch, validos = [], []
            for nome in lote_nomes:
                img = cv2.imread(str(d / nome))
                if img is None:
                    continue
                batch.append(preparar_entrada(redimensionar_por_geometria(img, geometria))[0])
                validos.append(nome)
            if not batch:
                continue
            preds = model.predict(np.array(batch), verbose=0)
            for nome, pred in zip(validos, preds):
                arquivos.append(nome)
                y_true.append(1 if label == "PNEUMONIA" else 0)
                y_score.append(float(pred[0]))
            feitas += len(validos)
            if progresso:
                print(f"    [{feitas:6}/{total}] {label}", flush=True)
    return arquivos, np.array(y_true), np.array(y_score)
