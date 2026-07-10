"""Pre-processamento de imagens para o classificador (fonte unica)."""
import cv2
import numpy as np

IMG_SIZE = (224, 224)


def redimensionar(img, size=IMG_SIZE):
    """Redimensiona para `size` com padding letterbox preto, preservando a proporcao.

    A imagem e reduzida pelo maior lado (escala = min(224/w, 224/h)) e o quadro
    224x224 e completado com bordas pretas centralizadas, sem deformar a anatomia.
    Interpolacao: INTER_AREA na reducao, INTER_LINEAR na ampliacao.
    """
    largura_alvo, altura_alvo = size
    h, w = img.shape[:2]
    escala = min(largura_alvo / w, altura_alvo / h)
    novo_w, novo_h = max(1, round(w * escala)), max(1, round(h * escala))
    interp = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
    redimensionada = cv2.resize(img, (novo_w, novo_h), interpolation=interp)

    dw, dh = largura_alvo - novo_w, altura_alvo - novo_h
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    return cv2.copyMakeBorder(redimensionada, top, bottom, left, right,
                              borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))


def preparar_entrada(img_bgr):
    """Converte uma imagem BGR (ja redimensionada) no tensor de entrada do modelo.

    Normaliza para [-1, 1] e da o shape (1, 224, 224, 3).
    """
    arr = np.asarray(img_bgr, dtype=np.float32)
    return ((arr / 127.5) - 1).reshape(1, IMG_SIZE[1], IMG_SIZE[0], 3)
