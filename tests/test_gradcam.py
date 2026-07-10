"""
Teste de sanidade do Grad-CAM: a reconstrucao manual do forward (extrair out_relu e
reaplicar GAP -> denso -> softmax) reproduz EXATAMENTE a predicao do modelo original?

Se as predicoes baterem, o Grad-CAM opera sobre o forward correto e o mapa e valido.

Uso:
    python tests/test_gradcam.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

from pneumoshift import paths, gradcam
from pneumoshift.preprocess import redimensionar, preparar_entrada


def uma_imagem():
    """Primeira imagem disponivel em qualquer base/classe, para o teste."""
    for fonte in ("kaggle", "rsna"):
        for classe in ("PNEUMONIA", "NORMAL"):
            d = paths.DADOS_TESTE / fonte / classe
            if d.is_dir():
                arqs = [p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpeg", ".jpg")]
                if arqs:
                    return arqs[0]
    raise FileNotFoundError("Nenhuma imagem de teste encontrada.")


def main():
    model = load_model(str(paths.MODELO), compile=False)
    owner, conv, camadas = gradcam.preparar(model)
    print("Camadas reaplicadas apos a conv-alvo:", [l.name for l in camadas])

    caminho = uma_imagem()
    print("Imagem de teste:", caminho.name)
    x = preparar_entrada(redimensionar(cv2.imread(str(caminho))))

    pred_oficial = model.predict(x, verbose=0)[0]

    extrator = tf.keras.models.Model(owner.inputs, conv.output)
    a = extrator(x)
    for l in camadas:
        a = l(a)
    pred_reconstruido = np.array(a)[0]

    print("\noficial     :", np.round(pred_oficial, 5))
    print("reconstruido:", np.round(pred_reconstruido, 5))
    ok = np.allclose(pred_oficial, pred_reconstruido, atol=1e-4)
    print("\nBATEM (atol=1e-4)?", ok)
    if ok:
        print(">> Grad-CAM VALIDO: opera sobre o forward correto.")
    else:
        dif = float(np.max(np.abs(pred_oficial - pred_reconstruido)))
        print(f">> DIVERGENCIA (max abs = {dif:.5f}). A reconstrucao esta errada.")


if __name__ == "__main__":
    main()
