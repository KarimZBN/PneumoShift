"""Grad-CAM para o classificador (MobileNetV2 aninhado, exportado pelo Teachable Machine).

A ultima conv espacial (7x7x1280) e 'out_relu', dentro do submodelo MobileNetV2, por
sua vez aninhado no Sequential do modelo. A camada e localizada por nome; o caminho
apos ela (GlobalAveragePooling -> denso -> softmax) e reconstruido para calcular o
gradiente da classe alvo em relacao a ativacao convolucional.
"""
import cv2
import numpy as np
import tensorflow as tf

CONV_LAYER_NAME = "out_relu"


def encontrar_camada(model, nome=CONV_LAYER_NAME):
    """Procura a camada `nome` recursivamente e devolve (submodelo_dono, camada)."""
    for layer in model.layers:
        if layer.name == nome:
            return model, layer
        if hasattr(layer, "layers") and layer.layers:
            owner, found = encontrar_camada(layer, nome)
            if found is not None:
                return owner, found
    return None, None


def _contem_conv(bloco, nome=CONV_LAYER_NAME):
    if not (hasattr(bloco, "layers") and bloco.layers):
        return False
    return any(l.name == nome or _contem_conv(l, nome) for l in bloco.layers)


def camadas_apos(model, nome=CONV_LAYER_NAME):
    """Camadas do topo, em ordem, a aplicar sobre a ativacao da conv-alvo ate a saida.

    O bloco que contem a conv e substituido por suas camadas POSTERIORES ao sub-bloco
    convolucional (o GlobalAveragePooling); os demais blocos do topo entram inteiros.
    Reconstroi: ativacao 7x7x1280 -> GAP -> denso -> softmax.
    """
    camadas = []
    for topo in model.layers:
        if _contem_conv(topo, nome):
            depois = False
            for sub in topo.layers:
                if _contem_conv(sub, nome):
                    depois = True
                    continue
                if depois:
                    camadas.append(sub)
        elif hasattr(topo, "layers") and topo.layers:
            camadas.extend(topo.layers)
        else:
            camadas.append(topo)
    return camadas


def preparar(model, nome=CONV_LAYER_NAME):
    """Localiza a conv-alvo e as camadas posteriores uma unica vez (reuso no lote)."""
    owner, conv = encontrar_camada(model, nome)
    if conv is None:
        raise RuntimeError(
            f"Camada '{nome}' nao encontrada. Ajuste o nome "
            f"(alternativas comuns: 'Conv_1', 'out_relu').")
    return owner, conv, camadas_apos(model, nome)


def gerar(img_norm, class_index, owner, conv_layer, camadas_pos, size=(224, 224)):
    """Mapa Grad-CAM (HxW, valores 0..1) para a classe indicada.

    img_norm: tensor pre-processado, shape (1, 224, 224, 3), normalizado [-1, 1].
    owner/conv_layer/camadas_pos: saida de preparar(model).
    """
    extrator = tf.keras.models.Model(owner.inputs, conv_layer.output)
    with tf.GradientTape() as tape:
        conv_out = extrator(img_norm)
        tape.watch(conv_out)
        x = conv_out
        for layer in camadas_pos:
            x = layer(x)
        loss = x[:, class_index]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))         # peso por canal
    heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1)  # combinacao ponderada
    heatmap = tf.maximum(heatmap, 0)                       # ReLU
    maxv = tf.reduce_max(heatmap)
    if maxv > 0:
        heatmap = heatmap / maxv
    return cv2.resize(heatmap.numpy(), size)


def sobrepor(img_bgr, heatmap, alpha=0.4):
    """Sobrepoe o heatmap (0..1) colorido (JET) sobre a imagem BGR de mesmo tamanho."""
    hm = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return cv2.addWeighted(hm, alpha, img_bgr, 1 - alpha, 0)
