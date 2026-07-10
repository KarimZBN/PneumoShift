"""PneumoShift — avaliacao por inferencia de um classificador binario de pneumonia.

Pacote com os componentes reutilizaveis pelos scripts em scripts/ e testes em tests/:
    paths       resolucao de caminhos do projeto
    preprocess  redimensionamento (padding letterbox) e preparo da entrada
    data        selecao reprodutivel das imagens de teste
    metrics     matriz de confusao, AUC e rotulagem VP/VN/FP/FN
    gradcam     Grad-CAM sobre o MobileNetV2 aninhado
"""
from . import paths, preprocess, data, metrics, gradcam

__all__ = ["paths", "preprocess", "data", "metrics", "gradcam"]
