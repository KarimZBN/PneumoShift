# PneumoShift

Esse repositório contem os scripts em Python para avaliar, por inferência, um classificador binário de
pneumonia (modelo Keras `.h5`) em radiografias de tórax. O modelo é aplicado a
dois conjuntos de teste — Kaggle (*Chest X-Ray*) e RSNA — e o código calcula as
métricas de cada base a partir da matriz de confusão usados no meu Trabalho de Conclusão de curso

> Para reproduzir os testes, veja [REPRODUCAO.md](REPRODUCAO.md).

## Conteúdo

```
src/
  converter_rsna.py   Converte os DICOM do RSNA em PNG (dados/test/rsna)
  main_batch.py       Avalia uma base em lote e grava um CSV com as métricas
  main.py             Classifica e exibe uma única imagem (demonstração visual)
modelo/
  keras_model.h5      Modelo pré-treinado avaliado
requirements.txt      Dependências (TensorFlow/Keras, OpenCV, pydicom, ...)
REPRODUCAO.md         Guia passo a passo de execução
```

Não versionados (ver `.gitignore`): `dados/` (imagens, baixadas à parte) e
`resultados/` (CSVs gerados pela execução).

## O que cada script faz

- **`converter_rsna.py`** — lê os rótulos do RSNA, sorteia imagens de cada classe
  com semente fixa (`SEED = 13`), converte de DICOM para PNG (normalização 0–255,
  resize 224×224) e grava em `dados/test/rsna/{NORMAL,PNEUMONIA}`.
- **`main_batch.py`** — carrega o modelo, seleciona uma amostra pareada de
  234 normais + 390 pneumonia (`SEED = 42`), roda a inferência em lote e grava um
  CSV com a predição por imagem e o resumo das métricas (acurácia, precisão,
  sensibilidade, especificidade, F1 e AUC). Troque `FONTE` para `"kaggle"` ou
  `"rsna"` no início do arquivo.
- **`main.py`** — classifica uma imagem sorteada e mostra o resultado sobre a
  radiografia, para inspeção visual.

## Licença

Distribuído sob a licença MIT. Veja [LICENSE](LICENSE).
