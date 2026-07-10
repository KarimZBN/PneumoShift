# PneumoShift

Avaliação por inferência de um classificador binário de pneumonia (modelo Keras `.h5`,
MobileNetV2) em radiografias de tórax. O modelo é aplicado a dois conjuntos de teste —
Kaggle (*Chest X-Ray*, pediátrico) e RSNA (adulto) — e o código calcula as métricas de
cada base, compara o efeito da geometria de entrada, verifica o pipeline de conversão
DICOM e gera mapas de explicabilidade (Grad-CAM). Trabalho de Conclusão de Curso.

> Para reproduzir os testes, veja [REPRODUCAO.md](REPRODUCAO.md).

## Estrutura

```
src/pneumoshift/          Pacote com o codigo reutilizavel
  paths.py                  Resolucao de caminhos do projeto
  preprocess.py             Redimensionamento (padding letterbox) e preparo da entrada
  data.py                   Selecao reprodutivel das imagens de teste (sorted + seed)
  metrics.py                Matriz de confusao, AUC e rotulagem VP/VN/FP/FN
  gradcam.py                Grad-CAM sobre o MobileNetV2 aninhado

scripts/                  Executaveis (o que se roda)
  converter_rsna.py         DICOM -> PNG (resize direto)         [dados/test/rsna]
  converter_rsna_padding.py DICOM -> PNG (padding letterbox)     [dados/test/rsna_padding]
  avaliar_lote.py           Avaliacao em lote de uma base -> CSV de metricas
  demo_imagem.py            Classifica 1 imagem + Grad-CAM (inspecao visual)
  gradcam_lote.py           Grad-CAM em lote por categoria (VP/VN/FP/FN)
  analise_foco.py           Mede foco do Grad-CAM (periferia x miolo) por categoria
  inspecionar_dicom.py      Verifica metadados DICOM (Photometric, bits, VOI LUT...)
  comparar_dicom_png.py     Prova de orientacao DICOM x PNG + aspect ratio das bases

tests/
  test_gradcam.py           Valida que o Grad-CAM opera sobre o forward correto

modelo/keras_model.h5     Modelo pre-treinado avaliado
requirements.txt          Dependencias
REPRODUCAO.md             Guia passo a passo
```

Não versionados (ver `.gitignore`): `dados/` (imagens, baixadas à parte) e
`resultados/` (CSVs e figuras gerados pela execução).

## Componentes

**Pacote `pneumoshift`** — concentra a lógica compartilhada, evitando duplicação. O
pré-processamento (padding letterbox 224×224), a seleção reprodutível das amostras
(ordenação + semente fixa, para não depender da ordem do sistema de arquivos), as métricas
e o Grad-CAM ficam definidos uma única vez e são importados pelos scripts.

**Avaliação** — `avaliar_lote.py` seleciona uma amostra pareada de 234 normais + 390
pneumonia (a mesma proporção do conjunto de teste do Kaggle usado por Shao, 2021), roda a
inferência e grava um CSV com a predição por imagem e o resumo (acurácia, precisão,
sensibilidade, especificidade, F1 e AUC). Troque `FONTE` no topo do arquivo.

**Geometria** — a entrada usa padding letterbox (preserva a proporção). `converter_rsna_padding.py`
gera a variante com padding aplicado já na conversão DICOM, para comparação com o resize direto.

**Explicabilidade** — `gradcam_lote.py` gera os mapas Grad-CAM separados por categoria de
acerto/erro; `analise_foco.py` quantifica objetivamente onde a ativação se concentra;
`test_gradcam.py` confirma que o Grad-CAM está sobre o caminho de inferência correto.

**Verificação do pipeline DICOM** — `inspecionar_dicom.py` e `comparar_dicom_png.py`
documentam que a conversão é sem perdas e não altera a orientação da imagem.

## Sementes

`converter_rsna.py` / `converter_rsna_padding.py` → `SEED = 13`;
`avaliar_lote.py` e demais → `SEED = 42`.

## Licença

Distribuído sob a licença MIT. Veja [LICENSE](LICENSE).
