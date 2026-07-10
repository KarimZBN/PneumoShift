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
  split_rsna.py             Separa a RSNA em validacao x pool de teste (sem vazamento) -> dados/splits/
  converter_rsna.py         DICOM -> PNG no TAMANHO ORIGINAL   [dados/test/rsna_validacao, rsna_pool]
  avaliar_lote.py           Avalia uma execucao (base+geometria) -> predicoes.csv + metricas.json
  avaliar_rsna_rodadas.py   Limiar em validacao separada + N rodadas com seeds -> media +- desvio
  analise_estatistica.py    AUC/calibracao/IC bootstrap + figuras (individuais e comparativas)
  demo_imagem.py            Classifica 1 imagem + Grad-CAM (inspecao visual)
  gradcam_lote.py           Grad-CAM em lote por categoria (VP/VN/FP/FN)
  analise_foco.py           Mede foco do Grad-CAM (periferia x miolo) por categoria
  inspecionar_dicom.py      Verifica metadados DICOM (Photometric, bits, VOI LUT...)
  comparar_dicom_png.py     Prova de orientacao DICOM x PNG + aspect ratio das bases

tests/
  test_gradcam.py           Valida que o Grad-CAM opera sobre o forward correto
  test_split.py             Valida que validacao e pool da RSNA sao disjuntos (zero vazamento)

resultados/               Saidas por execucao (nao versionadas; so a estrutura via .gitkeep)
  <base>/<geometria>_<timestamp>_seed<N>/
                            -> predicoes.csv, metricas.json, roc.png, calibracao.png
  gradcam/<base>_<geometria>_<timestamp>/
                            -> VP/VN/FP/FN (overlays), foco.csv, foco_resumo.csv, indice.csv
  _comparacoes/run_<timestamp>/
                            -> dominio/ (cxray vs rsna), geometria/ (padding vs esticar), resumo_geral.csv

modelo/keras_model.h5     Modelo pre-treinado avaliado
requirements.txt          Dependencias
REPRODUCAO.md             Guia passo a passo
```

Não versionados (ver `.gitignore`): `dados/` (imagens, baixadas à parte) e o conteúdo de
`resultados/` (CSVs e figuras gerados pela execução ficam local/nuvem). Apenas a **estrutura
de pastas** de `resultados/` é preservada no repositório (arquivos `.gitkeep`) — cada execução
tem sua pasta autocontida, identificada por base, geometria e timestamp.

## Componentes

**Pacote `pneumoshift`** — concentra a lógica compartilhada, evitando duplicação. O
pré-processamento (padding letterbox 224×224), a seleção reprodutível das amostras
(ordenação + semente fixa, para não depender da ordem do sistema de arquivos), as métricas
e o Grad-CAM ficam definidos uma única vez e são importados pelos scripts.

**Avaliação** — `avaliar_lote.py` seleciona uma amostra pareada de 234 normais + 390
pneumonia (a mesma proporção do conjunto de teste do Kaggle usado por Shao, 2021), roda a
inferência e grava um CSV com a predição por imagem e o resumo (acurácia, precisão,
sensibilidade, especificidade, F1 e AUC). Ajuste `BASE` e `GEOMETRIA` no topo do arquivo.

**Geometria** — a entrada usa padding letterbox (preserva a proporção), aplicado no
pré-processamento da inferência. As imagens são gravadas no tamanho original (os PNGs da RSNA
por `converter_rsna.py`, os JPEGs da Chest X-Ray originais), sem geometria fixada no disco;
assim a mesma pasta serve tanto ao padding (letterbox) quanto ao esticar (resize direto).

**Explicabilidade** — `gradcam_lote.py` gera os mapas Grad-CAM separados por categoria de
acerto/erro; `analise_foco.py` quantifica objetivamente onde a ativação se concentra;
`test_gradcam.py` confirma que o Grad-CAM está sobre o caminho de inferência correto.

**Verificação do pipeline DICOM** — `inspecionar_dicom.py` e `comparar_dicom_png.py`
documentam que a conversão é sem perdas e não altera a orientação da imagem.

## Sementes

`split_rsna.py` → `SEED_SPLIT = 20260710` (fixa a partição validação/pool; `converter_rsna.py`
apenas lê os splits, não sorteia); `avaliar_lote.py` e a seleção pareada → `SEED = 42`;
`avaliar_rsna_rodadas.py` → seeds `42..51` (uma por rodada).

## Licença

Distribuído sob a licença MIT. Veja [LICENSE](LICENSE).
