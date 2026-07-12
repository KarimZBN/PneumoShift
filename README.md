# PneumoShift

Avaliação por inferência de um classificador binário de pneumonia (modelo Keras `.h5`,
MobileNetV2) em radiografias de tórax. O modelo é aplicado a duas bases de teste — Chest
X-Ray (Kaggle, pediátrica, interna) e RSNA (adulta, externa) — e o código calcula as
métricas de cada base, avalia calibração e intervalos de confiança, verifica o pipeline de
conversão DICOM e gera mapas de explicabilidade (Grad-CAM).

> **Classe positiva:** pneumonia (índice 0 da saída do modelo). Todas as métricas de
> sensibilidade/recall referem-se à detecção de pneumonia.

Para reproduzir os testes, veja [REPRODUCAO.md](REPRODUCAO.md).

## Estrutura de diretórios

```
src/pneumoshift/          Pacote com o codigo reutilizavel
  paths.py                  Caminhos do projeto + organizacao das saidas (prefixos, LEIA)
  preprocess.py             Redimensionamento (padding letterbox) e preparo da entrada
  data.py                   Selecao reprodutivel e leitura dos rotulos da RSNA
  metrics.py                Matriz de confusao, AUC, Brier, ECE (via scikit-learn)
  inferencia.py             Inferencia sobre uma pasta de imagens
  scores.py                 Cache de scores por paciente + avaliacao de subconjuntos
  gradcam.py                Grad-CAM sobre o MobileNetV2 aninhado

scripts/                  Executaveis
  converter_rsna.py         DICOM -> PNG da base RSNA INTEIRA (por paciente) [processed/rsna]
  avaliar_rsna_completa.py  Avaliacao PRINCIPAL: base inteira, distribuicao real, limiar 0,5
  split_rsna.py             Gera CSVs de selecao (balanceado, pareado) sem copiar imagens
  avaliar_rsna_balanceado.py  Apoio: balanceado 1:1, N sementes, media +- desvio
  avaliar_rsna_pareado.py   Apoio: pareado a Chest X-Ray (390/234), N sementes
  avaliar_lote.py           Avalia a Chest X-Ray (amostra pareada) -> PRINCIPAL_cxray
  analise_estatistica.py    Funde IC/calibracao no metricas.json da cxray + comparacao geometria
  gradcam_analise.py        Grad-CAM em lote por categoria + medida de foco (periferia x miolo)
  demo_imagem.py            Classifica 1 imagem + Grad-CAM
  inspecionar_dicom.py      Metadados DICOM (prova de conversao sem perdas)
  comparar_dicom_png.py     Orientacao DICOM x PNG + aspect ratio das bases

dados/
  raw/rsna/                 DICOM originais + stage_2_train_labels.csv
  raw/chest_xray/           Chest X-Ray original (Kaggle)
  processed/rsna/           RSNA inteira convertida em PNG (fonte unica) + scores_por_id.csv
  test/cxray/               Chest X-Ray (JPEGs, conjunto de teste)
  splits/                   CSVs de selecao por semente (balanceado, pareado)

resultados/               Saidas organizadas por PAPEL (prefixo no nome da pasta)
  PRINCIPAL_*/              resultados que vao para o artigo/defesa
  APOIO_*/                  analises complementares (robustez, explicabilidade, provas)
  DESCARTAVEL_*/            exploracao nao apresentada (ex: geometria esticar)
    (cada pasta: LEIA.md + metricas/resumo + imagens/)

modelo/keras_model.h5     Modelo pre-treinado avaliado
requirements.txt          Dependencias (versoes fixadas)
REPRODUCAO.md             Guia passo a passo
```

Não versionados (`.gitignore`): `dados/` (imagens, baixadas à parte) e o conteúdo de
`resultados/`. Apenas a estrutura de pastas de `resultados/` é preservada (`.gitkeep`).

## Instalação

```
pip install -r requirements.txt
```

Ambiente de referência: **Python 3.11**, TensorFlow/Keras 2.12.0, NumPy 1.23.5,
OpenCV 4.11.0.86, scikit-learn 1.3.2, matplotlib 3.7.5, pydicom 3.0.2 (com pylibjpeg
para DICOM comprimidos). A série NumPy 1.x é exigida pelo TensorFlow 2.12. As versões
completas estão em [`requirements.txt`](requirements.txt).

## Bases: origem, obtenção e critérios

**Chest X-Ray Images (Pneumonia) — Kaggle (interna).** Radiografias pediátricas; usa-se o
conjunto de teste (234 normais + 390 pneumonia), o mesmo utilizado por Shao (2021). Baixada
diretamente do Kaggle. Rótulos originais atribuídos por especialistas.

**RSNA Pneumonia Detection Challenge (externa).** Radiografias de adultos em DICOM. Baixada
do desafio RSNA (imagens em `stage_2_train_images/`, rótulos em `stage_2_train_labels.csv`).

**Agregação por paciente (RSNA).** Cada paciente corresponde a uma única imagem (`.dcm`); as
30.227 linhas do `labels.csv` são *bounding boxes* (3.543 pacientes têm mais de uma caixa).
Os rótulos são agregados por `patientId` **antes** de contar/converter, para que 1 paciente =
1 unidade.

**Critérios de inclusão/exclusão (RSNA).** `Target = 1` → pneumonia; `Target = 0` → ausência
de anotação de opacidade compatível com pneumonia (**não** necessariamente exame normal).
Pacientes presentes nas duas classes (ambíguos) são **descartados**. Contagem elegível:

| Classe | Pacientes |
|---|---:|
| Pneumonia (Target=1) | 6.012 |
| Normal (Target=0) | 20.672 |
| Ambíguos (descartados) | 0 |
| **Total elegível** | **26.684** |

> A classe negativa da RSNA **não** é equivalente à normal da Chest X-Ray; mesmo pareando as
> quantidades, permanecem diferenças de população, aquisição, instituição e rotulagem.

## Pipeline de conversão DICOM (RSNA)

`converter_rsna.py` lê cada DICOM com **pydicom**, normaliza a matriz de pixels para a escala
0–255 (min-max) e salva como PNG **no tamanho original** (sem redimensionar), organizado por
classe em `dados/processed/rsna/`. A conversão é sem perdas de orientação: os metadados
são 100% `MONOCHROME2` e `BitsStored = 8`, sem Rescale Slope/Intercept, VOI LUT ou windowing
a aplicar (verificado por `inspecionar_dicom.py`).

## Pré-processamento e redimensionamento

A entrada do modelo é 224×224, normalizada para [-1, 1]. A geometria é aplicada no
pré-processamento da inferência (não fixada no disco): **padding letterbox** (preserva a
proporção, adotado) ou **esticar** (resize direto, exploratório). A Chest X-Ray tem AR médio
1,46 (99,75% não-quadradas) e a RSNA 1,00 — o padding evita deformar a anatomia.

## Modelo e pesos

MobileNetV2 pré-treinado, exportado do Teachable Machine, em `modelo/keras_model.h5`. O
código apenas realiza **inferência** (sem reajuste de pesos), o que garante saídas idênticas
entre execuções.

## Avaliações e comandos

| Avaliação | Comando | Saída |
|---|---|---|
| RSNA base inteira (principal) | `python scripts/avaliar_rsna_completa.py` | `PRINCIPAL_rsna-base-inteira/` |
| RSNA balanceado 1:1 (apoio) | `python scripts/avaliar_rsna_balanceado.py` | `APOIO_rsna-balanceado/` |
| RSNA pareado à cxray (apoio) | `python scripts/avaliar_rsna_pareado.py` | `APOIO_rsna-pareado-cxray/` |
| Chest X-Ray (interna) | `python scripts/avaliar_lote.py` | `PRINCIPAL_cxray/` |
| Explicabilidade | `python scripts/gradcam_analise.py` | `APOIO_explicabilidade/` |
| Provas DICOM | `python scripts/inspecionar_dicom.py` / `comparar_dicom_png.py` | `APOIO_provas-dicom/` |

Pré-requisitos: `converter_rsna.py` (uma vez) e `avaliar_rsna_completa.py` (gera o cache)
antes de `split_rsna.py` e das avaliações de subconjunto. Ordem detalhada em REPRODUCAO.md.

**Limiar de decisão:** 0,5 (padrão do modelo) em todas as avaliações. Não há otimização de
limiar. **Intervalos de confiança:** bootstrap (2000 reamostragens), com o **paciente** como
unidade de reamostragem (na RSNA, paciente = imagem).

## Sementes aleatórias

- `SEED = 42` — seleção pareada da Chest X-Ray e do pareado da RSNA; bootstrap.
- Sementes `42..51` — as 10 repetições dos conjuntos balanceado e pareado da RSNA.
- `converter_rsna.py` não sorteia (converte a base inteira).

## Arquivos de saída esperados

Cada pasta de resultado contém um **`LEIA.md`** (números-chave, explicação do que é cada
arquivo). Avaliações produzem `metricas.json`/`resumo.json`, `predicoes.csv`/`rodadas.csv` e
uma subpasta `imagens/` (ROC, calibração).

## Licença

Distribuído sob a licença MIT. Veja [LICENSE](LICENSE).
