# Guia de reprodução dos testes

Passo a passo para reproduzir a avaliação descrita no [README](README.md).

## Requisitos

```
pip install -r requirements.txt
```

Ambiente de referência: Python 3.11, com as versões fixadas em `requirements.txt`
(TensorFlow/Keras 2.12.0, NumPy 1.23.5, OpenCV 4.11.0.86, scikit-learn 1.3.2,
matplotlib 3.7.5, pydicom 3.0.2). Detalhes de hardware e bibliotecas na seção
[Ambiente computacional](#ambiente-computacional).

## Dados

Os conjuntos são públicos e devem ser baixados diretamente das fontes:

- **Kaggle — Chest X-Ray Images (Pneumonia):** conjunto de teste pediátrico
  (234 normais + 390 pneumonia), o mesmo utilizado por Shao (2021).
- **RSNA Pneumonia Detection Challenge:** imagens em DICOM de pacientes adultos, com os
  rótulos em `stage_2_train_labels.csv`.

Estrutura esperada:

```
dados/
  raw/rsna/stage_2_train_images/*.dcm
  raw/rsna/stage_2_train_labels.csv
  test/cxray/{NORMAL,PNEUMONIA}/*.jpeg
  processed/rsna/{NORMAL,PNEUMONIA}/*.png     (gerado pela conversão; + scores_por_id.csv)
  splits/rsna_{balanceado,pareado}_seed<N>.csv  (gerado por split_rsna.py)
```

Observação sobre os rótulos do RSNA: `Target = 0` indica **ausência de anotação de
opacidade compatível com pneumonia**, e não necessariamente um exame normal. Portanto, a
classe negativa do RSNA não é equivalente à classe normal do Kaggle. Cada paciente = uma
imagem; os rótulos são agregados por `patientId` antes de contar/converter (as 30.227
linhas do CSV são *bounding boxes*). Elegíveis: 6.012 pneumonia + 20.672 normal = 26.684,
0 ambíguos.

## Passo a passo

Rode da raiz do projeto (os caminhos são resolvidos automaticamente). A ordem importa: a
avaliação da base inteira **gera o cache de scores** que o split e as avaliações de
subconjunto consomem.

### 1. Converter a base RSNA inteira (DICOM → PNG)

```
python scripts/converter_rsna.py
```

Converte todos os 26.684 pacientes elegíveis (agregados por `patientId`) para
`dados/processed/rsna/{NORMAL,PNEUMONIA}/`, **no tamanho original** (só normaliza 0–255). A
geometria não é fixada no disco — é aplicada no pré-processamento da inferência.

### 2. Avaliação PRINCIPAL: base inteira (gera o cache)

```
python scripts/avaliar_rsna_completa.py
```

Avalia toda a base (distribuição real das classes), limiar 0,5. Reporta sensibilidade,
especificidade, AUC, Brier, precisão e curva de calibração, com IC 95% por bootstrap
(unidade = paciente) e conferência manual das métricas a partir da matriz. Salva em
`resultados/PRINCIPAL_rsna-base-inteira/` e grava o cache
`dados/processed/rsna/scores_por_id.csv` (usado nos passos 3–5).

### 3. Gerar as seleções amostradas (CSVs, sem copiar imagens)

```
python scripts/split_rsna.py
```

A partir da base convertida, grava em `dados/splits/` os CSVs de identificadores por
semente: **balanceado 1:1** (todos os positivos + iguais negativos) e **pareado à Chest
X-Ray** (390 pneumonia + 234 normal). Não copia PNGs.

### 4. Avaliação de apoio: balanceado 1:1 (controla a proporção)

```
python scripts/avaliar_rsna_balanceado.py
```

Consulta o cache do passo 2 (sem reinferência), 10 sementes, limiar 0,5. Reporta média ±
desvio e IC 95% entre as sementes em `resultados/APOIO_rsna-balanceado/`.

### 5. Avaliação de apoio: pareado à Chest X-Ray

```
python scripts/avaliar_rsna_pareado.py
```

Igual ao anterior, mas com 390/234 (mesmo tamanho e proporção da base interna), para
comparar as duas bases isolando a origem dos dados. Salva em
`resultados/APOIO_rsna-pareado-cxray/`.

### 6. Chest X-Ray (base interna)

Em `scripts/avaliar_lote.py`, no topo: `BASE = "cxray"`, `GEOMETRIA = "padding"`.

```
python scripts/avaliar_lote.py
```

Avalia a amostra pareada (234 normais + 390 pneumonia). A estatística (IC bootstrap,
calibração) é acrescentada ao `metricas.json` e as figuras vão para `imagens/`. Salva em
`resultados/PRINCIPAL_cxray/`.

> (Opcional, exploratório) Rodando também com `GEOMETRIA = "esticar"` gera
> `resultados/DESCARTAVEL_cxray-esticar/`; depois `python scripts/analise_estatistica.py`
> produz a comparação de geometria padding × esticar.

### 7. Explicabilidade (Grad-CAM + foco)

Em `scripts/gradcam_analise.py`, ajuste `BASE` e `GEOMETRIA` e rode:

```
python scripts/gradcam_analise.py
```

Numa passada, gera os overlays Grad-CAM por categoria (VP/VN/FP/FN), nomeados
sequencialmente (`VP_001.png`, `FP_001.png`, ...), e mede o foco da ativação (periferia ×
miolo). Salva em `resultados/APOIO_explicabilidade/<base>_<geometria>/`: os overlays, o
`indice.csv` (nome sequencial → arquivo original, categoria, score, foco) e o
`foco_resumo.csv` (média por categoria).

### 8. Verificações do pipeline DICOM

```
python scripts/inspecionar_dicom.py     # metadados (conversão sem perdas)
python scripts/comparar_dicom_png.py     # orientação DICOM × PNG + aspect ratio
python tests/test_gradcam.py             # valida o Grad-CAM
```

Saem em `resultados/APOIO_provas-dicom/`.

## Reprodutibilidade

- A seleção das amostras usa sementes fixas (`SEED = 42`; sementes `42..51` nas 10
  repetições dos conjuntos balanceado e pareado), com ordenação prévia dos arquivos antes
  do embaralhamento (independe da ordem do sistema de arquivos). O `converter_rsna.py` não
  sorteia (converte a base inteira).
- O modelo apenas realiza inferência, sem reajuste de pesos, o que garante saídas idênticas
  entre execuções. Por isso, o cache de scores da base inteira produz, nos subconjuntos, os
  mesmos valores que uma reinferência — sem o custo de rodar o modelo de novo.
- Os scores são preservados sem arredondamento; as métricas via scikit-learn são conferidas
  manualmente a partir da matriz de confusão.

### Ambiente computacional

Os experimentos foram conduzidos em um notebook equipado com processador **AMD Ryzen 7
6800H**, **16 GB de memória RAM** e placa gráfica **NVIDIA GeForce RTX 3070 Ti**, com
implementação em **Python 3.11**.

O carregamento e a inferência do modelo Keras (`.h5`) apoiaram-se nas bibliotecas
**TensorFlow** e **Keras** (versão 2.12.0); o tratamento numérico das imagens utilizou a
**NumPy** (versão 1.23.5); a leitura e o redimensionamento das radiografias, a **OpenCV**
(versão 4.11.0.86); e a conversão da base RSNA, a **pydicom** (versão 3.0.2, com
`pylibjpeg` para os DICOM comprimidos). Cabe registrar que a versão do TensorFlow adotada
requer a série NumPy 1.x, condição observada na configuração do ambiente.

As análises estatísticas (AUC, calibração, intervalos de confiança por bootstrap e curvas)
utilizam o **scikit-learn** (versão 1.3.2), e as figuras (ROC, calibração) são geradas com
o **matplotlib** (versão 3.7.5). As versões completas estão em
[`requirements.txt`](requirements.txt).

> **Nota de implementação (avaliação da base inteira).** Avaliar as 26.684 imagens exige
> processá-las em *streaming* por lotes: carregar as imagens todas de uma vez custaria
> ~16 GB (26.684 × 224 × 224 × 3 × 4 bytes), inviável em 16 GB de RAM. A inferência carrega
> um lote de cada vez, guarda apenas o escore por imagem e descarta o lote, mantendo o pico
> de memória baixo independentemente do tamanho da base.

## Nota aos orientadores — reproduzindo e relacionando os números da devolutiva

Esta seção mostra como chegar aos números da devolutiva a partir do código atual. A
**reprodução exata é garantida na Chest X-Ray** (base interna, conjunto de teste fixo). Na
**RSNA** (base externa), a devolutiva pediu justamente avaliar toda a base e repetir a
seleção com várias sementes — o que foi feito —, então o valor de referência mudou, e a
nota documenta essa relação de forma transparente.

**RSNA — avaliação ampliada.** O valor original (AUC 0,7320) vinha de uma seleção única de
624 imagens da conversão antiga. O pipeline atual avalia a **base inteira** (26.684
pacientes, distribuição real) como resultado principal e, como apoio, **repete a seleção
balanceada com 10 sementes** (média ± desvio). Como a conversão da versão da banca foi
substituída por este pipeline, o 0,7320 não é reproduzível byte a byte; o número de
referência da base externa passa a ser o das avaliações atuais.

**Chest X-Ray — reprodução exata.** A versão atual (com `INTER_AREA`) dá AUC 0,9358, e o
trabalho original reportava 0,9469. A causa foi investigada e **confirmada por reprodução
direta**: é o método de **interpolação** do redimensionamento — **não** a amostra (as 624
imagens são as mesmas nas duas versões, pois o conjunto de teste tem exatamente 234 + 390,
então a seleção usa todas e o `sorted`/semente não altera nada) nem o cálculo (AUC via
scikit-learn ≡ Mann-Whitney). A versão original usava `cv2.resize` sem especificar
interpolação (default `INTER_LINEAR`); o pipeline atual usa `INTER_AREA` na redução,
recomendada para diminuir imagens (menos *aliasing*). Ao forçar `INTER_LINEAR`, o AUC volta
a **0,9469 exato** — confirmando que a interpolação é a única fonte da diferença. As 248
imagens cujo score muda são as **maiores** (maior lado médio 1570 px vs 1269 px nas
inalteradas), onde as duas interpolações mais divergem.

**Para reproduzir o AUC 0,9469 na Chest X-Ray:** em `src/pneumoshift/preprocess.py`, na
função `esticar`, force `INTER_LINEAR`:

```python
# def esticar(img, size=IMG_SIZE):
#     ...
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)   # reproduz a versao original
```

Depois rode `avaliar_lote.py` (cxray, esticar). A versao corrigida mantem `INTER_AREA` por
ser tecnicamente superior na reducao.

> A versao corrigida adota `GEOMETRIA = "padding"` (preserva a proporcao da imagem, conforme
> a devolutiva); a geometria `esticar` fica disponivel para rastreabilidade e para a analise
> de sensibilidade da geometria.
