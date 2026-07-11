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
- **RSNA Pneumonia Detection Challenge:** imagens em DICOM de pacientes
  adultos, com os rótulos em `stage_2_train_labels.csv`.

Estrutura esperada após o download:

```
dados/
  raw/rsna/stage_2_train_images/*.dcm
  raw/rsna/stage_2_train_labels.csv
  test/cxray/{NORMAL,PNEUMONIA}/*.jpeg
  test/rsna_pool/{NORMAL,PNEUMONIA}/*.png        (gerado pela conversão — teste)
  test/rsna_validacao/{NORMAL,PNEUMONIA}/*.png   (gerado pela conversão — limiar)
  splits/{validacao,pool_teste}.csv              (gerado por split_rsna.py)
```

Observação sobre os rótulos do RSNA: `Target = 0` indica **ausência de anotação
de opacidade compatível com pneumonia**, e não necessariamente um exame normal.
Portanto, a classe negativa do RSNA não é equivalente à classe normal do Kaggle.

## Passo a passo

Os scripts ficam em `scripts/` e importam o pacote em `src/pneumoshift/`. Rode-os a
partir da raiz do projeto (os caminhos são resolvidos automaticamente).

1. **Separar o RSNA em validação e teste (sem vazamento):**

   ```
   python scripts/split_rsna.py
   ```

   Particiona os pacientes do RSNA de forma reprodutível (`SEED_SPLIT = 20260710`) em um
   conjunto de **validação** (define o limiar) e um **pool de teste** (as rodadas amostram
   dele), sem sobreposição entre eles. Grava as listas em `dados/splits/`.

2. **Converter o RSNA (DICOM → PNG):**

   ```
   python scripts/converter_rsna.py
   ```

   Lê os splits e converte exatamente os pacientes selecionados, gravando os PNGs **no
   tamanho original** em `dados/test/rsna_validacao/` e `dados/test/rsna_pool/`. A geometria
   (padding/esticar) não é fixada no disco — é aplicada no pré-processamento da inferência.

3. **Avaliar uma execução (base + geometria):**

   ```
   python scripts/avaliar_lote.py
   ```

   Ajuste `BASE` (`"cxray"`/`"rsna"`) e `GEOMETRIA` (`"padding"`/`"esticar"`) no início do
   arquivo. Cada execução grava sua própria pasta com timestamp em
   `resultados/<base>/<geometria>_<AAAAMMDD-HHMMSS>_seed<N>/` (nunca sobrescreve):
   `predicoes.csv` (resumo no topo + predição por imagem, score cru), `metricas.json` e, ao
   final, a estatística da execução (`metricas_estat.json`, `roc.png`, `calibracao.png`).
   Repita para as 4 combinações (cxray/rsna × padding/esticar) para o conjunto completo.

4. **Limiar em validação separada + múltiplas rodadas (RSNA):**

   ```
   python scripts/avaliar_rsna_rodadas.py
   ```

   Define o limiar de Youden no conjunto de validação e o aplica ao pool de teste, repetindo
   a amostragem pareada com várias sementes (`42..51`). Grava média ± desvio em
   `resultados/rsna/rodadas_<timestamp>/`.

5. **Análise estatística comparativa (AUC/calibração/IC + figuras):**

   ```
   python scripts/analise_estatistica.py
   ```

   Lê a execução **mais recente** de cada `resultados/<base>/<geometria>_*/` e grava as
   comparativas numa pasta única `resultados/_comparacoes/run_<timestamp>/` (domínio:
   cxray×rsna; geometria: padding×esticar) mais o `resumo_geral.csv`.

6. **Explicabilidade e verificações:**

   ```
   python scripts/gradcam_lote.py        # Grad-CAM em lote por categoria (VP/VN/FP/FN)
   python scripts/analise_foco.py        # foco do Grad-CAM (periferia x miolo)
   python scripts/demo_imagem.py         # classifica 1 imagem + Grad-CAM
   python scripts/inspecionar_dicom.py   # metadados DICOM
   python scripts/comparar_dicom_png.py  # orientacao DICOM x PNG + aspect ratio
   python tests/test_gradcam.py          # valida o Grad-CAM
   python tests/test_split.py            # valida que validacao e teste sao disjuntos
   ```

   O `gradcam_lote.py` e o `analise_foco.py` gravam em
   `resultados/gradcam/<base>_<geometria>_<timestamp>/`.

## Reprodutibilidade

- A partição e a seleção da amostra usam sementes fixas (`SEED_SPLIT = 20260710` na
  separação validação/teste e `SEED = 42` na avaliação), com ordenação prévia dos arquivos
  antes do embaralhamento. O `converter_rsna.py` apenas lê os splits, não sorteia.
- O pareamento (mesma quantidade e proporção nas duas bases) isola a variável
  "origem dos dados".
- O modelo apenas realiza inferência, sem qualquer reajuste de pesos, o que
  garante saídas idênticas entre execuções.

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

As análises estatísticas incorporadas na reestruturação (AUC, calibração, intervalos de
confiança por bootstrap e curvas) utilizam o **scikit-learn** (versão 1.3.2), e as figuras
(ROC, calibração) são geradas com o **matplotlib** (versão 3.7.5). As versões completas e
fixadas de todas as dependências estão em [`requirements.txt`](requirements.txt).

## Nota aos orientadores — reproduzindo e relacionando os números da devolutiva

Esta seção mostra como chegar aos números da devolutiva a partir do código atual. A
**reprodução exata é garantida na Chest X-Ray** (base interna, conjunto de teste fixo); na
**RSNA** (base externa), a devolutiva pediu justamente ampliar a amostra / repetir a seleção,
o que foi feito — por isso o valor de referência da RSNA mudou, e a nota documenta essa relação
de forma transparente.

| Métrica | Reportado na devolutiva | Pipeline atual (`esticar`) |
|---|:---:|:---:|
| AUC — Chest X-Ray (esticar) | 0,9469 (original) | 0,9358 → **0,9469** ao forçar `INTER_LINEAR` |
| Brier — Chest X-Ray (esticar) | ~0,094 | 0,1310 |
| AUC — RSNA (esticar) | 0,7320 (original) | 0,6798 (amostra ampliada) |
| Brier — RSNA (esticar) | ~0,281 | 0,3257 |

**RSNA (0,7320 → 0,6798): ampliação da amostra, atendendo à devolutiva.** O valor original
vinha de uma seleção única de 624 imagens da conversão antiga. Atendendo ao pedido de avaliar
mais dados / repetir a seleção, as imagens passaram a ser convertidas para um **pool de teste**
maior, do qual a amostra pareada de 624 é sorteada. As **10 rodadas com sementes distintas**
(`avaliar_rsna_rodadas.py`) dão AUC média de **0,69 ± 0,02** — ou seja, ~0,68 é o valor típico
da base externa, e o 0,7320 correspondia a uma seleção pequena favorável. O resultado atual é
mais robusto por repousar sobre mais dados e sobre a média de várias amostragens. Como a
conversão da versão da banca foi substituída por este pipeline, o 0,7320 não é reproduzível
byte a byte pelo código atual; o número de referência da base externa passa a ser o das rodadas.

Na **Chest X-Ray**, a versão atual (com `INTER_AREA`) dá AUC 0,9358, e o trabalho original
reportava 0,9469. A causa foi investigada e **confirmada por reprodução direta**: é o método
de **interpolação** do redimensionamento — **não** a amostra (as 624 imagens são as mesmas nas
duas versões, pois o conjunto de teste tem exatamente 234 + 390, então a seleção usa todas e o
`sorted`/semente não altera nada) nem o cálculo (AUC via scikit-learn ≡ Mann-Whitney). A versão
original usava `cv2.resize` sem especificar interpolação (default `INTER_LINEAR`); o pipeline
atual usa `INTER_AREA` na redução, recomendada para diminuir imagens (menos *aliasing*). Ao
forçar `INTER_LINEAR`, o AUC volta a **0,9469 exato** — confirmando que a interpolação é a
única fonte da diferença. As 248 imagens cujo score muda são as **maiores** (maior lado médio
1570 px vs 1269 px nas inalteradas), onde as duas interpolações mais divergem.

**Para reproduzir o valor original na Chest X-Ray (AUC 0,9469):** em
`src/pneumoshift/preprocess.py`, na função `esticar`, force `INTER_LINEAR`:

```python
# def esticar(img, size=IMG_SIZE):
#     ...
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)   # reproduz a versao original
```

Depois rode `avaliar_lote.py` (kaggle, esticar) + `analise_estatistica.py`. A versão corrigida
mantém `INTER_AREA` por ser tecnicamente superior na redução.

> A versão corrigida do trabalho adota `GEOMETRIA = "padding"` (preserva a proporção da
> imagem, conforme a devolutiva); a geometria `esticar` fica disponível para rastreabilidade
> e para a análise de sensibilidade da geometria.

Uma nota final: o passo `kaggle` acima refere-se à base **Chest X-Ray** (`BASE = "cxray"`).
