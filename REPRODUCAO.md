# Guia de reprodução dos testes

Passo a passo para reproduzir a avaliação descrita no [README](README.md).

## Requisitos

```
pip install -r requirements.txt
```

Ambiente de referência: Python 3.10, TensorFlow/Keras 2.12, OpenCV, NumPy
1.23.5 e pydicom (para a conversão do RSNA).

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
  test/kaggle/{NORMAL,PNEUMONIA}/*.jpeg
  test/rsna/{NORMAL,PNEUMONIA}/*.png   (gerado pela conversão)
```

Observação sobre os rótulos do RSNA: `Target = 0` indica **ausência de anotação
de opacidade compatível com pneumonia**, e não necessariamente um exame normal.
Portanto, a classe negativa do RSNA não é equivalente à classe normal do Kaggle.

## Passo a passo

Os scripts ficam em `scripts/` e importam o pacote em `src/pneumoshift/`. Rode-os a
partir da raiz do projeto (os caminhos são resolvidos automaticamente).

1. **Converter o RSNA (DICOM → PNG):**

   ```
   python scripts/converter_rsna.py
   ```

   Sorteia, de forma reprodutível (`SEED = 13`), imagens de cada classe e as
   grava em `dados/test/rsna/`. Para a variante com padding letterbox aplicado na
   conversão, use `python scripts/converter_rsna_padding.py` (grava em `dados/test/rsna_padding/`).

2. **Avaliar uma base em lote:**

   ```
   python scripts/avaliar_lote.py
   ```

   Ajuste `FONTE` para `"kaggle"` ou `"rsna"` no início do arquivo. A avaliação
   usa uma amostra pareada de 234 normais + 390 pneumonia (`SEED = 42`), gravando
   um CSV em `resultados/csv/` com a predição por imagem e o resumo das métricas.

3. **Classificar uma única imagem (demonstração + Grad-CAM):**

   ```
   python scripts/demo_imagem.py
   ```

4. **Explicabilidade e verificações (opcional):**

   ```
   python scripts/gradcam_lote.py        # Grad-CAM em lote por categoria
   python scripts/analise_foco.py        # foco do Grad-CAM (periferia x miolo)
   python scripts/inspecionar_dicom.py   # metadados DICOM
   python scripts/comparar_dicom_png.py  # orientacao DICOM x PNG + aspect ratio
   python tests/test_gradcam.py          # valida o Grad-CAM
   ```

## Reprodutibilidade

- A conversão e a seleção da amostra usam sementes fixas (`SEED = 13` na
  conversão e `SEED = 42` na avaliação), com ordenação prévia dos arquivos antes
  do embaralhamento.
- O pareamento (mesma quantidade e proporção nas duas bases) isola a variável
  "origem dos dados".
- O modelo apenas realiza inferência, sem qualquer reajuste de pesos, o que
  garante saídas idênticas entre execuções.
