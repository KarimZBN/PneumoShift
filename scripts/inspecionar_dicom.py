"""
Inspeciona os metadados DICOM da RSNA relevantes para a conversao em PNG.

Para uma amostra dos arquivos .dcm reporta, por imagem, os campos que determinam
se a conversao min-max 0-255 e sem perdas:

    - Photometric Interpretation  (MONOCHROME1 => invertido; MONOCHROME2 => nao)
    - BitsStored                  (8 => pixel ja em 0-255; 12/16 => faixa maior)
    - Rescale Slope / Intercept   (transformacao linear de unidade a respeitar)
    - VOI LUT Sequence            (look-up table de visualizacao)
    - Window Center / Width        (windowing a respeitar)

Amostragem reprodutivel (SEED fixa): os N_SEQUENCIAIS primeiros arquivos (em ordem)
mais N_ALEATORIOS sorteados do restante, cobrindo vizinhanca e dispersao.

Uso:
    python src/inspecionar_dicom.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv
import random

import pydicom

from pneumoshift import paths

# --- Configuracao ---
SEED = 13
N_SEQUENCIAIS = 300        # primeiros arquivos em ordem (vizinhanca)
N_ALEATORIOS = 1500        # sorteados do restante (dispersao)

# --- Caminhos ---
IMAGES_DIR = paths.DADOS / "raw" / "rsna" / "stage_2_train_images"
RESULTS_DIR = paths.RESULTADOS / "APOIO_provas-dicom"
OUT_CSV = RESULTS_DIR / "inspecao_dicom.csv"


def selecionar_arquivos():
    """Lista os .dcm ordenados e devolve N_SEQUENCIAIS iniciais + N_ALEATORIOS do resto."""
    todos = sorted(p.name for p in IMAGES_DIR.iterdir() if p.suffix.lower() == ".dcm")
    if not todos:
        raise FileNotFoundError(f"Nenhum .dcm encontrado em {IMAGES_DIR}")

    sequenciais = todos[:N_SEQUENCIAIS]
    resto = todos[N_SEQUENCIAIS:]
    random.seed(SEED)
    random.shuffle(resto)
    aleatorios = resto[:N_ALEATORIOS]

    # Marca a origem (sequencial/aleatorio) de cada arquivo no CSV.
    escolhidos = [(nome, "sequencial") for nome in sequenciais]
    escolhidos += [(nome, "aleatorio") for nome in aleatorios]
    return escolhidos


def inspecionar(nome):
    """Le o cabecalho DICOM (sem carregar os pixels) e extrai os campos de interesse."""
    ds = pydicom.dcmread(str(IMAGES_DIR / nome), stop_before_pixels=True)
    return {
        "arquivo": nome,
        "photometric": getattr(ds, "PhotometricInterpretation", "AUSENTE"),
        "bits_stored": getattr(ds, "BitsStored", "AUSENTE"),
        "bits_allocated": getattr(ds, "BitsAllocated", "AUSENTE"),
        "tem_rescale_slope": "RescaleSlope" in ds,
        "tem_rescale_intercept": "RescaleIntercept" in ds,
        "tem_voi_lut": "VOILUTSequence" in ds,
        "tem_window_center": "WindowCenter" in ds,
        "tem_window_width": "WindowWidth" in ds,
    }


def main():
    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Pasta de imagens nao encontrada: {IMAGES_DIR}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    escolhidos = selecionar_arquivos()
    print(f"Inspecionando {len(escolhidos)} DICOMs "
          f"({N_SEQUENCIAIS} sequenciais + ate {N_ALEATORIOS} aleatorios)...\n")

    registros = []
    contagem = {
        "photometric": {},
        "bits_stored": {},
        "rescale_slope": 0,
        "rescale_intercept": 0,
        "voi_lut": 0,
        "window_center": 0,
    }

    for nome, origem in escolhidos:
        try:
            info = inspecionar(nome)
        except Exception as exc:            # arquivo corrompido/ausente: registra e segue
            print(f"  Erro em {nome}: {exc}")
            continue
        info["origem_amostra"] = origem
        registros.append(info)

        contagem["photometric"][info["photometric"]] = \
            contagem["photometric"].get(info["photometric"], 0) + 1
        contagem["bits_stored"][info["bits_stored"]] = \
            contagem["bits_stored"].get(info["bits_stored"], 0) + 1
        contagem["rescale_slope"] += info["tem_rescale_slope"]
        contagem["rescale_intercept"] += info["tem_rescale_intercept"]
        contagem["voi_lut"] += info["tem_voi_lut"]
        contagem["window_center"] += info["tem_window_center"]

    # --- CSV por imagem ---
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        campos = ["arquivo", "origem_amostra", "photometric", "bits_stored",
                  "bits_allocated", "tem_rescale_slope", "tem_rescale_intercept",
                  "tem_voi_lut", "tem_window_center", "tem_window_width"]
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for r in registros:
            writer.writerow({k: r[k] for k in campos})

    # --- Resumo agregado (a evidencia) ---
    n = len(registros)
    print("=" * 55)
    print(f"RESUMO ({n} imagens inspecionadas)")
    print("=" * 55)
    print("\nPhotometric Interpretation:")
    for k, v in sorted(contagem["photometric"].items()):
        print(f"  {k:15s} {v:5d}  ({v/n*100:5.1f}%)")
    print("\nBits Stored:")
    for k, v in sorted(contagem["bits_stored"].items(), key=lambda x: str(x[0])):
        print(f"  {str(k):15s} {v:5d}  ({v/n*100:5.1f}%)")
    print("\nPresenca de transformacoes (nenhuma esperada):")
    for rotulo, chave in (("Rescale Slope", "rescale_slope"),
                          ("Rescale Intercept", "rescale_intercept"),
                          ("VOI LUT Sequence", "voi_lut"),
                          ("Window Center", "window_center")):
        v = contagem[chave]
        print(f"  {rotulo:20s} {v:5d}  ({v/n*100:5.1f}%)")

    print(f"\nDetalhamento por imagem salvo em: {OUT_CSV}")
    print("\nInterpretacao: se Photometric = 100% MONOCHROME2, BitsStored = 8 e as")
    print("quatro transformacoes = 0%, a conversao min-max 0-255 e sem perdas.")


if __name__ == "__main__":
    main()
