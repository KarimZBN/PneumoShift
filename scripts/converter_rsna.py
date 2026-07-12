"""
Converte a base RSNA INTEIRA (DICOM -> PNG), agregada por paciente.

Le stage_2_train_labels.csv, agrega por patientId (1 paciente = 1 imagem .dcm; as
multiplas linhas do CSV sao bounding boxes na mesma imagem) e converte TODOS os
pacientes elegiveis para PNG, organizados por classe:

    Target=1 -> PNEUMONIA ; Target=0 -> NORMAL (ausencia de anotacao de opacidade,
    nao necessariamente exame normal). Pacientes presentes nas duas classes sao
    descartados (ambiguidade).

Esta e a pasta-fonte unica. As avaliacoes que precisam de subconjuntos (balanceado,
validacao) copiam PNGs daqui via split_rsna.py, sem reconverter. A avaliacao da base
inteira le esta pasta diretamente.

PRESERVA O TAMANHO ORIGINAL do DICOM (apenas normaliza 0-255). A geometria (padding
ou esticar) e aplicada no pre-processamento da inferencia.

Destino:
    dados/processed/rsna/{NORMAL,PNEUMONIA}/*.png

Uso:
    python scripts/converter_rsna.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import time

import cv2
import numpy as np
import pydicom

from pneumoshift import paths
from pneumoshift.data import ler_rotulos_rsna

IMAGES_DIR = paths.DADOS / "raw" / "rsna" / "stage_2_train_images"
LABELS_CSV = paths.DADOS / "raw" / "rsna" / "stage_2_train_labels.csv"
OUT_DIR = paths.RSNA_FULL


def dicom_para_png(patient_id, destino):
    """Converte um DICOM em PNG normalizado 0-255, NO TAMANHO ORIGINAL (sem resize)."""
    dcm_path = IMAGES_DIR / f"{patient_id}.dcm"
    if not dcm_path.is_file():
        return False
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr = arr / arr.max() * 255.0
    arr = arr.astype(np.uint8)          # tamanho original preservado
    cv2.imwrite(str(destino / f"{patient_id}.png"), arr)
    return True


def converter_classe(ids, classe, out_dir):
    destino = out_dir / classe
    destino.mkdir(parents=True, exist_ok=True)
    ids = sorted(ids)                   # ordem estavel entre execucoes/SO
    print(f"\n{classe}: {len(ids)} pacientes")
    t0 = time.time()
    feitas = faltando = 0
    for i, pid in enumerate(ids, 1):
        if dicom_para_png(pid, destino):
            feitas += 1
        else:
            faltando += 1
        if i % 500 == 0:
            print(f"  [{i:6}/{len(ids)}] {feitas} ok, {faltando} sem .dcm ({time.time()-t0:.0f}s)")
    print(f"  Convertidas: {feitas} | .dcm nao encontrados: {faltando} | {time.time()-t0:.0f}s")
    return feitas, faltando


def main():
    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Pasta de imagens nao encontrada: {IMAGES_DIR}")
    if not LABELS_CSV.is_file():
        raise FileNotFoundError(f"CSV de rotulos nao encontrado: {LABELS_CSV}")

    pneu, norm, ambiguos = ler_rotulos_rsna(LABELS_CSV)
    print("=" * 60)
    print("Conversao da base RSNA INTEIRA (agregada por paciente).")
    print(f"Elegiveis -> PNEUMONIA: {len(pneu)} | NORMAL: {len(norm)}")
    print(f"Descartados (ambiguos, nas duas classes): {len(ambiguos)}")
    print(f"Total elegivel: {len(pneu) + len(norm)} pacientes")
    print("Convertendo no TAMANHO ORIGINAL (geometria aplicada no pre-processamento).")

    fp, xp = converter_classe(pneu, "PNEUMONIA", OUT_DIR)
    fn, xn = converter_classe(norm, "NORMAL", OUT_DIR)

    print(f"\n{'=' * 60}")
    print(f"CONCLUIDO -> {OUT_DIR}")
    print(f"  PNEUMONIA: {fp} convertidas ({xp} sem .dcm)")
    print(f"  NORMAL:    {fn} convertidas ({xn} sem .dcm)")
    print(f"  TOTAL:     {fp + fn} PNGs")


if __name__ == "__main__":
    main()
