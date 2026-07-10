"""
Variante de converter_rsna.py que aplica PADDING LETTERBOX na conversao DICOM -> PNG.

No pipeline original o padding aplicado na avaliacao nao afeta a RSNA, porque
converter_rsna.py ja grava os PNGs em 224x224 esticados. Para testar padding real na
RSNA ele precisa ser aplicado aqui, sobre a imagem DICOM original. Mesma SEED (13) e
mesmos pacientes; destino separado (dados/test/rsna_padding) para nao sobrescrever.

Para avaliar depois: em avaliar_lote.py use FONTE = "rsna_padding".

Uso:
    python scripts/converter_rsna_padding.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv
import random
import time

import cv2
import numpy as np
import pydicom

from pneumoshift import paths
from pneumoshift.preprocess import redimensionar

# --- Configuracao ---
N_PER_CLASS = 500
SEED = 13

IMAGES_DIR = paths.DADOS / "raw" / "rsna" / "stage_2_train_images"
LABELS_CSV = paths.DADOS / "raw" / "rsna" / "stage_2_train_labels.csv"
OUTPUT_DIR = paths.DADOS_TESTE / "rsna_padding"   # destino SEPARADO


def ler_rotulos():
    pneumonia_ids, normal_ids = set(), set()
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row["Target"].strip()
            if t == "1":
                pneumonia_ids.add(row["patientId"])
            elif t == "0":
                normal_ids.add(row["patientId"])
    ambiguos = pneumonia_ids & normal_ids
    pneumonia_ids -= ambiguos
    normal_ids -= ambiguos
    print(f"  Pneumonia: {len(pneumonia_ids)} | Normal: {len(normal_ids)} | Ambiguos: {len(ambiguos)}")
    return pneumonia_ids, normal_ids


def sortear(ids, n):
    sel = sorted(ids)
    random.shuffle(sel)
    return sel[:n]


def dicom_para_png(patient_id, destino):
    """Converte um DICOM em PNG 0-255 com PADDING letterbox ate 224x224."""
    dcm_path = IMAGES_DIR / f"{patient_id}.dcm"
    if not dcm_path.is_file():
        return False
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr = arr / arr.max() * 255.0
    arr = arr.astype(np.uint8)
    arr = redimensionar(arr)   # padding letterbox (funciona em grayscale 2D)
    cv2.imwrite(str(destino / f"{patient_id}.png"), arr)
    return True


def main():
    print("=" * 55)
    print("CONVERSOR RSNA - VARIANTE PADDING LETTERBOX")
    print("=" * 55)
    if not IMAGES_DIR.is_dir() or not LABELS_CSV.is_file():
        raise FileNotFoundError("Imagens ou CSV de rotulos nao encontrados.")

    pneumonia_ids, normal_ids = ler_rotulos()
    n_por_classe = min(N_PER_CLASS, len(pneumonia_ids), len(normal_ids))

    random.seed(SEED)
    sel_pneu = sortear(pneumonia_ids, n_por_classe)
    sel_norm = sortear(normal_ids, n_por_classe)
    print(f"  Selecionados: {len(sel_pneu)} pneumonia + {len(sel_norm)} normal")

    for classe in ("PNEUMONIA", "NORMAL"):
        (OUTPUT_DIR / classe).mkdir(parents=True, exist_ok=True)

    jobs = [("PNEUMONIA", sel_pneu), ("NORMAL", sel_norm)]
    total_geral = sum(len(ids) for _, ids in jobs)
    print(f"\nCONVERTENDO {total_geral} IMAGENS (com padding)...\n")

    t0 = time.time()
    feitas = faltando = 0
    for classe, ids in jobs:
        destino = OUTPUT_DIR / classe
        for pid in ids:
            if dicom_para_png(pid, destino):
                feitas += 1
            else:
                faltando += 1
            if (feitas + faltando) % 100 == 0:
                print(f"  [{feitas+faltando:4}/{total_geral}] {classe}")

    print(f"\nCONCLUIDO em {time.time()-t0:.0f}s")
    print(f"  Convertidas: {feitas} | .dcm nao encontrados: {faltando}")
    print(f"  Destino: {OUTPUT_DIR}")
    print(f"  Para avaliar: em avaliar_lote.py use FONTE = 'rsna_padding'.")


if __name__ == "__main__":
    main()
