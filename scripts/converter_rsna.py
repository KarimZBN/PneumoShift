"""
Converte imagens do RSNA (DICOM) para PNG na pasta de teste, por classe.

Uso exclusivo para TESTE (sem split treino/validacao): o modelo avaliado e externo.
Gera N_PER_CLASS de cada classe com folga, para o avaliador selecionar de forma
reprodutivel o conjunto pareado com o Kaggle (234 normais + 390 pneumonia).

Uso:
    python scripts/converter_rsna.py
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

# --- Configuracao ---
N_PER_CLASS = 500          # folga; o teste usa apenas o n desejado
SEED = 13
IMG_SIZE = (224, 224)

IMAGES_DIR = paths.DADOS / "raw" / "rsna" / "stage_2_train_images"
LABELS_CSV = paths.DADOS / "raw" / "rsna" / "stage_2_train_labels.csv"
OUTPUT_DIR = paths.DADOS_TESTE / "rsna"   # destino: {NORMAL,PNEUMONIA}


def ler_rotulos():
    """Le o CSV de rotulos e devolve os conjuntos de IDs por classe (um por paciente)."""
    pneumonia_ids, normal_ids = set(), set()
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        print(f"  Colunas: {reader.fieldnames}")
        for row in reader:
            target = row["Target"].strip()
            if target == "1":
                pneumonia_ids.add(row["patientId"])
            elif target == "0":
                normal_ids.add(row["patientId"])

    ambiguos = pneumonia_ids & normal_ids
    pneumonia_ids -= ambiguos
    normal_ids -= ambiguos
    print(f"  Pneumonia: {len(pneumonia_ids)} | Normal: {len(normal_ids)} | Ambiguos removidos: {len(ambiguos)}")
    return pneumonia_ids, normal_ids


def sortear(ids, n):
    """Ordena e embaralha o conjunto (semente definida uma vez em main), devolve n IDs."""
    selecionados = sorted(ids)
    random.shuffle(selecionados)
    return selecionados[:n]


def dicom_para_png(patient_id, destino):
    """Converte um DICOM em PNG normalizado 0-255 e redimensionado (resize direto)."""
    dcm_path = IMAGES_DIR / f"{patient_id}.dcm"
    if not dcm_path.is_file():
        return False

    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr = arr / arr.max() * 255.0
    arr = arr.astype(np.uint8)
    arr = cv2.resize(arr, IMG_SIZE)
    cv2.imwrite(str(destino / f"{patient_id}.png"), arr)
    return True


def main():
    print("=" * 55)
    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Pasta de imagens nao encontrada: {IMAGES_DIR}")
    if not LABELS_CSV.is_file():
        raise FileNotFoundError(f"CSV de rotulos nao encontrado: {LABELS_CSV}")
    print("Caminhos OK.\nLENDO CSV DE ROTULOS...")
    pneumonia_ids, normal_ids = ler_rotulos()

    n_por_classe = min(N_PER_CLASS, len(pneumonia_ids), len(normal_ids))
    if n_por_classe < N_PER_CLASS:
        print(f"  ATENCAO: so ha {n_por_classe} por classe. Usando {n_por_classe}.")

    random.seed(SEED)
    sel_pneu = sortear(pneumonia_ids, n_por_classe)
    sel_norm = sortear(normal_ids, n_por_classe)
    print(f"  Selecionados: {len(sel_pneu)} pneumonia + {len(sel_norm)} normal")

    for classe in ("PNEUMONIA", "NORMAL"):
        (OUTPUT_DIR / classe).mkdir(parents=True, exist_ok=True)

    jobs = [("PNEUMONIA", sel_pneu), ("NORMAL", sel_norm)]
    total_geral = sum(len(ids) for _, ids in jobs)
    print(f"\nCONVERTENDO {total_geral} IMAGENS...\n")

    t0 = time.time()
    feitas = faltando = 0
    for classe, ids in jobs:
        destino = OUTPUT_DIR / classe
        for pid in ids:
            if dicom_para_png(pid, destino):
                feitas += 1
            else:
                faltando += 1
            processadas = feitas + faltando
            if processadas % 100 == 0:
                dec = time.time() - t0
                vel = processadas / dec if dec > 0 else 0
                print(f"  [{processadas:4}/{total_geral}] {processadas/total_geral*100:5.1f}% | {vel:.1f} img/s | {classe}")

    print(f"\n{'=' * 55}")
    print(f"CONCLUIDO em {time.time()-t0:.0f}s")
    print(f"  Convertidas: {feitas} | .dcm nao encontrados: {faltando}")
    print(f"  Destino: {OUTPUT_DIR}/(PNEUMONIA|NORMAL)")


if __name__ == "__main__":
    main()
