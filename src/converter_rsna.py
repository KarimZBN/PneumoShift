"""
Converte imagens do RSNA (DICOM) para PNG na pasta de teste, por classe.

Uso exclusivo para TESTE (sem split treino/validacao): o modelo avaliado e
externo. Gera N_PER_CLASS de cada classe com folga, garantindo material
suficiente para o batch selecionar, de forma reprodutivel, o conjunto pareado
com o Kaggle (234 normais + 390 pneumonia).
"""
from pathlib import Path
import csv
import random
import time

import cv2
import numpy as np
import pydicom

# --- Configuracao ---
N_PER_CLASS = 500          # folga; o teste usa apenas o n desejado
SEED = 13
IMG_SIZE = (224, 224)

# --- Caminhos (raiz = pasta acima de src/) ---
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE_DIR / "dados" / "raw" / "rsna" / "stage_2_train_images"
LABELS_CSV = BASE_DIR / "dados" / "raw" / "rsna" / "stage_2_train_labels.csv"
OUTPUT_DIR = BASE_DIR / "dados" / "test" / "rsna"   # destino: {NORMAL,PNEUMONIA}


def ler_rotulos():
    """Le o CSV de rotulos e devolve os conjuntos de IDs por classe (uma entrada por paciente)."""
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

    # Descarta IDs presentes nas duas classes para evitar rotulo duplo.
    ambiguos = pneumonia_ids & normal_ids
    pneumonia_ids -= ambiguos
    normal_ids -= ambiguos
    print(f"  Pneumonia: {len(pneumonia_ids)} | Normal: {len(normal_ids)} | Ambiguos removidos: {len(ambiguos)}")
    return pneumonia_ids, normal_ids


def sortear(ids, n):
    """Ordena e embaralha o conjunto, devolvendo n IDs.

    A semente e definida uma unica vez em main(), antes das duas chamadas,
    de modo que a sequencia de sorteios seja reprodutivel entre execucoes.
    """
    selecionados = sorted(ids)
    random.shuffle(selecionados)
    return selecionados[:n]


def dicom_para_png(patient_id, destino):
    """Converte um DICOM em PNG normalizado 0-255 e redimensionado."""
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
    print("VERIFICANDO CAMINHOS...")
    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Pasta de imagens nao encontrada: {IMAGES_DIR}")
    if not LABELS_CSV.is_file():
        raise FileNotFoundError(f"CSV de rotulos nao encontrado: {LABELS_CSV}")
    print("Caminhos OK.")

    print("\nLENDO CSV DE ROTULOS...")
    pneumonia_ids, normal_ids = ler_rotulos()

    n_por_classe = min(N_PER_CLASS, len(pneumonia_ids), len(normal_ids))
    if n_por_classe < N_PER_CLASS:
        print(f"  ATENCAO: so ha {n_por_classe} por classe (pedido: {N_PER_CLASS}). Usando {n_por_classe}.")

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
                decorrido = time.time() - t0
                vel = processadas / decorrido if decorrido > 0 else 0
                resta = (total_geral - processadas) / vel if vel > 0 else 0
                print(f"  [{processadas:4}/{total_geral}] {processadas/total_geral*100:5.1f}% | "
                      f"{vel:.1f} img/s | resta ~{resta:4.0f}s | {classe}")

    decorrido = time.time() - t0
    print(f"\n{'=' * 55}")
    print(f"CONCLUIDO em {decorrido:.0f}s ({decorrido/60:.1f} min)")
    print(f"  Convertidas: {feitas} | .dcm nao encontrados: {faltando}")
    print(f"  Destino: {OUTPUT_DIR}/(PNEUMONIA|NORMAL)")


if __name__ == "__main__":
    main()
