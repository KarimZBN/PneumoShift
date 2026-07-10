"""
Converte imagens do RSNA (DICOM -> PNG) a partir dos SPLITS versionados.

Le dados/splits/validacao.csv e pool_teste.csv (gerados por split_rsna.py) e converte
exatamente esses IDs — nao sorteia. Assim os PNGs em disco correspondem a particao
registrada, sem sobreposicao entre o conjunto de validacao (limiar) e o de teste.

PRESERVA O TAMANHO ORIGINAL do DICOM (apenas normaliza 0-255). A geometria (padding ou
esticar) e aplicada depois, no pre-processamento da inferencia — igual a Chest X-Ray.
Assim nao ha "geometria assada" no disco: um unico conjunto de PNGs serve as duas.

Destino:
    dados/test/rsna_validacao/{NORMAL,PNEUMONIA}/*.png
    dados/test/rsna_pool/{NORMAL,PNEUMONIA}/*.png

Uso:
    python scripts/converter_rsna.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv
import time

import cv2
import numpy as np
import pydicom

from pneumoshift import paths

IMAGES_DIR = paths.DADOS / "raw" / "rsna" / "stage_2_train_images"
SPLITS_DIR = paths.DADOS / "splits"
DESTINOS = {
    "validacao.csv": paths.DADOS_TESTE / "rsna_validacao",
    "pool_teste.csv": paths.DADOS_TESTE / "rsna_pool",
}


def ler_split_com_classe(caminho):
    """Le um split (patientId, classe) e devolve lista de (patientId, classe)."""
    itens = []
    with open(caminho, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            pid = (row.get("patientId") or "").strip()
            classe = (row.get("classe") or "").strip().upper()
            if pid and classe in ("PNEUMONIA", "NORMAL"):
                itens.append((pid, classe))
    return itens


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


def converter_split(split_csv, out_dir):
    caminho = SPLITS_DIR / split_csv
    if not caminho.is_file():
        raise FileNotFoundError(
            f"Split nao encontrado: {caminho}. Rode antes: python scripts/split_rsna.py")
    itens = ler_split_com_classe(caminho)
    for classe in ("PNEUMONIA", "NORMAL"):
        (out_dir / classe).mkdir(parents=True, exist_ok=True)

    print(f"\n{split_csv} -> {out_dir.name}: {len(itens)} imagens")
    t0 = time.time()
    feitas = faltando = 0
    for pid, classe in itens:
        if dicom_para_png(pid, out_dir / classe):
            feitas += 1
        else:
            faltando += 1
        if (feitas + faltando) % 200 == 0:
            print(f"  [{feitas+faltando:5}/{len(itens)}] {classe}")
    print(f"  Convertidas: {feitas} | .dcm nao encontrados: {faltando} | {time.time()-t0:.0f}s")


def main():
    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Pasta de imagens nao encontrada: {IMAGES_DIR}")
    print("=" * 55)
    print("Convertendo no TAMANHO ORIGINAL (geometria aplicada no pre-processamento).")
    for split_csv, out_dir in DESTINOS.items():
        converter_split(split_csv, out_dir)
    print(f"\n{'=' * 55}\nCONCLUIDO. Destinos: {', '.join(d.name for d in DESTINOS.values())}")


if __name__ == "__main__":
    main()
