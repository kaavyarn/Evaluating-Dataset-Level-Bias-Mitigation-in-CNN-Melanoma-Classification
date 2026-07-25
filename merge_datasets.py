"""
Dataset Merge Script — Fitzpatrick17k + ISIC Archive
======================================================
Merges both source datasets into the single CSV format expected by
MelanomaDataset in melanoma_pipeline.py:

    image_path, label, fitzpatrick

Usage:
    python merge_datasets.py


Update the CONFIG paths below to match your local setup.
"""

import os
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ─────────────────────────────────────────────
# CONFIG — update these paths
# ─────────────────────────────────────────────

CONFIG = {
    "fitzpatrick_csv":      "raw_data/fitzpatrick17k.csv",
    "fitzpatrick_image_dir": "raw_data/fitzpatrick_images",   # downloaded images go here

    "isic_csv":              "raw_data/isic_metadata.csv",
    "isic_image_dir":        "raw_data/isic_images",          # downloaded images go here

    "output_dir":            "data",
    "download_fitzpatrick_images": True,   # set False if you already have local images
    "max_download_workers":  16,
    "random_seed":           42,
    "train_frac":            0.70,
    "val_frac":              0.15,
    "test_frac":              0.15,
}

np.random.seed(CONFIG["random_seed"])


# ─────────────────────────────────────────────
# 1. LOAD + CLEAN FITZPATRICK17K
# ─────────────────────────────────────────────
# Real fitzpatrick17k.csv columns (per mattgroh/fitzpatrick17k):
#   md5hash, fitzpatrick_scale, fitzpatrick_centaur, label,
#   nine_partition_label, three_partition_label, qc, url, url_alphanum

def load_fitzpatrick17k():
    print("── Loading Fitzpatrick17k ──")
    df = pd.read_csv(CONFIG["fitzpatrick_csv"])
    print(f"  Raw rows: {len(df)}")

    # Drop rows with missing/unknown Fitzpatrick scale (-1 = not annotated)
    df = df[df["fitzpatrick_scale"].isin([1, 2, 3, 4, 5, 6])].copy()
    print(f"  After removing unlabeled skin type: {len(df)}")

    # Drop rows flagged by quality control (qc column; non-null = flagged issue)
    if "qc" in df.columns:
        df = df[df["qc"].isna()].copy()
        print(f"  After removing QC-flagged rows: {len(df)}")

    # Drop duplicate md5hash (Fitzpatrick17k has known duplicate images)
    df = df.drop_duplicates(subset="md5hash").copy()
    print(f"  After de-duplication: {len(df)}")

    # Drop rows with missing URLs (can't download the image)
    df = df.dropna(subset=["url"]).copy()
    print(f"  After removing missing URLs: {len(df)}")

    # ── Map to binary melanoma label ──
    # three_partition_label is one of: 'malignant', 'benign', 'non-neoplastic'
    # 'label' holds the specific condition (114 values), so we isolate melanoma
    # specifically rather than treating all malignant lesions as melanoma.
    df["label_clean"] = df["label"].str.lower().str.strip()
    df["binary_label"] = (df["label_clean"] == "melanoma").astype(int)

    print(f"  Melanoma positive: {df['binary_label'].sum()} | "
          f"Negative: {(df['binary_label'] == 0).sum()}")

    # Rename to common schema
    df = df.rename(columns={"fitzpatrick_scale": "fitzpatrick"})
    df["image_id"] = df["md5hash"]
    df["source"]    = "fitzpatrick17k"

    return df[["image_id", "url", "binary_label", "fitzpatrick", "source"]].rename(
        columns={"binary_label": "label"}
    )


def download_fitzpatrick_images(df: pd.DataFrame):
    """
    Download Fitzpatrick17k images from their `url` column.
    Some links are known to be broken (per the dataset README) —
    failures are logged and dropped from the final dataframe.
    """
    print("── Downloading Fitzpatrick17k images ──")
    os.makedirs(CONFIG["fitzpatrick_image_dir"], exist_ok=True)

    def _download_one(row):
        image_id  = row["image_id"]
        url       = row["url"]
        dest_path = os.path.join(CONFIG["fitzpatrick_image_dir"], f"{image_id}.jpg")

        if os.path.exists(dest_path):
            return image_id, dest_path, True

        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(resp.content)
            return image_id, dest_path, True
        except Exception:
            return image_id, None, False

    results = {}
    with ThreadPoolExecutor(max_workers=CONFIG["max_download_workers"]) as executor:
        futures = [executor.submit(_download_one, row) for _, row in df.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            image_id, path, success = future.result()
            results[image_id] = (path, success)

    df["image_path"] = df["image_id"].map(lambda x: results[x][0])
    df["download_ok"] = df["image_id"].map(lambda x: results[x][1])

    n_failed = (~df["download_ok"]).sum()
    print(f"  Failed downloads: {n_failed} / {len(df)} "
          f"({n_failed/len(df)*100:.1f}%) — these will be dropped")

    df = df[df["download_ok"]].drop(columns=["download_ok", "url"])
    return df


# ─────────────────────────────────────────────
# 2. LOAD + CLEAN ISIC ARCHIVE
# ─────────────────────────────────────────────
# Real ISIC metadata columns (per ISIC API v2 / archive export):
#   isic_id, age_approx, anatom_site_general, benign_malignant,
#   diagnosis (or diagnosis_1..5), sex, pixels_x, pixels_y, ...
# NOTE: ISIC metadata generally does NOT include Fitzpatrick skin type.
# These rows are tone-unlabeled and must be excluded from fairness evaluation.

def load_isic(local_image_dir: str):
    print("── Loading ISIC Archive metadata ──")
    df = pd.read_csv(CONFIG["isic_csv"])
    print(f"  Raw rows: {len(df)}")

    # Drop rows with no malignancy label at all
    df = df.dropna(subset=["benign_malignant"]).copy()
    print(f"  After removing missing benign_malignant: {len(df)}")

    # Determine the diagnosis column name (varies by ISIC export version)
    diag_col = "diagnosis" if "diagnosis" in df.columns else "diagnosis_1"

    df["label_clean"] = df[diag_col].astype(str).str.lower().str.strip()
    df["binary_label"] = (df["label_clean"] == "melanoma").astype(int)

    print(f"  Melanoma positive: {df['binary_label'].sum()} | "
          f"Negative: {(df['binary_label'] == 0).sum()}")

    # ISIC has no Fitzpatrick annotation — mark explicitly rather than guessing
    df["fitzpatrick"] = np.nan
    df["source"]       = "isic"
    df["image_id"]     = df["isic_id"]

    # Build expected local image path (assumes images already downloaded
    # via the ISIC Gallery "Download as zip" or API — these archives are
    # large, so this script does not auto-download ISIC images)
    df["image_path"] = df["image_id"].apply(
        lambda x: os.path.join(local_image_dir, f"{x}.jpg")
    )

    # Drop rows whose image file doesn't actually exist locally
    exists_mask = df["image_path"].apply(os.path.exists)
    n_missing = (~exists_mask).sum()
    if n_missing > 0:
        print(f"  Warning: {n_missing} ISIC images not found locally — dropping. "
              f"Confirm they were downloaded to {local_image_dir}")
    df = df[exists_mask].copy()

    return df[["image_id", "image_path", "binary_label", "fitzpatrick", "source"]].rename(
        columns={"binary_label": "label"}
    )


# ─────────────────────────────────────────────
# 3. MERGE + STRATIFIED SPLIT
# ─────────────────────────────────────────────

def merge_and_dedupe(fitz_df: pd.DataFrame, isic_df: pd.DataFrame) -> pd.DataFrame:
    print("── Merging sources ──")
    combined = pd.concat([fitz_df, isic_df], ignore_index=True)
    print(f"  Combined rows: {len(combined)} "
          f"(Fitzpatrick17k: {len(fitz_df)}, ISIC: {len(isic_df)})")

    combined = combined.drop_duplicates(subset="image_path").copy()
    print(f"  After de-duplication on image_path: {len(combined)}")

    return combined


def stratified_split(df: pd.DataFrame):
    """
    Split into train/val/test while preserving:
      - melanoma vs. non-melanoma proportion
      - skin-tone group proportion (where known)
    Rows with unknown Fitzpatrick type (ISIC-only rows) are stratified
    by label only, then merged back in — they're usable for training
    but excluded from the fairness-evaluation slice downstream.
    """
    print("── Creating stratified train/val/test split ──")

    df = df.copy()
    df["has_tone"] = df["fitzpatrick"].notna()

    # Stratify key: combination of label and tone-group (light/dark/unknown)
    def tone_group(row):
        if not row["has_tone"]:
            return "unknown"
        return "dark" if row["fitzpatrick"] >= 4 else "light"

    df["strata"] = df.apply(lambda r: f"{tone_group(r)}_{r['label']}", axis=1)

    train_rows, val_rows, test_rows = [], [], []

    for stratum, group in df.groupby("strata"):
        group = group.sample(frac=1.0, random_state=CONFIG["random_seed"])  # shuffle
        n = len(group)
        n_train = int(n * CONFIG["train_frac"])
        n_val   = int(n * CONFIG["val_frac"])

        train_rows.append(group.iloc[:n_train])
        val_rows.append(group.iloc[n_train:n_train + n_val])
        test_rows.append(group.iloc[n_train + n_val:])

    train_df = pd.concat(train_rows).sample(frac=1.0, random_state=CONFIG["random_seed"])
    val_df   = pd.concat(val_rows).sample(frac=1.0, random_state=CONFIG["random_seed"])
    test_df  = pd.concat(test_rows).sample(frac=1.0, random_state=CONFIG["random_seed"])

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    return train_df, val_df, test_df


def restrict_fairness_eval_to_labeled_tone(df: pd.DataFrame) -> pd.DataFrame:
    """
    For val/test sets specifically: drop rows with unknown Fitzpatrick type.
    Fairness metrics (FNR diff, EOD, DPD) are meaningless without a tone label,
    so evaluation should only run on rows where skin tone is known.
    Training data can still include tone-unknown rows for extra volume.
    """
    before = len(df)
    df = df[df["fitzpatrick"].notna()].copy()
    print(f"  Restricting to tone-labeled rows for eval: {before} → {len(df)}")
    return df


# ─────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # ── Fitzpatrick17k ──
    fitz_df = load_fitzpatrick17k()
    if CONFIG["download_fitzpatrick_images"]:
        fitz_df = download_fitzpatrick_images(fitz_df)
    else:
        # Assume images already downloaded with filenames {image_id}.jpg
        fitz_df["image_path"] = fitz_df["image_id"].apply(
            lambda x: os.path.join(CONFIG["fitzpatrick_image_dir"], f"{x}.jpg")
        )
        fitz_df = fitz_df.drop(columns=["url"], errors="ignore")
        exists_mask = fitz_df["image_path"].apply(os.path.exists)
        print(f"  Local Fitzpatrick images found: {exists_mask.sum()} / {len(fitz_df)}")
        fitz_df = fitz_df[exists_mask].copy()

    # ── ISIC Archive ──
    isic_df = load_isic(CONFIG["isic_image_dir"])

    # ── Merge ──
    combined = merge_and_dedupe(
        fitz_df[["image_path", "label", "fitzpatrick", "source"]],
        isic_df[["image_path", "label", "fitzpatrick", "source"]],
    )

    # ── Split ──
    train_df, val_df, test_df = stratified_split(combined)

    # Fairness evaluation requires known skin tone — restrict val/test
    val_df  = restrict_fairness_eval_to_labeled_tone(val_df)
    test_df = restrict_fairness_eval_to_labeled_tone(test_df)

    # ── Save final CSVs (matches MelanomaDataset's expected schema) ──
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = os.path.join(CONFIG["output_dir"], f"{split_name}.csv")
        split_df[["image_path", "label", "fitzpatrick"]].to_csv(out_path, index=False)
        print(f"  Saved {split_name}.csv → {out_path} ({len(split_df)} rows)")

    # ── Summary report ──
    print("\n══ MERGE SUMMARY ════════════════════════════════════")
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_total = len(split_df)
        n_mel   = split_df["label"].sum()
        n_dark  = (split_df["fitzpatrick"] >= 4).sum() if n_total else 0
        n_light = split_df["fitzpatrick"].notna().sum() - n_dark
        print(f"  {split_name:5s} | total={n_total:6d}  melanoma={n_mel:5d}  "
              f"light={n_light:5d}  dark={n_dark:5d}")
    print("══════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
