"""
Melanoma Bias Mitigation Pipeline
==================================
Full pipeline for binary melanoma classification using EfficientNet-B0
with two dataset balancing strategies to reduce racial bias:
  - Short-term: Dataset filtering (remove high-loss / outlier samples)
  - Long-term:  Dataset diversification (weighted sampling + augmentation)

Requirements:
  pip install torch torchvision scikit-learn matplotlib seaborn pandas
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from PIL import Image
import os


# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    "image_size":        224,
    "batch_size":        32,
    "num_epochs":        20,
    "learning_rate":     1e-4,
    "dropout":           0.2,
    "filter_percentile": 95,    # flag top 5% highest-loss samples
    "device":            "cuda" if torch.cuda.is_available() else "cpu",
    "seed":              42,
}

torch.manual_seed(CONFIG["seed"])
device = torch.device(CONFIG["device"])
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────

class MelanomaDataset(Dataset):
    """
    Expects a CSV with columns:
      - image_path     : path to the image file
      - label          : 0 = non-melanoma, 1 = melanoma
      - fitzpatrick    : Fitzpatrick skin type (1–6)

    Skin-tone groups:
      Light = Fitzpatrick I–III  (value 1, 2, 3)
      Dark  = Fitzpatrick IV–VI  (value 4, 5, 6)
    """

    def __init__(self, csv_path: str, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.transform = transform

        # Binary skin-tone group: 0 = Light, 1 = Dark
        self.df["skin_tone_group"] = (self.df["fitzpatrick"] >= 4).astype(int)

        # Expose as lists for sampler / fairness eval
        self.labels           = self.df["label"].tolist()
        self.skin_tone_labels = self.df["skin_tone_group"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label     = torch.tensor(row["label"],           dtype=torch.float32)
        skin_tone = torch.tensor(row["skin_tone_group"], dtype=torch.long)
        return image, label, skin_tone, idx   # idx needed for filtering


# ─────────────────────────────────────────────
# 3. TRANSFORMS
# ─────────────────────────────────────────────

# Standard transform for Light-skin-tone samples and validation/test
STANDARD_TRANSFORM = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                         std= [0.229, 0.224, 0.225]),
])

# Heavier augmentation applied to Dark-skin-tone samples (long-term strategy)
DARK_TONE_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = STANDARD_TRANSFORM


# ─────────────────────────────────────────────
# 4. MODEL SETUP
# ─────────────────────────────────────────────

def build_model() -> nn.Module:
    """
    Load EfficientNet-B0 with ImageNet pretrained weights.
    Replace the final classifier for binary melanoma output.
    All layers in self.features are kept intact — only the head changes.
    """
    model       = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features          # 1280 for B0

    model.classifier = nn.Sequential(
        nn.Dropout(p=CONFIG["dropout"], inplace=True),
        nn.Linear(in_features, 1),                         # 1 logit → BCEWithLogitsLoss
    )
    return model.to(device)


# ─────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0

    for images, labels, skin_tones, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels, _, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images).squeeze(1)
            loss   = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    auc      = roc_auc_score(all_labels, all_preds)
    return avg_loss, auc


def run_training(model, train_loader, val_loader, tag="baseline"):
    """Train for CONFIG['num_epochs'] and return the best model weights."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_auc    = 0.0
    best_state  = None
    history     = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        train_loss        = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_auc = validate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        print(f"[{tag}] Epoch {epoch:02d}/{CONFIG['num_epochs']} | "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"[{tag}] Best val AUC: {best_auc:.4f}\n")
    return model, history


# ─────────────────────────────────────────────
# 6. SHORT-TERM STRATEGY — DATASET FILTERING
# ─────────────────────────────────────────────

def flag_high_loss_samples(model, loader, percentile=95):
    """
    Run inference over the training set with the baseline model.
    Flag samples whose per-sample loss exceeds the given percentile.

    Returns:
        flagged_indices (set): dataset indices to remove before retraining.
    """
    model.eval()
    criterion    = nn.BCEWithLogitsLoss(reduction="none")
    sample_losses = {}

    with torch.no_grad():
        for images, labels, _, indices in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images).squeeze(1)
            losses = criterion(logits, labels).cpu().tolist()

            for idx, loss_val in zip(indices.tolist(), losses):
                sample_losses[idx] = loss_val

    loss_values = np.array(list(sample_losses.values()))
    threshold   = np.percentile(loss_values, percentile)

    flagged = {idx for idx, lv in sample_losses.items() if lv > threshold}
    print(f"[Filtering] Threshold (p{percentile}): {threshold:.4f} | "
          f"Flagged {len(flagged)} / {len(sample_losses)} samples")
    return flagged


def apply_filtering(dataset, flagged_indices):
    """Return a Subset with flagged samples removed."""
    keep = [i for i in range(len(dataset)) if i not in flagged_indices]
    print(f"[Filtering] Keeping {len(keep)} / {len(dataset)} samples")
    return Subset(dataset, keep)


# ─────────────────────────────────────────────
# 7. LONG-TERM STRATEGY — DIVERSIFICATION
# ─────────────────────────────────────────────

def make_skin_tone_sampler(dataset):
    """
    Build a WeightedRandomSampler that oversamples Dark (Fitzpatrick IV–VI)
    images so every training batch is skin-tone balanced.
    """
    tone_labels = torch.tensor(dataset.skin_tone_labels)
    counts      = torch.bincount(tone_labels).float()         # [n_light, n_dark]
    weights_per_class = 1.0 / counts
    sample_weights    = weights_per_class[tone_labels]

    sampler = WeightedRandomSampler(
        weights    = sample_weights,
        num_samples= len(sample_weights),
        replacement= True,
    )
    print(f"[Diversification] Light: {counts[0].int()} | Dark: {counts[1].int()} | "
          f"Sampler weight ratio: {(weights_per_class[1]/weights_per_class[0]):.2f}x")
    return sampler


class AugmentedMelanomaDataset(Dataset):
    """
    Wraps MelanomaDataset and applies DARK_TONE_TRANSFORM to Fitzpatrick IV–VI
    images and STANDARD_TRANSFORM to Fitzpatrick I–III images.
    Used for the long-term diversification strategy.
    """

    def __init__(self, base_dataset: MelanomaDataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        row       = self.base.df.iloc[idx]
        image     = Image.open(row["image_path"]).convert("RGB")
        skin_tone = row["skin_tone_group"]

        transform = DARK_TONE_TRANSFORM if skin_tone == 1 else STANDARD_TRANSFORM
        image     = transform(image)

        label     = torch.tensor(row["label"],     dtype=torch.float32)
        skin_tone = torch.tensor(skin_tone,        dtype=torch.long)
        return image, label, skin_tone, idx


# ─────────────────────────────────────────────
# 8. FAIRNESS EVALUATION
# ─────────────────────────────────────────────

def evaluate_fairness(model, loader, threshold=0.5):
    """
    Evaluate model performance broken down by skin-tone group.

    Fairness metrics computed:
      - Recall (sensitivity) per group
      - False Negative Rate (FNR) per group
      - FNR Difference          (primary bias indicator)
      - Equal Opportunity Diff  (EOD  = recall difference)
      - Demographic Parity Diff (DPD)
      - Disparate Impact Ratio  (DIR)

    Returns:
        dict with per-group and aggregate metrics.
    """
    model.eval()

    groups = {
        "light": {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "probs": [], "labels": []},
        "dark":  {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "probs": [], "labels": []},
    }

    with torch.no_grad():
        for images, labels, skin_tones, _ in loader:
            logits = model(images.to(device)).squeeze(1).cpu()
            probs  = torch.sigmoid(logits)
            preds  = (probs > threshold).long()

            for pred, prob, label, tone in zip(preds, probs, labels, skin_tones):
                g = "dark" if tone.item() == 1 else "light"
                groups[g]["probs"].append(prob.item())
                groups[g]["labels"].append(label.item())

                if label == 1 and pred == 1:   groups[g]["tp"] += 1
                elif label == 1 and pred == 0: groups[g]["fn"] += 1
                elif label == 0 and pred == 1: groups[g]["fp"] += 1
                else:                          groups[g]["tn"] += 1

    metrics = {}
    for g, r in groups.items():
        tp, fn, fp, tn = r["tp"], r["fn"], r["fp"], r["tn"]
        total    = tp + fn + fp + tn
        pos      = tp + fn
        neg      = fp + tn
        recall   = tp / (pos  + 1e-9)
        fnr      = fn / (pos  + 1e-9)
        fpr      = fp / (neg  + 1e-9)
        prec     = tp / (tp + fp + 1e-9)
        f1       = 2 * prec * recall / (prec + recall + 1e-9)
        pred_pos = (tp + fp) / (total + 1e-9)   # predicted positive rate
        auc      = roc_auc_score(r["labels"], r["probs"]) if len(set(r["labels"])) > 1 else float("nan")

        metrics[g] = {
            "recall":    recall,
            "fnr":       fnr,
            "fpr":       fpr,
            "precision": prec,
            "f1":        f1,
            "auc":       auc,
            "pred_pos_rate": pred_pos,
            "n_samples": total,
        }

    # Aggregate fairness metrics
    metrics["fnr_difference"]       = abs(metrics["light"]["fnr"]      - metrics["dark"]["fnr"])
    metrics["eod"]                  = abs(metrics["light"]["recall"]    - metrics["dark"]["recall"])
    metrics["dpd"]                  = abs(metrics["light"]["pred_pos_rate"] - metrics["dark"]["pred_pos_rate"])
    metrics["dir"]                  = (metrics["dark"]["pred_pos_rate"] /
                                       (metrics["light"]["pred_pos_rate"] + 1e-9))

    # Print summary
    print("\n── Fairness Evaluation ──────────────────────────────")
    for g in ["light", "dark"]:
        m = metrics[g]
        print(f"  {g.upper():5s} | Recall={m['recall']:.3f}  FNR={m['fnr']:.3f}  "
              f"F1={m['f1']:.3f}  AUC={m['auc']:.3f}  n={m['n_samples']}")
    print(f"  FNR Difference : {metrics['fnr_difference']:.4f}  (target → 0)")
    print(f"  EOD            : {metrics['eod']:.4f}            (target → 0)")
    print(f"  DPD            : {metrics['dpd']:.4f}            (target → 0)")
    print(f"  DIR            : {metrics['dir']:.4f}            (target → 1)")
    print("─────────────────────────────────────────────────────\n")
    return metrics


# ─────────────────────────────────────────────
# 9. VISUALISATION
# ─────────────────────────────────────────────

def plot_training_history(histories: dict, save_path="training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for tag, h in histories.items():
        axes[0].plot(h["val_loss"], label=tag)
        axes[1].plot(h["val_auc"],  label=tag)

    axes[0].set_title("Validation Loss");  axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].set_title("Validation AUC");   axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training history → {save_path}")


def plot_fairness_comparison(all_metrics: dict, save_path="fairness_comparison.png"):
    """
    Bar chart comparing FNR Difference, EOD, DPD across all strategies.
    Lower is better for all three metrics.
    """
    strategies = list(all_metrics.keys())
    fnr_diffs  = [all_metrics[s]["fnr_difference"] for s in strategies]
    eods       = [all_metrics[s]["eod"]            for s in strategies]
    dpds       = [all_metrics[s]["dpd"]            for s in strategies]

    x    = np.arange(len(strategies))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x - w, fnr_diffs, w, label="FNR Difference")
    ax.bar(x,     eods,      w, label="EOD")
    ax.bar(x + w, dpds,      w, label="DPD")

    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylabel("Metric value  (lower = less bias)")
    ax.set_title("Fairness metrics by strategy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved fairness comparison → {save_path}")


def plot_per_group_recall(all_metrics: dict, save_path="recall_by_group.png"):
    """Grouped bar chart: recall for Light vs Dark per strategy."""
    strategies    = list(all_metrics.keys())
    light_recalls = [all_metrics[s]["light"]["recall"] for s in strategies]
    dark_recalls  = [all_metrics[s]["dark"]["recall"]  for s in strategies]

    x   = np.arange(len(strategies))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x - w/2, light_recalls, w, label="Light (I–III)", color="steelblue")
    ax.bar(x + w/2, dark_recalls,  w, label="Dark  (IV–VI)", color="coral")

    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall  (higher = fewer missed melanomas)")
    ax.set_title("Recall by skin-tone group and strategy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved recall by group → {save_path}")


# ─────────────────────────────────────────────
# 10. MAIN PIPELINE
# ─────────────────────────────────────────────

def main(train_csv, val_csv, test_csv):
    """
    Run all three experimental conditions:
      1. Baseline         — original imbalanced dataset
      2. Filtered         — short-term: remove high-loss samples
      3. Diversified      — long-term: weighted sampling + augmentation

    Args:
        train_csv : path to training split CSV
        val_csv   : path to validation split CSV
        test_csv  : path to test split CSV
    """

    # ── Load datasets ──────────────────────────────────────────
    train_dataset = MelanomaDataset(train_csv, transform=STANDARD_TRANSFORM)
    val_dataset   = MelanomaDataset(val_csv,   transform=VAL_TRANSFORM)
    test_dataset  = MelanomaDataset(test_csv,  transform=VAL_TRANSFORM)

    val_loader  = DataLoader(val_dataset,  batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    all_metrics  = {}
    all_histories = {}

    # ── CONDITION 1: Baseline ──────────────────────────────────
    print("=" * 55)
    print("CONDITION 1 — Baseline")
    print("=" * 55)

    baseline_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True
    )
    baseline_model = build_model()
    baseline_model, history = run_training(
        baseline_model, baseline_loader, val_loader, tag="baseline"
    )
    all_histories["baseline"] = history
    all_metrics["baseline"]   = evaluate_fairness(baseline_model, test_loader)

    # ── CONDITION 2: Short-term — Dataset Filtering ────────────
    print("=" * 55)
    print("CONDITION 2 — Short-term: Dataset Filtering")
    print("=" * 55)

    # Use the baseline model to identify problematic samples
    flagged        = flag_high_loss_samples(
        baseline_model, baseline_loader,
        percentile=CONFIG["filter_percentile"]
    )
    filtered_ds    = apply_filtering(train_dataset, flagged)
    filtered_loader = DataLoader(
        filtered_ds, batch_size=CONFIG["batch_size"], shuffle=True
    )

    filtered_model = build_model()   # fresh model, retrained on cleaned data
    filtered_model, history = run_training(
        filtered_model, filtered_loader, val_loader, tag="filtered"
    )
    all_histories["filtered"] = history
    all_metrics["filtered"]   = evaluate_fairness(filtered_model, test_loader)

    # ── CONDITION 3: Long-term — Diversification ───────────────
    print("=" * 55)
    print("CONDITION 3 — Long-term: Diversification")
    print("=" * 55)

    aug_dataset  = AugmentedMelanomaDataset(train_dataset)
    sampler      = make_skin_tone_sampler(train_dataset)
    diverse_loader = DataLoader(
        aug_dataset,
        batch_size=CONFIG["batch_size"],
        sampler=sampler,               # shuffle=False when sampler is provided
    )

    diverse_model = build_model()
    diverse_model, history = run_training(
        diverse_model, diverse_loader, val_loader, tag="diversified"
    )
    all_histories["diversified"] = history
    all_metrics["diversified"]   = evaluate_fairness(diverse_model, test_loader)

    # ── Visualise results ──────────────────────────────────────
    plot_training_history(all_histories)
    plot_fairness_comparison(all_metrics)
    plot_per_group_recall(all_metrics)

    # ── Summary table ──────────────────────────────────────────
    print("\n══ FINAL COMPARISON ════════════════════════════════════")
    print(f"{'Strategy':<15} {'FNR Diff':>10} {'EOD':>8} {'DPD':>8} {'DIR':>8}")
    print("-" * 50)
    for s, m in all_metrics.items():
        print(f"{s:<15} {m['fnr_difference']:>10.4f} {m['eod']:>8.4f} "
              f"{m['dpd']:>8.4f} {m['dir']:>8.4f}")
    print("════════════════════════════════════════════════════════\n")

    return all_metrics


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Replace these paths with your actual CSV files
    main(
        train_csv="data/train.csv",
        val_csv  ="data/val.csv",
        test_csv ="data/test.csv",
    )
