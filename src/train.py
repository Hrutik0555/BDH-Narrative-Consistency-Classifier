"""Two-phase training: unsupervised BDH pretraining, then supervised fine-tuning.

Run from the repository root:
    python -m src.train
"""
import copy
import os
import sys

# Allow `python src/train.py` as well as `python -m src.train`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src import config as C
from src.dataset import NarrativeDataset
from src.models import BDHModel
from src.utils import set_seed, to_label_tensor


def _to_id(x):
    return x.item() if isinstance(x, torch.Tensor) else x


def train_model():
    set_seed(C.SEED)
    device = C.DEVICE
    print(f"Using device: {device}")

    train_df = pd.read_csv(C.TRAIN_CSV)
    test_df = pd.read_csv(C.TEST_CSV)

    # Full labeled set is used (label-free) for pretraining
    full_train_ds = NarrativeDataset(train_df)

    # Train / validation split for the supervised phase
    val_size = int(len(full_train_ds) * C.VAL_SPLIT)
    train_size = len(full_train_ds) - val_size
    train_ds, val_ds = random_split(full_train_ds, [train_size, val_size])

    pretrain_loader = DataLoader(full_train_ds, batch_size=C.BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=C.BATCH_SIZE)
    test_loader = DataLoader(NarrativeDataset(test_df, train=False), batch_size=C.BATCH_SIZE)

    model = BDHModel().to(device)

    pretrain_optimizer = torch.optim.AdamW(model.bdh.parameters(), lr=C.LR_PRETRAIN)
    finetune_optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=C.LR_FINETUNE
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        finetune_optimizer, T_max=C.FINETUNE_EPOCHS
    )

    # ---------------- Phase 1: unsupervised BDH pretraining ----------------
    print("=== Phase 1: Unsupervised BDH Pretraining ===")
    model.train()
    for epoch in range(C.PRETRAIN_EPOCHS):
        total_loss = 0.0
        pretrain_optimizer.zero_grad()
        for step, batch in enumerate(
            tqdm(pretrain_loader, desc=f"Pretrain {epoch + 1}/{C.PRETRAIN_EPOCHS}")
        ):
            captions, contents, _, _ = batch
            _, aux_loss, _ = model(captions, contents)
            loss = C.PRETRAIN_LOSS_WEIGHT * aux_loss
            loss.backward()
            total_loss += loss.item()

            if (step + 1) % C.ACCUM_STEPS == 0 or (step + 1) == len(pretrain_loader):
                pretrain_optimizer.step()
                pretrain_optimizer.zero_grad()

        print(f"Pretrain Epoch {epoch + 1} | Avg Friction Loss: "
              f"{total_loss / len(pretrain_loader):.6f}")

    # ---------------- Phase 2: supervised fine-tuning ----------------
    print("\n=== Phase 2: Supervised Fine-tuning ===")
    best_val_acc = 0.0
    best_model_state = None
    val_friction_sums = []
    criterion = nn.CrossEntropyLoss()

    for epoch in range(C.FINETUNE_EPOCHS):
        model.train()
        finetune_optimizer.zero_grad()
        for step, batch in enumerate(
            tqdm(train_loader, desc=f"Finetune {epoch + 1}/{C.FINETUNE_EPOCHS}")
        ):
            captions, contents, labels, _ = batch
            labels = to_label_tensor(labels, device)
            logits, aux_loss, _ = model(captions, contents)
            loss = criterion(logits, labels) + C.AUX_LOSS_WEIGHT * aux_loss
            loss.backward()

            if (step + 1) % C.ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                finetune_optimizer.step()
                finetune_optimizer.zero_grad()

        # Validation
        model.eval()
        correct = total = 0
        epoch_frictions = []
        with torch.no_grad():
            for batch in val_loader:
                captions, contents, labels, _ = batch
                labels = to_label_tensor(labels, device)
                logits, _, _, _, friction_sum = model(
                    captions, contents, return_friction_stats=True
                )
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                epoch_frictions.extend(friction_sum.cpu().numpy())

        val_acc = correct / max(total, 1)
        print(f"\nValidation Accuracy: {val_acc:.4f}")
        val_friction_sums.extend(epoch_frictions)

        if val_acc > best_val_acc or best_model_state is None:
            best_val_acc = val_acc
            # deepcopy: state_dict() alone returns live references to the weights
            best_model_state = copy.deepcopy(model.state_dict())

        scheduler.step()

    model.load_state_dict(best_model_state)

    # Friction threshold calibrated on validation data
    threshold = (
        np.percentile(val_friction_sums, C.THRESHOLD_PERCENTILE)
        if val_friction_sums else 0.1
    )

    # ---------------- Test inference ----------------
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Inference"):
            captions, contents, _, ids = batch
            logits, _, _, _, friction_sum = model(
                captions, contents, return_friction_stats=True
            )
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            friction_vals = friction_sum.cpu().numpy()

            for i in range(len(ids)):
                rationale = (
                    "High belief friction indicates narrative inconsistency."
                    if friction_vals[i] > threshold
                    else "Stable belief evolution across narrative events."
                )
                outputs.append(
                    {"id": _to_id(ids[i]), "label": int(preds[i]), "rationale": rationale}
                )

    pd.DataFrame(outputs).to_csv(C.OUTPUT_CSV, index=False)
    print(f"\nPredictions saved to {C.OUTPUT_CSV}. Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    train_model()
