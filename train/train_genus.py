"""train_genus.py — fine-tune the cloud-genus classifier (Well-Tuned badge).

Loads CCSN, makes a stratified 80/10/10 split, fine-tunes a DeiT-Tiny head, and
`push_to_hub`. The training logic lives in a plain `_train()` function wrapped
two ways:

    # Remote, on a Modal GPU (HF_TOKEN comes from a Modal Secret named "huggingface"):
    modal run train/train_genus.py

    # Local fallback (uses HF_TOKEN from your environment / .env):
    python train/train_genus.py --local --epochs 10

Dataset notes (Plan.md §4): CCSN ships a single `train` split, so we split here.
Verify `id2label` after the first run — the mirror's README is empty.
"""

from __future__ import annotations

import argparse
import os

# Defaults (override via CLI or env). Keep repo id in sync with genus.py.
DATASET_ID = os.environ.get("GENUS_DATASET_ID", "aduuuuuu/CCSN")
BASE_MODEL = os.environ.get("GENUS_BASE_MODEL", "facebook/deit-tiny-patch16-224")
REPO_ID = os.environ.get("GENUS_MODEL_ID", "KaiyueWei/nephodex-cloud-genus")
GPU = os.environ.get("GENUS_GPU", "t4")
SEED = 42


# ---------------------------------------------------------------------------
# Core training logic (pure — no Modal, runs anywhere with a GPU or CPU).
# ---------------------------------------------------------------------------
def _train(
    dataset_id: str = DATASET_ID,
    base_model: str = BASE_MODEL,
    repo_id: str = REPO_ID,
    epochs: int = 10,
    lr: float = 5e-4,
    batch_size: int = 32,
    freeze_backbone: bool = False,
    push: bool = True,
) -> dict:
    import numpy as np
    import torch
    from datasets import load_dataset
    from torchvision.transforms import (
        CenterCrop,
        Compose,
        Normalize,
        RandomHorizontalFlip,
        RandomResizedCrop,
        Resize,
        ToTensor,
    )
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        Trainer,
        TrainingArguments,
    )

    # --- Data ---------------------------------------------------------------
    ds = load_dataset(dataset_id, split="train")
    image_col = "image" if "image" in ds.column_names else ds.column_names[0]
    label_col = "label" if "label" in ds.column_names else ds.column_names[-1]

    labels = ds.features[label_col].names  # ClassLabel -> ordered names
    id2label = {i: name for i, name in enumerate(labels)}
    label2id = {name: i for i, name in enumerate(labels)}
    print(f"[train] {len(ds)} images · {len(labels)} classes: {labels}")

    # Stratified 80/10/10.
    split = ds.train_test_split(test_size=0.2, stratify_by_column=label_col, seed=SEED)
    tmp = split["test"].train_test_split(
        test_size=0.5, stratify_by_column=label_col, seed=SEED
    )
    train_ds, val_ds, test_ds = split["train"], tmp["train"], tmp["test"]

    # --- Transforms (clouds are orientation-tolerant -> flips + mild crops) --
    proc = AutoImageProcessor.from_pretrained(base_model)
    size = proc.size.get("height", proc.size.get("shortest_edge", 224))
    normalize = Normalize(mean=proc.image_mean, std=proc.image_std)
    train_tf = Compose(
        [RandomResizedCrop(size, scale=(0.8, 1.0)), RandomHorizontalFlip(), ToTensor(), normalize]
    )
    eval_tf = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])

    def _apply(transform):
        def fn(batch):
            batch["pixel_values"] = [
                transform(img.convert("RGB")) for img in batch[image_col]
            ]
            return batch

        return fn

    train_ds.set_transform(_apply(train_tf))
    val_ds.set_transform(_apply(eval_tf))
    test_ds.set_transform(_apply(eval_tf))

    # --- Model --------------------------------------------------------------
    model = AutoModelForImageClassification.from_pretrained(
        base_model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # replaces the pretrained head
    )
    if freeze_backbone:
        # Transfer-learning mode: train only the classifier head (faster, a few
        # points lower accuracy). Full fine-tune is the default for this tiny model.
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def collate(examples):
        return {
            "pixel_values": torch.stack([e["pixel_values"] for e in examples]),
            "labels": torch.tensor([e[label_col] for e in examples]),
        }

    def compute_metrics(pred):
        preds = np.argmax(pred.predictions, axis=1)
        return {"accuracy": float((preds == pred.label_ids).mean())}

    args = TrainingArguments(
        output_dir="genus-out",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=20,
        remove_unused_columns=False,
        report_to="none",
        push_to_hub=push,
        hub_model_id=repo_id,
        hub_token=os.environ.get("HF_TOKEN"),
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_metrics = trainer.evaluate(test_ds)
    print(f"[train] held-out test: {test_metrics}")

    if push:
        trainer.push_to_hub(commit_message=f"genus classifier — {test_metrics}")
        proc.push_to_hub(repo_id, token=os.environ.get("HF_TOKEN"))
        print(f"[train] pushed to https://huggingface.co/{repo_id}")

    return test_metrics


# ---------------------------------------------------------------------------
# Modal wrapper (only when `modal` is importable, i.e. via `modal run`).
# ---------------------------------------------------------------------------
try:
    import modal

    _image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch",
        "torchvision",
        "transformers",
        "datasets",
        "accelerate",
        "pillow",
    )
    app = modal.App("nephodex-genus")

    @app.function(
        gpu=GPU,
        image=_image,
        timeout=60 * 60,
        secrets=[modal.Secret.from_name("huggingface")],  # provides HF_TOKEN
    )
    def train_remote(epochs: int = 10, freeze_backbone: bool = False) -> dict:
        return _train(epochs=epochs, freeze_backbone=freeze_backbone)

    @app.local_entrypoint()
    def main(epochs: int = 10, freeze_backbone: bool = False) -> None:
        print(train_remote.remote(epochs=epochs, freeze_backbone=freeze_backbone))

except ImportError:
    # modal not installed — local-only mode still works via __main__ below.
    pass


# ---------------------------------------------------------------------------
# Local fallback: `python train/train_genus.py --local`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the cloud-genus classifier.")
    parser.add_argument("--local", action="store_true", help="train in this process (no Modal)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-push", action="store_true", help="skip push_to_hub")
    cli = parser.parse_args()

    if not cli.local:
        parser.error("run with --local, or use `modal run train/train_genus.py` for GPU.")

    _train(
        epochs=cli.epochs,
        lr=cli.lr,
        batch_size=cli.batch_size,
        freeze_backbone=cli.freeze_backbone,
        push=not cli.no_push,
    )
