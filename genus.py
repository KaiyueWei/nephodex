"""genus.py — fine-tuned cloud-genus classifier (the Well-Tuned stage).

Stage 4 of the Nephodex pipeline, kept as a clean swappable function so app.py
can be wired up *before* training finishes. Until the fine-tuned weights are
published to the Hub, `classify_genus()` returns a stub and the app still runs
end to end.

Contract (see Plan.md §5):
    classify_genus(rgb_crop: PIL.Image) -> (label: str, confidence: float)
"""

from __future__ import annotations

import functools
import os

import torch
from PIL import Image

# Override with the env var if your Hub username/repo differs.
MODEL_ID = os.environ.get("GENUS_MODEL_ID", "KaiyueWei/nephodex-cloud-genus")

# Returned until real weights land (lets app.py ship stage 4 early).
_STUB: tuple[str, float] = ("unidentified", 0.0)

_device = "cuda" if torch.cuda.is_available() else "cpu"


@functools.lru_cache(maxsize=1)
def _load():
    """Load processor + model once.

    Returns None if the weights aren't available yet (repo missing, offline,
    or load error) so callers transparently fall back to the stub.
    """
    try:
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
        )

        proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = (
            AutoModelForImageClassification.from_pretrained(MODEL_ID)
            .to(_device)
            .eval()
        )
        return proc, model
    except Exception as exc:  # noqa: BLE001 — any failure means "not ready yet"
        print(f"[genus] model '{MODEL_ID}' unavailable ({exc}); using stub.")
        return None


def classify_genus(rgb_crop: Image.Image) -> tuple[str, float]:
    """Classify a flattened RGB sticker into a cloud genus.

    Args:
        rgb_crop: the sticker composited onto white (see app._flatten_on_white).

    Returns:
        (label, confidence). Falls back to the stub until real weights exist.
    """
    loaded = _load()
    if loaded is None:
        return _STUB

    proc, model = loaded
    inputs = proc(images=rgb_crop, return_tensors="pt").to(_device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = logits.softmax(-1)[0]
    idx = int(probs.argmax())
    return model.config.id2label[idx], float(probs[idx])
