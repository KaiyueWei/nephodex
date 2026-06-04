# Nephodex — Implementation Plan

**Event:** Build Small Hackathon · Chapter Two ("An Adventure in Thousand Token Wood")
**Build window:** June 5–15, 2026 · **Builder:** solo · **Submit:** June 15
**Targeting badges:** Off the Grid · Off-Brand · Field Notes · Well-Tuned
**Resources secured:** Modal credits (training) · Hugging Face credits (GPU Space hosting/inference)

**Guiding rule:** the fine-tuning work is  *additive, never blocking* . There must be a
complete, submittable Space by the end of Weekend 1. If the Well-Tuned work later
fails, you drop one badge — not the project.

---

## 1. Final architecture

A four-stage local pipeline. The fine-tuned classifier is given a *visible* job so
the Well-Tuned model is load-bearing in the UI, not hidden.

```
Sky photo ─► MobileSAM ─► OpenCV alpha-matte ─► transparent "sticker" crop
                                                   │
                          ┌────────────────────────┼─────────────────────────┐
                          ▼                         ▼                         ▼
                    CLIP embedding          SmolVLM2 (fun name)     fine-tuned classifier
                          │                         │                  (real genus)
                          ▼                         ▼                         ▼
                  "seen this before?"      "The Drowsy Leviathan"     "likely Cumulus 91%"
                          └────────────────────────┴─────────────────────────┘
                                                   ▼
                                         Nephodex card + dex gallery
```

### Parameter budget (cap = 32B)

| Stage                                 | Model                                            | ~Params             |
| ------------------------------------- | ------------------------------------------------ | ------------------- |
| Segment                               | MobileSAM (`dhkim2810/MobileSAM`)              | ~0.01B              |
| Embed / similarity                    | CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) | ~0.15B              |
| Whimsical name                        | SmolVLM2-2.2B-Instruct                           | ~2.2B               |
| **Real genus (your fine-tune)** | ViT-Tiny / DeiT-Tiny head                        | ~0.005–0.022B      |
| **Total**                       |                                                  | **≈ 2.4B**✅ |

Everything runs locally with no cloud APIs → **Off the Grid** is satisfied by the
architecture itself.

---

## 2. Repo / Space file manifest

```
nephodex/
├── app.py               # Gradio app (have draft; extend with classifier stage)
├── requirements.txt     # have draft; add timm if used
├── README.md            # HF Space header + write-up (Field Notes lives here or links out)
├── genus.py             # NEW: loads your fine-tuned model, classify_genus()
├── assets/              # 10–15 sample sky photos for demo + testing
└── train/
    └── train_genus.py     # Modal training job (`modal run train/train_genus.py`)
```

Published separately to the Hub: `your-username/nephodex-cloud-genus` (the fine-tuned weights).

---

## 3. Phased build

### Phase 0 — June 4 (today, planning only — do NOT build the app yet)

* [ ] Confirm you registered before the June 3 deadline; if you missed it, check whether the org still accepts joins.
* [ ] **De-risk the dataset:** confirm you can actually download  **HBMCD** . If not, switch to **CCSN** (publicly available, ~11 genera) — identical job for the badge. Decide  *now* , write the choice here: `DATASET = ____`.
* [ ] Training runs on **Modal** (credits secured) — a GPU function trains and pushes straight to the Hub; no Colab session timeouts.
* [ ] Shoot/collect 10–15 of your own sky photos → `assets/`.
* [ ] Re-read the HTML mockup; the CSS variables at its top are your design tokens.

### Weekend 1 — the core (priority order; ship something every day)

**Fri Jun 5 — zero-shot pipeline live.** *Riskiest thing first.*

* [ ] Create the Space — use a **credit-covered GPU Space** (HF credits secured); this also removes the ZeroGPU device-placement risk entirely.
* [ ] Deploy `app.py`; get segment → name → collect working end to end.
* [ ] **De-risk the SmolVLM2 processor call** — that's the single most fragile line; if it errors, fix today.

* **Done when:** you can upload a photo, click a cloud, and get a named sticker in the dex. → **Off the Grid secured.**

**Sat Jun 6 — custom UI.**

* [ ] Fold the mockup styling into the Gradio CSS: twilight palette, Fraunces/Karla, dex cards, rarity chips, specimen counter, checkerboard sticker stage.

* **Done when:** the live Space visibly matches the mockup. → **Off-Brand secured.**

**Sun Jun 7 — fine-tune + publish (no integration yet).**

* [ ] Train the genus classifier (see §4) and `push_to_hub`.

* **Done when:** `your-username/nephodex-cloud-genus` exists on the Hub and classifies a test crop. → **Well-Tuned model published.**

### Weekdays — Jun 8–11: integrate & harden

* [ ] Add `genus.py` and wire the classifier into `app.py` as stage 4 (see §5).
* [ ] Surface the genus label on the result card and dex entries.
* [ ] Robustness: no-cloud-found fallback, webcam capture, mobile layout.
* [ ] Keep **one buffer day** (Jun 11) for whatever broke.

* **Done when:** Well-Tuned is visible in the UI and the app survives messy inputs.

### Weekend 2 — Jun 12–14: show, don't tell

* [ ] **Demo video** (60–90s): the click → scan → name → collect loop on real photos. *This is scored, not optional.*
* [ ] **Field Notes** write-up: small-VLM quirks, the segmentation fix, fine-tuning a tiny classifier, what surprised you. → **Field Notes secured.**
* [ ] Social post drafted.
* [ ] (Optional) stretch badges — see §7.

### Jun 15 — submit early

* [ ] Space link · demo video · social post · model link all locked **before** the deadline.

---

## 4. Fine-tuning sub-plan (Well-Tuned)

**Dataset:** HBMCD if obtainable, else CCSN (decide in Phase 0). Both label ground-based
sky photos into ~10–11 cloud genera (cumulus, cirrus, stratus, …). Standard split 80/10/10.

**Backbone:** a small pretrained image model fine-tuned with a new classification head —
`facebook/deit-tiny-patch16-224` (~5M) or a `timm` ViT-Tiny. Keeps you far under the cap
and trains in minutes on a **Modal** GPU function (credits secured).

**Approach:** transfer learning — freeze most of the backbone, train the head (+ optionally
unfreeze the last block), `transformers` `Trainer` or a short PyTorch loop. Augment with
flips / mild crops; clouds are orientation-tolerant.

**Target:** top-1 accuracy ~85–90% (published models reach ~90%+ on these sets). Good enough —
it's a factual garnish, not the headline.

**Publish:** `model.push_to_hub("your-username/nephodex-cloud-genus")` + a short model card
(dataset, classes, accuracy). The published model is what the badge requires.

---

## 5. Integration contract for the classifier stage

Keep it a clean, swappable function so the app can be wired up *before* training finishes —
stub it to return `("…", 0.0)` until the real weights land.

```python
# genus.py
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch

_MODEL_ID = "your-username/nephodex-cloud-genus"
_proc  = AutoImageProcessor.from_pretrained(_MODEL_ID)
_model = AutoModelForImageClassification.from_pretrained(_MODEL_ID).to(device).eval()

def classify_genus(rgb_crop):           # input: PIL RGB (the flattened sticker)
    inputs = _proc(images=rgb_crop, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = logits.softmax(-1)[0]
    idx = int(probs.argmax())
    return _model.config.id2label[idx], float(probs[idx])   # (label, confidence)
```

**Where it plugs in (`app.py`):** inside `process_pipeline`, after the sticker is made,
call `classify_genus(_flatten_on_white(sticker))` and append to the card, e.g.
`✨ {nickname} · likely {genus} ({conf*100:.0f}%)`. Store the genus on the dex card too.

---

## 6. Risk register

| Risk                                                           | Likelihood | Mitigation                                                                      |
| -------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------- |
| SmolVLM2 processor API differs across `transformers`versions | High       | De-risk on Fri Jun 5; pin a version once it works; Moondream2 as fallback namer |
| HBMCD not publicly downloadable                                | Medium     | Decide in Phase 0; CCSN is the drop-in fallback                                 |
| ZeroGPU device-placement errors                                | Resolved   | HF credits cover a standard GPU Space — skip ZeroGPU                           |
| Fine-tuning eats a day you don't have                          | Medium     | It's isolated to Sun + integration; MVP ships without it                        |
| Solo single-point failure                                      | —         | Commit a working Space early; there's always a fallback submission              |
| Demo/write-up rushed at the end                                | Medium     | Whole of Weekend 2 reserved for it; nothing else scheduled then                 |

---

## 7. Stretch (only if Weekend 2 is calm)

* **Open trace (Sharing is Caring):** share a run/trace on the Hub — nearly free.
* **Llama Champion:** run SmolVLM2 (or a GGUF VLM) through llama.cpp — real extra work; skip unless ahead.

---

## 8. Definition of done

* [ ] Gradio Space live, custom-styled, fully local (no cloud APIs)
* [ ] Fine-tuned genus model on the Hub **and** visibly used in the app
* [ ] 60–90s demo video
* [ ] Field Notes write-up published/linked
* [ ] Social post live
* [ ] Submitted before the Jun 15 deadline
