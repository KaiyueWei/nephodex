---
title: Nephodex
emoji: ☁️
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 5.38.2
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
---

# ☁️ Nephodex — The Semantic Cloud Pokédex


A field journal for cloud-gazers. Snap the sky, click the cloud you mean, and a
small **local** model isolates it, names the shape it sees, and files it in your
dex. Built for the [Build Small Hackathon](https://huggingface.co/build-small-hackathon)
(Chapter Two:  *An Adventure in Thousand Token Wood* ).


## How it works


| Stage                                  | Model                                 | Params |
| ---------------------------------------- | --------------------------------------- | -------- |
| Isolate the clicked cloud              | SlimSAM (`Zigeng/SlimSAM-uniform-77`) | ~0.08B |
| Alpha-matte into a transparent sticker | OpenCV                                | —     |
| Embed + compare to your collection     | CLIP ViT-B/32                         | ~0.15B |
| Name the shape                         | SmolVLM2-2.2B-Instruct                | ~2.2B  |


Everything runs locally — **no cloud APIs** — which keeps it well under the 32B
cap and targets the *Off the Grid* badge.


## Constraint check


* **≤ 32B params:** ✅ ~2.4B total across all three models.
* **Gradio app on a Space:** ✅
* **Demo video + social post:** still to record before submission.


## Running


This needs a **GPU Space** to be snappy (it works on CPU but slowly). On a normal
GPU Space it runs as-is. On ZeroGPU the `@spaces.GPU` decorators are already in
place; if you hit a device-placement error, the usual fix is to move the model
`.to("cuda")` inside the decorated functions rather than at import.


```bash
pip install -r requirements.txt
python app.py
```


## Swapping the namer


`VLM_ID` in `app.py` controls the naming model. Alternatives under the cap:


* `vikhyatk/moondream2` — ~2B, purpose-built for "what is this" (uses a
  `trust_remote_code` API, so the call site changes slightly).
* `Qwen/Qwen2.5-VL-7B-Instruct` — more capable, heavier.


## Ideas to push further


* **Off-Brand badge:** the UI is already custom-styled; lean harder into the
  field-journal / dex-card look.
* **Field Notes badge:** write up what you learned.
* Let users name regions by tapping multiple points for multi-cloud capture.
