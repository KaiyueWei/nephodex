"""
Nephodex — The Semantic Cloud Pokédex
Build Small Hackathon prototype (Chapter Two: An Adventure in Thousand Token Wood)

Pipeline (all local, no cloud APIs -> targets the "Off the Grid" badge):
  MobileSAM  -> isolate the cloud you click on
  OpenCV     -> alpha-matte the mask into a transparent .png "sticker"
  CLIP       -> embed the sticker, compare against your collection
  SmolVLM2   -> a small (~2.2B) local vision-language model names the shape

Total params well under the 32B cap: MobileSAM (tiny) + CLIP ViT-B/32 (~0.15B)
+ SmolVLM2-2.2B (~2.2B).

Notes
-----
* Untested in the author's sandbox (no GPU / no model access there). Meant to
  run on a Hugging Face Space with a GPU. On CPU it will work but be slow.
* Click the cloud you want before pressing Scan. If you don't click, it falls
  back to the centre of the frame.
"""

import os
import json
import re

import cv2
import numpy as np
import torch
from PIL import Image
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity

from transformers import (
    SamModel,
    SamProcessor,
    CLIPModel,
    CLIPProcessor,
    AutoProcessor,
    AutoModelForImageTextToText,
)

# Optional ZeroGPU support. On a normal GPU/CPU Space `spaces` is absent and the
# decorator becomes a no-op.
try:
    import spaces

    GPU = spaces.GPU
except Exception:  # pragma: no cover
    def GPU(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func


device = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if device == "cuda" else torch.float32

# -------------------------------------------------------------------------
# 1. Model loading (once, at import)
# -------------------------------------------------------------------------
SAM_ID = "dhkim2810/MobileSAM"
CLIP_ID = "openai/clip-vit-base-patch32"
VLM_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"  # swap for Moondream2 / Qwen2.5-VL-7B

print("Loading MobileSAM…")
sam_model = SamModel.from_pretrained(SAM_ID).to(device)
sam_processor = SamProcessor.from_pretrained(SAM_ID)

print("Loading CLIP…")
clip_model = CLIPModel.from_pretrained(CLIP_ID).to(device)
clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)

print("Loading SmolVLM2…")
vlm_processor = AutoProcessor.from_pretrained(VLM_ID)
vlm_model = AutoModelForImageTextToText.from_pretrained(
    VLM_ID, torch_dtype=DTYPE
).to(device)


# -------------------------------------------------------------------------
# 2. Core processing
# -------------------------------------------------------------------------
def _flatten_on_white(pil_rgba):
    """Composite an RGBA sticker onto a white background -> RGB (for CLIP/VLM)."""
    rgb = Image.new("RGB", pil_rgba.size, (255, 255, 255))
    if pil_rgba.mode == "RGBA":
        rgb.paste(pil_rgba, mask=pil_rgba.split()[3])
    else:
        rgb.paste(pil_rgba.convert("RGB"))
    return rgb


@GPU
def segment_cloud(pil_img, click_xy):
    """
    Prompt MobileSAM with a SINGLE point (where the user clicked, or the image
    centre as a fallback) and keep the highest-confidence mask. Returns a
    transparent RGBA crop of that cloud.

    This is the corrected segmentation: a single point prompt yields three
    candidate masks and we pick the one SAM is most confident about, instead of
    the original code's grid-of-points (which SAM treats as ONE blob, not
    several discoverable objects).
    """
    image_np = np.array(pil_img.convert("RGB"))
    h, w, _ = image_np.shape

    if click_xy is None:
        point = [w // 2, h // 2]
    else:
        x = max(0, min(int(click_xy[0]), w - 1))
        y = max(0, min(int(click_xy[1]), h - 1))
        point = [x, y]

    input_points = [[point]]  # (batch=1, n_points=1, 2)
    inputs = sam_processor(
        pil_img, input_points=input_points, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0][0]  # -> [num_candidate_masks, H, W]

    if masks.shape[0] == 0:
        return pil_img.convert("RGBA")

    # Pick the mask SAM scored highest (iou_scores: [1, 1, n_masks]).
    best_idx = int(outputs.iou_scores[0, 0].argmax().item())
    best_mask = masks[best_idx].numpy().astype(np.uint8) * 255

    # OpenCV: mask -> alpha channel, then crop tight to the cloud.
    ys, xs = np.where(best_mask > 0)
    if len(xs) == 0:
        return pil_img.convert("RGBA")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()

    r, g, b = cv2.split(image_np)
    rgba = cv2.merge([r, g, b, best_mask])
    cropped = rgba[y0 : y1 + 1, x0 : x1 + 1]
    return Image.fromarray(cropped, "RGBA")


@GPU
def get_clip_embedding(pil_img):
    rgb = _flatten_on_white(pil_img)
    inputs = clip_processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
    emb = feats.cpu().numpy().flatten()
    return emb / (np.linalg.norm(emb) + 1e-8)


@GPU
def generate_cloud_lore(pil_img):
    """Ask SmolVLM2 what the cloud looks like. We request a simple
    'Name:/Description:' format because small VLMs are far more reliable at that
    than at strict JSON."""
    rgb = _flatten_on_white(pil_img)
    prompt = (
        "You are a whimsical cloud-gazing expert. Ignore meteorology. "
        "Looking at this cloud shape, what animal, object, or character do you "
        "see? Reply in exactly this format and nothing else:\n"
        "Name: <a fun 2-4 word name>\n"
        "Description: <one poetic sentence>"
    )
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    ]
    chat = vlm_processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = vlm_processor(text=chat, images=[rgb], return_tensors="pt").to(
        device, dtype=DTYPE
    )
    with torch.no_grad():
        gen = vlm_model.generate(**inputs, max_new_tokens=96, do_sample=False)
    text = vlm_processor.batch_decode(
        gen[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )[0].strip()

    name = re.search(r"Name:\s*(.+)", text)
    desc = re.search(r"Description:\s*(.+)", text)
    nickname = name.group(1).strip() if name else "Mystery Cloud"
    description = desc.group(1).strip() if desc else "A quiet drift across the blue."
    # Trim any trailing model chatter.
    nickname = nickname.split("\n")[0][:60]
    description = description.split("\n")[0][:200]
    return nickname, description


# -------------------------------------------------------------------------
# 3. Pipeline logic
# -------------------------------------------------------------------------
def process_pipeline(uploaded_image, click_xy, collection):
    if uploaded_image is None:
        return None, "📷 Upload or snap a photo of the sky first.", None, gr.update(interactive=False)

    sticker = segment_cloud(uploaded_image, click_xy)
    emb = get_clip_embedding(sticker)

    best_sim, match_name = 0.0, ""
    for item in collection:
        sim = float(cosine_similarity([emb], [item["embedding"]])[0][0])
        if sim > best_sim:
            best_sim, match_name = sim, item["nickname"]

    nickname, description = generate_cloud_lore(sticker)

    if not collection:
        verdict = "🌱 **First cloud found!** Your Nephodex has begun."
    elif best_sim > 0.88:
        verdict = f"⚠️ **Twin of _{match_name}_** — {best_sim*100:.0f}% match. Already in your dex?"
    else:
        verdict = f"✨ **New specimen!** Closest match in your dex is only {best_sim*100:.0f}%."

    report = f"## ☁️ {nickname}\n\n*{description}*\n\n---\n{verdict}"
    pending = {"image": sticker, "nickname": nickname, "embedding": emb}
    return sticker, report, pending, gr.update(interactive=True)


def save_to_album(pending, collection):
    if pending is None:
        return collection, gr.update(), gr.update(interactive=False)
    collection = collection + [pending]
    gallery = [(it["image"], it["nickname"]) for it in collection]
    return collection, gallery, gr.update(interactive=False)


def on_select(evt: gr.SelectData):
    return list(evt.index), f"📍 Pinned at {tuple(evt.index)} — press Scan."


def on_new_image(_):
    return None, "Click the cloud you want, then Scan."


# -------------------------------------------------------------------------
# 4. UI  (custom styling -> targets the "Off-Brand" badge)
# -------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,700;1,9..144,500&family=Karla:wght@400;600&display=swap');

.gradio-container {
    background:
        radial-gradient(1200px 600px at 75% -10%, #f6c66b33, transparent),
        radial-gradient(900px 500px at 10% 0%, #c98bd633, transparent),
        linear-gradient(180deg, #1b2a4a 0%, #243a63 45%, #3a4f7a 100%) !important;
    font-family: 'Karla', sans-serif !important;
    color: #f3ecdd !important;
}
#sky-title {
    font-family: 'Fraunces', serif !important;
    font-weight: 700; font-size: 2.7rem; line-height: 1.05;
    color: #f7e7c4 !important; letter-spacing: -0.5px; margin-bottom: 0;
    text-shadow: 0 2px 18px #0006;
}
#sky-sub { color: #d9d2c4 !important; font-size: 1.02rem; max-width: 60ch; }
.block, .gr-box, .gr-panel {
    background: #f3ecddee !important;
    border: 1px solid #f7e7c455 !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 30px #0a142e55 !important;
}
.gr-button-primary {
    background: linear-gradient(135deg, #e8a73c, #d98032) !important;
    border: none !important; color: #2a1c08 !important; font-weight: 600 !important;
    border-radius: 999px !important;
}
.gr-button-secondary {
    background: #243a63 !important; color: #f7e7c4 !important;
    border: 1px solid #f7e7c455 !important; border-radius: 999px !important;
}
label span { color: #2a3147 !important; font-weight: 600 !important; }
.tabitem { border-radius: 18px !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="Nephodex") as app:
    collection_state = gr.State([])
    pending_card = gr.State(None)
    click_state = gr.State(None)

    gr.Markdown("# ☁️ Nephodex", elem_id="sky-title")
    gr.Markdown(
        "_A field journal for cloud-gazers._ Snap the sky, **click the cloud "
        "you mean**, and a small local model isolates it, names the shape it "
        "sees, and files it in your dex.",
        elem_id="sky-sub",
    )

    with gr.Tab("🔭 Capture"):
        with gr.Row():
            with gr.Column():
                input_view = gr.Image(type="pil", label="Sky capture (webcam supported)")
                click_status = gr.Markdown("Click the cloud you want, then Scan.")
                scan_btn = gr.Button("🔍 Isolate & name this cloud", variant="primary")
            with gr.Column():
                crop_view = gr.Image(type="pil", label="Specimen sticker (.png)")
                details_view = gr.Markdown("Your reading will appear here.")
                add_btn = gr.Button("✨ File in my Nephodex", variant="secondary", interactive=False)

    with gr.Tab("📔 My Nephodex"):
        album_gallery = gr.Gallery(
            label="Collected specimens", columns=[4], rows=[2],
            object_fit="contain", height="600px",
        )

    # Capture click coordinates for the SAM point prompt.
    input_view.select(on_select, outputs=[click_state, click_status])
    input_view.change(on_new_image, inputs=[input_view], outputs=[click_state, click_status])

    scan_btn.click(
        process_pipeline,
        inputs=[input_view, click_state, collection_state],
        outputs=[crop_view, details_view, pending_card, add_btn],
    )
    add_btn.click(
        save_to_album,
        inputs=[pending_card, collection_state],
        outputs=[collection_state, album_gallery, add_btn],
    )


if __name__ == "__main__":
    app.launch()