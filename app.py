"""
Nephodex — The Semantic Cloud Pokédex
Build Small Hackathon prototype (Chapter Two: An Adventure in Thousand Token Wood)

Pipeline (all local, no cloud APIs -> targets the "Off the Grid" badge):
  SlimSAM    -> isolate the cloud you click on
  OpenCV     -> alpha-matte the mask into a transparent .png "sticker"
  CLIP       -> embed the sticker, compare against your collection
  SmolVLM2   -> a small (~2.2B) local vision-language model names the shape

Total params well under the 32B cap: SlimSAM (~0.08B) + CLIP ViT-B/32 (~0.15B)
+ SmolVLM2-2.2B (~2.2B).

Notes
-----
* Untested in the author's sandbox (no GPU / no model access there). Meant to
  run on a Hugging Face Space with a GPU. On CPU it will work but be slow.
* Click the cloud you want before pressing Scan. If you don't click, it falls
  back to the centre of the frame.
"""

import os
import glob
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

from genus import classify_genus  # stage 4: fine-tuned genus (stub until weights land)

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


# Defensive shim for a historical gradio_client schema-parser crash on boolean
# `additionalProperties` ("argument of type 'bool' is not iterable"). Fixed in
# modern gradio, but harmless to keep; guarded so it no-ops if internals differ.
try:
    import gradio_client.utils as _gc_utils

    _orig_js2pt = _gc_utils._json_schema_to_python_type

    def _safe_js2pt(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig_js2pt(schema, defs)

    _gc_utils._json_schema_to_python_type = _safe_js2pt
except Exception:  # pragma: no cover
    pass


device = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if device == "cuda" else torch.float32

# -------------------------------------------------------------------------
# 1. Model loading (once, at import)
# -------------------------------------------------------------------------
# Tiny transformers-native SAM. (The original dhkim2810/MobileSAM repo stores a raw
# .pt checkpoint, not an HF-format model, so SamModel.from_pretrained can't read it.)
SAM_ID = "Zigeng/SlimSAM-uniform-77"
SAM_PROCESSOR_ID = "facebook/sam-vit-base"  # SAM processors are variant-agnostic
CLIP_ID = "openai/clip-vit-base-patch32"
VLM_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"  # swap for Moondream2 / Qwen2.5-VL-7B

print("Loading SlimSAM…")
sam_model = SamModel.from_pretrained(SAM_ID).to(device)
sam_processor = SamProcessor.from_pretrained(SAM_PROCESSOR_ID)

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
    genus, genus_conf = classify_genus(_flatten_on_white(sticker))

    if not collection:
        verdict = "🌱 **First cloud found!** Your Nephodex has begun."
    elif best_sim > 0.88:
        verdict = f"⚠️ **Twin of _{match_name}_** — {best_sim*100:.0f}% match. Already in your dex?"
    else:
        verdict = f"✨ **New specimen!** Closest match in your dex is only {best_sim*100:.0f}%."

    # Surface the fine-tuned genus when available (stub returns conf 0.0 -> hidden).
    genus_line = f"\n\n🔬 likely **{genus}** ({genus_conf*100:.0f}%)" if genus_conf > 0 else ""

    report = f"## ☁️ {nickname}\n\n*{description}*{genus_line}\n\n---\n{verdict}"
    pending = {
        "image": sticker,
        "nickname": nickname,
        "embedding": emb,
        "genus": genus,
        "genus_conf": genus_conf,
    }
    return sticker, report, pending, gr.update(interactive=True)


def _counter_html(n: int) -> str:
    """The 'Specimens' tally chip in the header."""
    return f'<div class="nx-counter"><div class="n">{n}</div><div class="l">Specimens</div></div>'


def save_to_album(pending, collection):
    if pending is None:
        return collection, gr.update(), gr.update(interactive=False), gr.update()
    collection = collection + [pending]
    gallery = [
        (
            it["image"],
            it["nickname"] + (f" · {it['genus']}" if it.get("genus_conf", 0) > 0 else ""),
        )
        for it in collection
    ]
    return collection, gallery, gr.update(interactive=False), _counter_html(len(collection))


def on_select(evt: gr.SelectData):
    return list(evt.index), f"📍 Pinned at {tuple(evt.index)} — press Scan."


def on_new_image(_):
    return None, "Click the cloud you want, then Scan."


# -------------------------------------------------------------------------
# 4. UI  (custom styling -> targets the "Off-Brand" badge)
# -------------------------------------------------------------------------
# Design tokens + layout ported from doc/nephodex_demo.html (the mockup).
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;0,9..144,700;1,9..144,500&family=Karla:wght@400;500;600;700&display=swap');

:root{
  --ink:#f4eddd; --muted:#cfc7b6;
  --sky1:#16233f; --sky2:#26375f; --sky3:#5b446f;
  --sun:#f6c66b; --ember:#e8884c; --gold:#e8a73c;
  --card:#f4eddd; --card-ink:#2a3147; --card-sub:#5d6478;
}

/* ---- the twilight canvas ---- */
.gradio-container{
  max-width:1080px !important; margin:0 auto !important;
  font-family:'Karla',system-ui,sans-serif !important; color:var(--ink) !important;
  background:
    radial-gradient(900px 520px at 82% -8%, #f6c66b44, transparent 60%),
    radial-gradient(760px 520px at 6% 4%, #b07bd633, transparent 60%),
    radial-gradient(600px 400px at 50% 120%, #e8884c22, transparent 60%),
    linear-gradient(178deg,var(--sky1) 0%,var(--sky2) 48%,var(--sky3) 100%) !important;
  background-attachment:fixed !important;
}
/* film grain */
.gradio-container::before{
  content:""; position:fixed; inset:0; pointer-events:none; opacity:.05; z-index:0;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}
/* strip default block chrome so the sky shows through */
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .gap{ background:transparent !important; border:none !important; box-shadow:none !important; }
.gradio-container label span{ color:var(--ink) !important; font-weight:600 !important; }
.gradio-container .prose{ color:var(--ink); }

/* ---- header ---- */
#nx-brand{ font-family:'Fraunces',serif; font-weight:700; font-size:clamp(2.4rem,5vw,3.4rem);
  line-height:.95; letter-spacing:-1px; color:var(--sun); text-shadow:0 3px 24px #0007; margin:0; }
#nx-brand span{ font-style:italic; color:var(--ink); }
#nx-sub{ margin-top:8px; color:var(--muted); max-width:46ch; font-size:1.02rem; }
.nx-counter{ background:#0e1830aa; border:1px solid #f6c66b44; border-radius:16px; padding:12px 18px;
  text-align:center; backdrop-filter:blur(6px); display:inline-block; }
.nx-counter .n{ font-family:'Fraunces',serif; font-size:2rem; color:var(--sun); line-height:1; }
.nx-counter .l{ font-size:.72rem; letter-spacing:.18em; text-transform:uppercase; color:var(--muted); }

/* ---- pill tabs ---- */
.gradio-container .tab-nav{ border:none !important; gap:8px; margin:18px 0; }
.gradio-container .tab-nav button{
  font-family:'Karla'; font-weight:700; font-size:.95rem; color:var(--muted) !important;
  background:#0e183055 !important; border:1px solid #ffffff14 !important;
  padding:10px 20px !important; border-radius:999px !important; }
.gradio-container .tab-nav button.selected{
  background:linear-gradient(135deg,var(--gold),var(--ember)) !important; color:#2a1c08 !important;
  border-color:transparent !important; }

/* ---- buttons ---- */
.gradio-container button.primary{
  background:linear-gradient(135deg,var(--gold),var(--ember)) !important; color:#2a1c08 !important;
  border:none !important; border-radius:999px !important; font-weight:700 !important;
  box-shadow:0 8px 22px #e8884c44 !important; }
.gradio-container button.secondary{
  background:#0e183066 !important; color:var(--ink) !important; border:1px solid #ffffff22 !important;
  border-radius:999px !important; font-weight:700 !important; }

/* ---- cream result card ---- */
#nx-result{
  background:var(--card) !important; color:var(--card-ink) !important; border-radius:22px !important;
  padding:22px !important; box-shadow:0 18px 50px #0a1228aa !important; border:1px solid #ffffff55 !important; }
#nx-result .prose, #nx-result p, #nx-result strong{ color:var(--card-ink) !important; }
#nx-result :is(h1,h2,h3){ font-family:'Fraunces',serif !important; color:var(--card-ink) !important; }
#nx-result em, #nx-result i{ color:var(--card-sub) !important; }

/* ---- checkerboard sticker stage ---- */
#nx-sticker, #nx-sticker .image-container{
  border-radius:16px !important;
  background-image:
    linear-gradient(45deg,#00000010 25%,transparent 25%,transparent 75%,#00000010 75%),
    linear-gradient(45deg,#00000010 25%,transparent 25%,transparent 75%,#00000010 75%) !important;
  background-size:16px 16px !important; background-position:0 0,8px 8px !important; }

/* ---- dex gallery cards ---- */
#nx-dex .grid-wrap{ background:transparent !important; }
#nx-dex .thumbnail-item{
  background:var(--card) !important; border-radius:18px !important;
  box-shadow:0 12px 30px #0a122888 !important; border:1px solid #ffffff55 !important; }
#nx-dex .caption{ font-family:'Fraunces',serif !important; color:var(--card-ink) !important; }

footer{ display:none !important; }
"""

HEADER_HTML = """
<h1 id="nx-brand">Nepho<span>dex</span></h1>
<p id="nx-sub">A field journal for cloud-gazers. Snap the sky, click the cloud you
mean, and a small local model isolates it, names the shape it sees, and files it in your dex.</p>
"""

# Sample skies shipped in assets/ -> one-click examples (great for judges with no
# photo handy). Degrades gracefully to an empty list if the folder is missing.
EXAMPLE_PHOTOS = sorted(
    glob.glob(os.path.join("assets", "*.jpeg"))
    + glob.glob(os.path.join("assets", "*.jpg"))
    + glob.glob(os.path.join("assets", "*.png"))
)

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="Nephodex") as app:
    collection_state = gr.State([])
    pending_card = gr.State(None)
    click_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=4):
            gr.HTML(HEADER_HTML)
        with gr.Column(scale=1, min_width=150):
            counter_view = gr.HTML(_counter_html(0))

    with gr.Tab("🔭 Capture"):
        with gr.Row():
            with gr.Column():
                input_view = gr.Image(
                    type="pil", label="Sky capture (webcam supported)", elem_id="nx-view"
                )
                click_status = gr.Markdown("Click the cloud you want, then Scan.")
                if EXAMPLE_PHOTOS:
                    gr.Examples(
                        examples=EXAMPLE_PHOTOS, inputs=input_view,
                        label="Tap a sample sky", examples_per_page=8,
                    )
                scan_btn = gr.Button("🔍 Isolate & name this cloud", variant="primary")
            with gr.Column():
                crop_view = gr.Image(
                    type="pil", label="Specimen sticker (.png)", elem_id="nx-sticker"
                )
                details_view = gr.Markdown(
                    "Your reading will appear here.", elem_id="nx-result"
                )
                add_btn = gr.Button(
                    "✨ File in my Nephodex", variant="primary", interactive=False
                )

    with gr.Tab("📔 My Nephodex"):
        album_gallery = gr.Gallery(
            label="Collected specimens", columns=4,
            object_fit="contain", height=600, elem_id="nx-dex",
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
        outputs=[collection_state, album_gallery, add_btn, counter_view],
    )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)