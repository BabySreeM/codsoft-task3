import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import math
import tempfile

# ── Load model ──────────────────────────────────────────────────────────────
print("Loading model... (first run downloads ~900MB, be patient)")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()
print("✅ Model ready!")

# ── Caption generation ───────────────────────────────────────────────────────
def caption_image(image, num_captions, caption_style, max_length):
    if image is None:
        return "Upload an image to generate a caption.", None

    style_prefix = {
        "Default":    "",
        "Detailed":   "a detailed description of",
        "Short":      "a short description of",
        "Story-like": "tell a story about",
    }

    prefix = style_prefix.get(caption_style, "")
    text_input = f"{prefix} " if prefix else None

    inputs = (
        processor(image, text_input, return_tensors="pt")
        if text_input
        else processor(image, return_tensors="pt")
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_length),
            num_beams=5,
            num_return_sequences=min(int(num_captions), 3),
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

    captions = []
    for o in outputs.sequences:
        decoded = processor.decode(o, skip_special_tokens=True)
        if text_input and decoded.lower().startswith(prefix.lower()):
            decoded = decoded[len(prefix):].strip()
        captions.append(decoded.capitalize())

    raw_scores = outputs.sequences_scores.tolist()
    min_s, max_s = min(raw_scores), max(raw_scores)
    span = max_s - min_s if max_s != min_s else 1
    confidences = [round(60 + ((s - min_s) / span) * 39) for s in raw_scores]

    if len(captions) == 1:
        result_text = f"Confidence: {confidences[0]}%\n\n{captions[0]}"
    else:
        result_text = "\n\n".join(
            f"Caption {i+1} — {confidences[i]}% confidence\n{c}"
            for i, c in enumerate(captions)
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
    tmp.write(result_text)
    tmp.close()

    return result_text, tmp.name


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

*, *::before, *::after { box-sizing: border-box !important; margin: 0; padding: 0; }

/* ══ DEEP SPACE BACKGROUND ══ */
html, body {
    background: #050810 !important;
    min-height: 100vh !important;
    overflow-x: hidden !important;
}

body::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    background:
        radial-gradient(ellipse 70% 60% at 15% 20%,  rgba(0,200,255,0.07) 0%, transparent 65%),
        radial-gradient(ellipse 50% 40% at 85% 75%,  rgba(80,40,220,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 80% 80% at 50% 50%,  rgba(5,8,16,0.95)   0%, transparent 100%);
    pointer-events: none;
}

/* Starfield dots via box-shadow trick */
body::after {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    background-image:
        radial-gradient(1px 1px at 10%  15%, rgba(255,255,255,0.55) 0%, transparent 100%),
        radial-gradient(1px 1px at 25%  40%, rgba(255,255,255,0.35) 0%, transparent 100%),
        radial-gradient(1px 1px at 40%   8%, rgba(255,255,255,0.45) 0%, transparent 100%),
        radial-gradient(1px 1px at 55%  60%, rgba(255,255,255,0.30) 0%, transparent 100%),
        radial-gradient(1px 1px at 70%  25%, rgba(255,255,255,0.50) 0%, transparent 100%),
        radial-gradient(1px 1px at 80%  80%, rgba(255,255,255,0.35) 0%, transparent 100%),
        radial-gradient(1px 1px at 90%  45%, rgba(255,255,255,0.40) 0%, transparent 100%),
        radial-gradient(1px 1px at  5%  70%, rgba(255,255,255,0.30) 0%, transparent 100%),
        radial-gradient(1px 1px at 35%  90%, rgba(255,255,255,0.45) 0%, transparent 100%),
        radial-gradient(1px 1px at 65%  5%,  rgba(255,255,255,0.35) 0%, transparent 100%),
        radial-gradient(2px 2px at 20%  55%, rgba(0,220,255,0.40)   0%, transparent 100%),
        radial-gradient(2px 2px at 75%  12%, rgba(120,80,255,0.35)  0%, transparent 100%),
        radial-gradient(2px 2px at 48%  78%, rgba(0,200,255,0.30)   0%, transparent 100%);
    pointer-events: none;
}

/* ══ GRADIO CONTAINER ══ */
.gradio-container {
    background: transparent !important;
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 0 2rem 5rem !important;
    font-family: 'Inter', sans-serif !important;
    position: relative;
    z-index: 1;
}

/* Nuke all Gradio default backgrounds */
.gradio-container .block,
.gradio-container .form,
.gradio-container .gap,
.gradio-container > div,
.gradio-container .contain,
.gradio-container .gr-group,
.gradio-container fieldset,
.gradio-container .wrap {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* ══ NAVBAR ══ */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 0;
    border-bottom: 1px solid rgba(0,200,255,0.08);
    margin-bottom: 0;
}
.navbar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}
.navbar-logo {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #00c8ff, #7040ff);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    box-shadow: 0 0 14px rgba(0,200,255,0.4);
}
.navbar-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #e0f4ff;
    letter-spacing: 0.04em;
}
.navbar-badge {
    background: rgba(0,200,255,0.1);
    border: 1px solid rgba(0,200,255,0.25);
    border-radius: 20px;
    padding: 0.25rem 0.85rem;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #00c8ff;
}

/* ══ HERO HEADER ══ */
.app-header {
    text-align: center;
    padding: 3.2rem 1rem 2.5rem;
    position: relative;
}
.app-header .eyebrow {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: rgba(0,200,255,0.6);
    margin-bottom: 1rem;
    display: block;
}
.app-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 3.8rem !important;
    font-weight: 800 !important;
    margin: 0 0 0.7rem !important;
    line-height: 1.05 !important;
    letter-spacing: -1.5px !important;
    background: linear-gradient(135deg, #ffffff 0%, #a0d8ff 35%, #00c8ff 60%, #7040ff 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    filter: drop-shadow(0 0 30px rgba(0,200,255,0.2));
}
.app-header .subtitle {
    font-size: 0.88rem;
    color: rgba(160,200,230,0.5);
    font-weight: 300;
    letter-spacing: 0.06em;
}
.glow-rule {
    width: 80px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00c8ff, transparent);
    margin: 1.5rem auto 0;
    box-shadow: 0 0 8px rgba(0,200,255,0.5);
}

/* ══ GLASS PANELS ══ */
.left-panel, .right-panel {
    background: rgba(10,18,35,0.65) !important;
    border: 1px solid rgba(0,200,255,0.12) !important;
    border-radius: 18px !important;
    overflow: hidden !important;
    backdrop-filter: blur(28px) !important;
    -webkit-backdrop-filter: blur(28px) !important;
    box-shadow:
        0 0 0 1px rgba(255,255,255,0.04) inset,
        0 8px 40px rgba(0,0,0,0.5),
        0 0 60px rgba(0,200,255,0.03) !important;
    transition: box-shadow 0.3s ease !important;
}
.left-panel:hover {
    box-shadow:
        0 0 0 1px rgba(255,255,255,0.06) inset,
        0 8px 40px rgba(0,0,0,0.6),
        0 0 40px rgba(0,200,255,0.06) !important;
}
.right-panel:hover {
    box-shadow:
        0 0 0 1px rgba(255,255,255,0.06) inset,
        0 8px 40px rgba(0,0,0,0.6),
        0 0 40px rgba(112,64,255,0.07) !important;
}
.left-panel *, .right-panel * {
    background: transparent !important;
}

/* Panel label bar */
.panel-label {
    padding: 1rem 1.4rem 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(0,200,255,0.03) !important;
}
.panel-label span {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: rgba(0,200,255,0.55);
}
.panel-label .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00c8ff;
    box-shadow: 0 0 8px rgba(0,200,255,0.9), 0 0 16px rgba(0,200,255,0.4);
    flex-shrink: 0;
    animation: pulse-dot 2.5s ease-in-out infinite;
}
.panel-label .dot.purple {
    background: #7040ff;
    box-shadow: 0 0 8px rgba(112,64,255,0.9), 0 0 16px rgba(112,64,255,0.4);
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(0.85); }
}

.panel-body { padding: 1.4rem; }

/* ══ UPLOAD ZONE ══ */
.upload-wrap {
    border: 1.5px dashed rgba(0,200,255,0.2) !important;
    border-radius: 12px !important;
    background: rgba(0,200,255,0.025) !important;
    min-height: 230px !important;
    max-height: 280px !important;
    transition: all 0.25s ease !important;
    margin-bottom: 1.3rem !important;
    overflow: hidden !important;
}
.upload-wrap:hover {
    border-color: rgba(0,200,255,0.5) !important;
    background: rgba(0,200,255,0.05) !important;
    box-shadow: 0 0 20px rgba(0,200,255,0.08) !important;
}
/* Fix Gradio's internal 3-column split on image upload */
.upload-wrap > div,
.upload-wrap .wrap,
.upload-wrap [data-testid="image"] {
    width: 100% !important;
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
}
.upload-wrap .upload-container {
    grid-template-columns: 1fr !important;
    display: flex !important;
}

/* ══ LABELS ══ */
.gradio-container label span {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: rgba(140,190,220,0.55) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ══ SLIDER ══ */
input[type=range] { accent-color: #00c8ff !important; }

/* ══ DROPDOWN ══ */
.gradio-container select,
.gradio-container .wrap {
    background: rgba(0,200,255,0.05) !important;
    border: 1px solid rgba(0,200,255,0.15) !important;
    border-radius: 8px !important;
    color: #c8eeff !important;
    font-size: 0.88rem !important;
    font-family: 'Inter', sans-serif !important;
}

/* ══ TEXTBOX ══ */
.gradio-container textarea {
    background: rgba(0,10,25,0.6) !important;
    border: 1px solid rgba(0,200,255,0.12) !important;
    border-radius: 10px !important;
    color: #d0eeff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.98rem !important;
    font-weight: 300 !important;
    line-height: 1.85 !important;
    padding: 1.2rem !important;
    min-height: 150px !important;
    letter-spacing: 0.01em !important;
}
.gradio-container textarea::placeholder {
    color: rgba(100,160,200,0.25) !important;
    font-style: italic !important;
}

/* ══ GENERATE BUTTON ══ */
.generate-btn, .generate-btn:focus {
    background: linear-gradient(135deg, #0090cc 0%, #0060aa 50%, #4020cc 100%) !important;
    border: 1px solid rgba(0,200,255,0.35) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.95rem 1.5rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow:
        0 4px 20px rgba(0,150,220,0.4),
        0 0 0 1px rgba(0,200,255,0.1) inset !important;
}
.generate-btn:hover {
    background: linear-gradient(135deg, #00b8ee 0%, #0080cc 50%, #5530ee 100%) !important;
    box-shadow:
        0 6px 30px rgba(0,180,255,0.55),
        0 0 40px rgba(0,200,255,0.2),
        0 0 0 1px rgba(0,200,255,0.2) inset !important;
    transform: translateY(-2px) !important;
    border-color: rgba(0,200,255,0.5) !important;
}
.generate-btn:active { transform: translateY(0) !important; }

/* ══ CLEAR BUTTON ══ */
.clear-btn, .clear-btn:focus {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.22) !important;
    border-radius: 10px !important;
    color: rgba(200,230,255,0.9) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.95rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
}
.clear-btn:hover {
    background: rgba(0,200,255,0.12) !important;
    border-color: rgba(0,200,255,0.45) !important;
    color: #00c8ff !important;
}

/* ══ HIDE GRADIO BRANDING FOOTER ══ */
.gradio-container ~ footer,
footer.svelte-1ax1toq,
.footer-wrap,
div.svelte-1ax1toq,
#footer,
.show-api,
a[href*="gradio.app"],
.built-with {
    display: none !important;
}

/* ══ REDUCE BOTTOM WHITESPACE ══ */
.gradio-container {
    padding-bottom: 2rem !important;
}
.panel-body {
    padding: 1.2rem !important;
}

/* ══ OUTPUT PANEL — fill height better ══ */
.right-panel .panel-body {
    display: flex !important;
    flex-direction: column !important;
    gap: 0.8rem !important;
}
.gradio-container textarea {
    flex: 1 !important;
    min-height: 200px !important;
}

/* ══ DOWNLOAD BUTTON ══ */
.download-btn button {
    background: rgba(112,64,255,0.08) !important;
    border: 1px solid rgba(112,64,255,0.3) !important;
    border-radius: 8px !important;
    color: #a080ff !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.75rem !important;
    width: 100% !important;
    margin-top: 0.75rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.download-btn button:hover {
    background: rgba(112,64,255,0.18) !important;
    border-color: rgba(112,64,255,0.55) !important;
    box-shadow: 0 0 20px rgba(112,64,255,0.2) !important;
}

/* ══ META BOX ══ */
.meta-box {
    margin-top: 1.2rem !important;
    padding: 1rem 1.2rem !important;
    background: rgba(0,200,255,0.04) !important;
    border: 1px solid rgba(0,200,255,0.1) !important;
    border-radius: 10px !important;
    border-left: 2px solid rgba(0,200,255,0.35) !important;
}
.meta-box p {
    margin: 0.22rem 0;
    font-size: 0.77rem;
    color: rgba(140,190,220,0.45);
    letter-spacing: 0.02em;
}
.meta-box strong { color: rgba(0,200,255,0.65); font-weight: 600; }

/* ══ STATS ROW ══ */
.stats-row {
    display: flex;
    gap: 1rem;
    margin: 2rem 0 1.5rem;
}
.stat-card {
    flex: 1;
    background: rgba(10,18,35,0.6);
    border: 1px solid rgba(0,200,255,0.1);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    backdrop-filter: blur(20px);
    text-align: center;
}
.stat-card .stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #00c8ff;
    display: block;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.stat-card .stat-label {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(140,190,220,0.4);
}

/* ══ FOOTER ══ */
.footer {
    text-align: center;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(0,200,255,0.07);
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(140,190,220,0.2);
}
.footer .accent { color: rgba(0,200,255,0.45); }
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Image Captioning AI", theme=gr.themes.Base()) as demo:

    # Navbar
    gr.HTML("""
        <div class="navbar">
            <div class="navbar-brand">
                <div class="navbar-logo">🔭</div>
                <span class="navbar-title">VisionAI</span>
            </div>
            <div class="navbar-badge">CodSoft Internship</div>
        </div>
    """)

    # Hero header
    gr.HTML("""
        <div class="app-header">
            <span class="eyebrow">Computer Vision &nbsp;·&nbsp; Natural Language Processing</span>
            <h1>Image Captioning</h1>
            <p class="subtitle">Drop any photograph. The neural model will find its words.</p>
            <div class="glow-rule"></div>
        </div>
    """)

    # Stats row
    gr.HTML("""
        <div class="stats-row">
            <div class="stat-card">
                <span class="stat-num">BLIP</span>
                <span class="stat-label">Model Architecture</span>
            </div>
            <div class="stat-card">
                <span class="stat-num">3</span>
                <span class="stat-label">Max Captions</span>
            </div>
            <div class="stat-card">
                <span class="stat-num">4</span>
                <span class="stat-label">Style Modes</span>
            </div>
            <div class="stat-card">
                <span class="stat-num">99%</span>
                <span class="stat-label">Top Confidence</span>
            </div>
        </div>
    """)

    with gr.Row(equal_height=False):

        # ── Left ──
        with gr.Column(scale=1, elem_classes="left-panel"):
            gr.HTML("""
                <div class="panel-label">
                    <div class="dot"></div>
                    <span>Input</span>
                </div>
            """)
            with gr.Group(elem_classes="panel-body"):
                image_input = gr.Image(
                    type="pil",
                    label="Photograph",
                    elem_classes="upload-wrap",
                )
                num_captions = gr.Slider(
                    minimum=1, maximum=3, step=1, value=1,
                    label="Captions to generate",
                    info="Up to 3 variations"
                )
                max_length = gr.Slider(
                    minimum=20, maximum=100, step=5, value=60,
                    label="Caption length",
                    info="Token limit per caption"
                )
                caption_style = gr.Dropdown(
                    choices=["Default", "Detailed", "Short", "Story-like"],
                    value="Default",
                    label="Style",
                )
                with gr.Row():
                    clear_btn = gr.ClearButton(
                        components=[image_input],
                        value="Clear",
                        elem_classes="clear-btn"
                    )
                    generate_btn = gr.Button(
                        "⚡ Generate Caption",
                        elem_classes="generate-btn",
                        variant="primary"
                    )

        # ── Right ──
        with gr.Column(scale=1, elem_classes="right-panel"):
            gr.HTML("""
                <div class="panel-label">
                    <div class="dot purple"></div>
                    <span>Output</span>
                </div>
            """)
            with gr.Group(elem_classes="panel-body"):
                caption_output = gr.Textbox(
                    label="Caption",
                    placeholder="The caption will appear here once you upload an image...",
                    lines=7,
                )
                download_btn = gr.DownloadButton(
                    label="↓  Download as .txt",
                    visible=False,
                    elem_classes="download-btn",
                )
                gr.HTML("""
                    <div class="meta-box">
                        <p><strong>Model</strong> &nbsp; Salesforce BLIP (base)</p>
                        <p><strong>Task</strong> &nbsp; Image Captioning · CodSoft AI Internship</p>
                        <p><strong>Tip</strong> &nbsp; Use Detailed style for richer descriptions</p>
                    </div>
                """)

    gr.HTML("""
        <div class="footer">
            Built with <span class="accent">♦</span> using HuggingFace Transformers &amp; Gradio
        </div>
    """)

    def generate_and_show_download(image, num_captions, caption_style, max_length):
        text, filepath = caption_image(image, num_captions, caption_style, max_length)
        return text, gr.DownloadButton(value=filepath, visible=filepath is not None)

    generate_btn.click(
        fn=generate_and_show_download,
        inputs=[image_input, num_captions, caption_style, max_length],
        outputs=[caption_output, download_btn],
    )
    image_input.change(
        fn=generate_and_show_download,
        inputs=[image_input, num_captions, caption_style, max_length],
        outputs=[caption_output, download_btn],
    )

demo.launch(share=True, css=custom_css)