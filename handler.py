# ---------------------------------------------
# Startup log (VERY IMPORTANT)
# ---------------------------------------------
print("🚀 Handler starting up...")

import base64
import cv2
import numpy as np
import torch
import runpod

from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------
# GPU CHECK
# ---------------------------------------------
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ GPU NOT AVAILABLE – running on CPU")

# ---------------------------------------------
# LOAD OCR (CPU ONLY – STABLE)
# ---------------------------------------------
print("🔤 Initializing PaddleOCR (CPU)...")

ocr = PaddleOCR(
    lang="ru",
    use_angle_cls=False,   # ❗ avoid PaddleX
    det=True,
    rec=True,
    show_log=False
)

print("✅ PaddleOCR ready")

# ---------------------------------------------
# LOAD LLM (GPU)
# ---------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

print("🤖 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

print("🤖 Loading model on GPU...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cuda",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

model.eval()
print("✅ Qwen model loaded")

# ---------------------------------------------
# UTILS
# ---------------------------------------------
def pdf_to_images(pdf_bytes):
    pages = convert_from_bytes(pdf_bytes, dpi=300)
    images = []
    for page in pages:
        img = np.array(page)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
    return images


def ocr_images(images):
    texts = []
    for img in images:
        result = ocr.predict(img)
        for page in result:
            texts.extend(page["rec_texts"])
    return texts


def chunk_text(text, max_chars=1200):
    chunks, current = [], ""
    for line in text.split("\n"):
        if len(current) + len(line) < max_chars:
            current += line + "\n"
        else:
            chunks.append(current)
            current = line + "\n"
    if current:
        chunks.append(current)
    return chunks


def translate_ru_to_en(text):
    chunks = chunk_text(text)
    outputs = []

    for chunk in chunks:
        prompt = f"""
Translate the following Russian text to English.

Russian:
{chunk}

English:
"""
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.2,
                do_sample=False
            )

        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        outputs.append(decoded.split("English:")[-1].strip())

    return "\n".join(outputs)


def summarize_text(text):
    chunks = chunk_text(text, max_chars=2000)
    partials = []

    for chunk in chunks:
        prompt = f"""
Summarize the following text.

Text:
{chunk}

Summary:
"""
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=False
            )

        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        partials.append(decoded.split("Summary:")[-1].strip())

    final_prompt = f"""
Create a concise overall summary from the following summaries.

Summaries:
{chr(10).join(partials)}

Final Summary:
"""
    inputs = tokenizer(final_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.3,
            do_sample=False
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return decoded.split("Final Summary:")[-1].strip()

# ---------------------------------------------
# RUNPOD HANDLER
# ---------------------------------------------
def handler(event):
    print("📥 Event received")

    # Warmup
    if event.get("input", {}).get("warmup"):
        return {"status": "warm"}

    pdf_base64 = event.get("input", {}).get("pdf_base64")
    if not pdf_base64:
        return {"error": "No PDF provided"}

    # Decode PDF
    pdf_bytes = base64.b64decode(pdf_base64)

    # PDF → Images
    images = pdf_to_images(pdf_bytes)

    # OCR
    ru_texts = ocr_images(images)
    ru_text = "\n".join(ru_texts)

    if not ru_text.strip():
        return {
            "text_ru": "",
            "text_en": "",
            "summary": ""
        }

    # Translate
    en_text = translate_ru_to_en(ru_text)

    # Summarize
    summary = summarize_text(en_text)

    return {
        "text_ru": ru_text,
        "text_en": en_text,
        "summary": summary
    }

# ---------------------------------------------
# REQUIRED ENTRYPOINT (MUST BE LAST LINE)
# ---------------------------------------------
runpod.serverless.start({"handler": handler})
