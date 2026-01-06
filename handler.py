# =========================================================
# RunPod Serverless Handler
# PaddleOCR 2.7 + Qwen 14B (80GB GPU)
# =========================================================

print("🚀 Handler file imported")

import base64
import cv2
import numpy as np
import torch
import runpod

from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================================================
# GLOBALS (LAZY LOADED)
# =========================================================
ocr = None
tokenizer = None
model = None

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

# =========================================================
# LAZY LOADERS
# =========================================================
def load_ocr():
    global ocr
    if ocr is None:
        print("🔤 Loading PaddleOCR (CPU, v2.7)...")
        ocr = PaddleOCR(
            lang="ru",
            use_angle_cls=False,
            det=True,
            rec=True
        )
        print("✅ PaddleOCR loaded")
    return ocr


def load_llm():
    global tokenizer, model
    if model is None:
        print("🤖 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        print("🤖 Loading model on GPU (FP16)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model.eval()
        print("✅ Qwen model loaded")
    return tokenizer, model

# =========================================================
# UTILS
# =========================================================
def pdf_to_images(pdf_bytes):
    pages = convert_from_bytes(pdf_bytes, dpi=300)
    images = []
    for page in pages:
        img = np.array(page)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
    return images


def ocr_images(images):
    engine = load_ocr()
    texts = []
    for img in images:
        result = engine.ocr(img, cls=False)
        for line in result:
            texts.append(line[1][0])
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
    tokenizer, model = load_llm()
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
                do_sample=False,
                temperature=0.2
            )

        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        outputs.append(decoded.split("English:")[-1].strip())

    return "\n".join(outputs)


def summarize_text(text):
    tokenizer, model = load_llm()

    prompt = f"""
Summarize the following text.

Text:
{text}

Summary:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            temperature=0.3
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return decoded.split("Summary:")[-1].strip()

# =========================================================
# RUNPOD HANDLER
# =========================================================
def handler(event):
    print("📥 Event received")

    # Warmup
    if event.get("input", {}).get("warmup"):
        return {"status": "warm"}

    pdf_base64 = event.get("input", {}).get("pdf_base64")
    if not pdf_base64:
        return {"error": "No PDF provided"}

    pdf_bytes = base64.b64decode(pdf_base64)

    # PDF → Images
    images = pdf_to_images(pdf_bytes)

    # OCR
    ru_text = "\n".join(ocr_images(images))

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

# =========================================================
# REQUIRED ENTRYPOINT (MUST BE LAST LINE)
# =========================================================
print("✅ Starting RunPod serverless handler")
runpod.serverless.start({"handler": handler})
