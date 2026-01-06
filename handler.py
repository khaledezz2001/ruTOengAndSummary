# =========================================================
# RunPod Serverless Handler (FINAL, OFFLINE, STABLE)
# =========================================================

print("🚀 Handler file imported")

import os
import base64
import cv2
import numpy as np
import torch
import runpod

from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# ---------------------------------------------------------
# ENV: offline mode
# ---------------------------------------------------------
os.environ["HF_HOME"] = "/models/hf"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ---------------------------------------------------------
# LOCAL MODEL PATH (PRE-DOWNLOADED IN DOCKER)
# ---------------------------------------------------------
MODEL_DIR = "/models/hf/models--Qwen--Qwen2.5-7B-Instruct"

# ---------------------------------------------------------
# GLOBALS
# ---------------------------------------------------------
ocr = None
tokenizer = None
model = None

# ---------------------------------------------------------
# LOAD OCR (PRE-DOWNLOADED)
# ---------------------------------------------------------
def load_ocr():
    global ocr
    if ocr is None:
        print("🔤 Loading PaddleOCR...")
        ocr = PaddleOCR(
            lang="ru",
            use_angle_cls=False,
            det=True,
            rec=True
        )
        print("✅ PaddleOCR loaded")
    return ocr

# ---------------------------------------------------------
# LOAD LLM (4-BIT, GPU, LOCAL FILES ONLY)
# ---------------------------------------------------------
def load_llm():
    global tokenizer, model
    if model is None:
        print("🤖 Loading tokenizer (local path)...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            local_files_only=True
        )

        print("🤖 Loading model on GPU (4-bit, local path)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            device_map="cuda",
            quantization_config=bnb_config,
            trust_remote_code=True,
            local_files_only=True
        )

        model.eval()
        print("✅ Qwen 7B loaded from disk")

    return tokenizer, model

# ---------------------------------------------------------
# PDF → IMAGES
# ---------------------------------------------------------
def pdf_to_images(pdf_bytes):
    pages = convert_from_bytes(pdf_bytes, dpi=500)
    images = []
    for page in pages:
        img = np.array(page)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
    return images

# ---------------------------------------------------------
# OCR IMAGES (YOUR FUNCTION – CORRECT)
# ---------------------------------------------------------
def ocr_images(images):
    engine = load_ocr()
    texts = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10
        )

        result = engine.ocr(gray, cls=False)
        if not result:
            continue

        for line in result:
            if not line or len(line) < 2 or not line[1]:
                continue

            raw = line[1][0]

            # 🔥 FLATTEN ANY SHAPE TO STRING
            def flatten(x):
                if isinstance(x, str):
                    return x
                if isinstance(x, list):
                    return " ".join(flatten(i) for i in x)
                return str(x)

            text = flatten(raw).strip()
            if text:
                texts.append(text)

    return texts

# ---------------------------------------------------------
# TRANSLATE RU → EN
# ---------------------------------------------------------
def translate_ru_to_en(text):
    tokenizer, model = load_llm()

    prompt = f"""
Translate the following Russian text to English.

Russian:
{text}

English:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.2
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return decoded.split("English:")[-1].strip()

# ---------------------------------------------------------
# SUMMARIZE
# ---------------------------------------------------------
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
            max_new_tokens=256,
            do_sample=False,
            temperature=0.3
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return decoded.split("Summary:")[-1].strip()

# ---------------------------------------------------------
# RUNPOD HANDLER
# ---------------------------------------------------------
def handler(event):
    print("📥 Event received")

    if event.get("input", {}).get("warmup"):
        return {"status": "warm"}

    pdf_base64 = event.get("input", {}).get("pdf_base64")
    if not pdf_base64:
        return {"error": "No PDF provided"}

    pdf_bytes = base64.b64decode(pdf_base64)

    images = pdf_to_images(pdf_bytes)
    ru_text = "\n".join(ocr_images(images))

    if not ru_text.strip():
        return {"error": "No text detected"}

    en_text = translate_ru_to_en(ru_text)
    summary = summarize_text(en_text)

    # 🔍 DEBUG LOGS (THIS IS THE IMPORTANT PART)
    print("========== OCR RU TEXT ==========")
    print(ru_text[:2000])
    print("========== TRANSLATED EN TEXT ==========")
    print(en_text[:2000])
    print("========== SUMMARY ==========")
    print(summary)
    print("========== END OUTPUT ==========")

    return {
        "text_ru": ru_text,
        "text_en": en_text,
        "summary": summary
    }


# ---------------------------------------------------------
# ENTRYPOINT (REQUIRED)
# ---------------------------------------------------------
print("✅ Starting RunPod serverless handler")
runpod.serverless.start({"handler": handler})
