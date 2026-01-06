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

# -------------------------------------------------
# ENV
# -------------------------------------------------
os.environ["HF_HOME"] = "/models/hf"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

ocr = None
tokenizer = None
model = None

# -------------------------------------------------
# OCR
# -------------------------------------------------
def load_ocr():
    global ocr
    if ocr is None:
        print("🔤 Loading PaddleOCR...")
        ocr = PaddleOCR(lang="ru", use_angle_cls=False)
        print("✅ PaddleOCR loaded")
    return ocr

# -------------------------------------------------
# LLM (4-bit, GPU, LOCAL FILES ONLY)
# -------------------------------------------------
def load_llm():
    global tokenizer, model
    if model is None:
        print("🤖 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            local_files_only=True
        )

        print("🤖 Loading model on GPU (4-bit)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="cuda",
            quantization_config=bnb_config,
            trust_remote_code=True,
            local_files_only=True
        )
        model.eval()
        print("✅ Qwen 7B loaded")

    return tokenizer, model

# -------------------------------------------------
# PDF → IMAGES
# -------------------------------------------------
def pdf_to_images(pdf_bytes):
    pages = convert_from_bytes(pdf_bytes, dpi=500)
    images = []
    for p in pages:
        img = np.array(p)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
    return images

# -------------------------------------------------
# OCR IMAGES
# -------------------------------------------------
def ocr_images(images):
    engine = load_ocr()
    texts = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 10
        )

        result = engine.ocr(gray, cls=False)
        if not result:
            continue

        for line in result:
            if not line or not line[1]:
                continue
            text = line[1][0]
            if isinstance(text, list):
                text = " ".join(text)
            text = str(text).strip()
            if text:
                texts.append(text)

    return texts

# -------------------------------------------------
# LLM TASKS
# -------------------------------------------------
def translate_ru_to_en(text):
    tokenizer, model = load_llm()
    prompt = f"Translate the following Russian text to English:\n{text}\nEnglish:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(out[0], skip_special_tokens=True).split("English:")[-1].strip()

def summarize(text):
    tokenizer, model = load_llm()
    prompt = f"Summarize the following text:\n{text}\nSummary:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(out[0], skip_special_tokens=True).split("Summary:")[-1].strip()

# -------------------------------------------------
# RUNPOD HANDLER
# -------------------------------------------------
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
    summary = summarize(en_text)

    return {
        "text_ru": ru_text,
        "text_en": en_text,
        "summary": summary
    }

print("✅ Starting RunPod serverless handler")
runpod.serverless.start({"handler": handler})
