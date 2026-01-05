import base64
import cv2
import numpy as np
import torch
import runpod

from paddleocr import PaddleOCR
from pdf2image import convert_from_bytes
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------------------------
# LOAD MODELS (COLD START – VERY IMPORTANT)
# ------------------------------------------------

# OCR model
ocr = PaddleOCR(
    lang="ru",
    use_textline_orientation=True
)

# LLM model
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

model.eval()

# ------------------------------------------------
# UTILS
# ------------------------------------------------

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
    chunks = []
    current = ""

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
    translations = []

    for chunk in chunks:
        prompt = f"""
Translate the following Russian text to English.

Russian:
{chunk}

English:
"""
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.2,
                do_sample=False
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        translations.append(decoded.split("English:")[-1].strip())

    return "\n".join(translations)


def summarize_text(text):
    prompt = f"""
Summarize the following document in clear, concise English.
Focus on key points.

Document:
{text}

Summary:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.3,
            do_sample=False
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("Summary:")[-1].strip()

# ------------------------------------------------
# RUNPOD HANDLER
# ------------------------------------------------

def handler(event):
    # Warmup request
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
    ru_full_text = "\n".join(ru_texts)

    if not ru_full_text.strip():
        return {
            "text_ru": "",
            "text_en": "",
            "summary": ""
        }

    # Translation
    en_text = translate_ru_to_en(ru_full_text)

    # Summary
    summary = summarize_text(en_text)

    return {
        "text_ru": ru_full_text,
        "text_en": en_text,
        "summary": summary
    }

# ------------------------------------------------
# REQUIRED BY RUNPOD
# ------------------------------------------------
runpod.serverless.start({"handler": handler})
