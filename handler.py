# =========================================================
# RunPod Serverless Handler
# FINAL VERSION – OCR FIXED (DPI + PREPROCESSING)
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
    # 🔥 Higher DPI = better OCR for small text PDFs
    pages = convert_from_bytes(pdf_bytes, dpi=500)
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
        # ---- PREPROCESSING (CRITICAL) ----
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

        if result is None:
            continue

        for line in result:
            if line is None:
                continue
            if not isinstance(line, (list, tuple)):
                continue
            if len(line) < 2:
                continue
            if line[1] is None:
                continue
            if not isinstance(line[1], (list, tuple)):
                continue
            if len(line[1]) < 1:
                continue

            text = line[1][0]
            if text and isinstance(text, str):
                texts.append(
