FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# -------------------------------------------------
# System dependencies
# -------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip

# -------------------------------------------------
# Python dependencies
# -------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------
# 🔥 PRE-DOWNLOAD PADDLEOCR MODELS
# -------------------------------------------------
RUN python - <<'EOF'
from paddleocr import PaddleOCR
print("Pre-downloading PaddleOCR models...")
ocr = PaddleOCR(lang="ru", use_angle_cls=False)
print("PaddleOCR models downloaded")
EOF

# -------------------------------------------------
# 🔥 PRE-DOWNLOAD QWEN 7B (4-BIT)
# -------------------------------------------------
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf

RUN python - <<'EOF'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

print("Downloading tokenizer...")
AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print("Downloading model (4-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="cpu",  # 🔥 IMPORTANT: CPU during build
    trust_remote_code=True
)

print("Qwen model downloaded")
EOF

# -------------------------------------------------
# App
# -------------------------------------------------
COPY handler.py .

CMD ["python", "handler.py"]
