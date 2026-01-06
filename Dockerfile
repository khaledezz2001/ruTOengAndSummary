FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# -------------------------------------------------
# System deps
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
# Python deps
# -------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------
# 🔥 PRE-DOWNLOAD PADDLEOCR MODELS
# -------------------------------------------------
RUN python - <<'EOF'
from paddleocr import PaddleOCR
print("Pre-downloading PaddleOCR models...")
PaddleOCR(lang="ru", use_angle_cls=False)
print("PaddleOCR models downloaded")
EOF

# -------------------------------------------------
# 🔥 PRE-DOWNLOAD QWEN FILES (NO LOADING, NO CUDA)
# -------------------------------------------------
ENV HF_HOME=/models/hf

RUN python - <<'EOF'
from huggingface_hub import snapshot_download

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

print("Downloading Qwen model files only...")
snapshot_download(
    repo_id=MODEL_NAME,
    local_dir="/models/hf/models--Qwen--Qwen2.5-7B-Instruct",
    local_dir_use_symlinks=False
)
print("Qwen files downloaded")
EOF

# -------------------------------------------------
# App
# -------------------------------------------------
COPY handler.py .

CMD ["python", "handler.py"]
