FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# -------------------------------------------------
# System dependencies (REQUIRED)
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

# -------------------------------------------------
# Python setup
# -------------------------------------------------
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip

# -------------------------------------------------
# PyTorch GPU (CUDA 11.8)
# -------------------------------------------------
RUN pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# -------------------------------------------------
# Python dependencies
# -------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------
# App
# -------------------------------------------------
COPY handler.py .

CMD ["python", "handler.py"]
