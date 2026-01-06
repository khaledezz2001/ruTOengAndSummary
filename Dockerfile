FROM runpod/pytorch:2.1.0-cuda11.8.0-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ---- System dependencies (ONLY what we need) ----
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- App ----
COPY handler.py .

CMD ["python", "handler.py"]
