FROM runpod/pytorch:2.1.0-cuda11.8.0-runtime

WORKDIR /app

# ---- System dependencies for pdf2image ----
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# ---- Python dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- App ----
COPY handler.py .

CMD ["python", "handler.py"]
