FROM runpod/pytorch:cuda11.8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ---- System dependencies ----
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# ---- Python dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- App ----
COPY handler.py .

CMD ["python", "handler.py"]
