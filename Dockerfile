FROM runpod/pytorch:2.0.0-cuda11.8-runtime

WORKDIR /app

# Required for PDF + OpenCV
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python", "handler.py"]
