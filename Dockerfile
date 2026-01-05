FROM runpod/pytorch:2.1.0-cuda11.8.0-runtime

WORKDIR /app

# System deps for PDF
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python", "handler.py"]
