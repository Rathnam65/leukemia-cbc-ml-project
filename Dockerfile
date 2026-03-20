# Use a slim Python base image
FROM python:3.12-slim

# Install system dependencies (if needed by pdfplumber)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpoppler-cpp-dev \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY backend/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Ensure a model exists (train if missing)
RUN python backend/train_model.py

EXPOSE 5000

# Run using a production-ready server
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "backend.app:app"]
