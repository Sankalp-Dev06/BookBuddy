# Use Python 3.10 base image
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire app
COPY . .

# Run with gunicorn
CMD ["gunicorn", "api:app", "--bind", "0.0.0.0:8000"]
