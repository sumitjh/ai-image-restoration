FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir "numpy==1.26.4"
RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --no-build-isolation basicsr facexlib gfpgan realesrgan
RUN pip install --no-cache-dir fastapi uvicorn python-multipart pillow
RUN pip install --no-cache-dir "numpy==1.26.4"

# Copy application code
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
