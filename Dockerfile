FROM python:3.12.1-slim-bookworm
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Minimal runtime libs (libgl1 usually not needed with headless opencv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN python -m pip install --upgrade pip \
    # ✅ Force CPU-only PyTorch (prevents CUDA wheels -> huge images)
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    # ✅ PaddlePaddle CPU (you already do this correctly)
    && python -m pip install --no-cache-dir -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
        paddlepaddle==3.1.1 \
    # Install the rest (ultralytics will reuse the torch you already installed)
    && python -m pip install --no-cache-dir -r requirements.txt 
    

COPY plates_api.py ./
COPY plate_model.pt ./

EXPOSE 8001
CMD ["uvicorn", "plates_api:app", "--host", "0.0.0.0", "--port", "8001"]