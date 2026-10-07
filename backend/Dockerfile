FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# CPU-only torch keeps the image small. For GPU hosts, use a CUDA base image and drop this line.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV DATA_DIR=/tmp/yolo-merge
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
