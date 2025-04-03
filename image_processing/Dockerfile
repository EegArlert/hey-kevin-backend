FROM python:3.10-bookworm

RUN pip install fastapi \
    httpx \
    opencv-python-headless python-multipart \
    python-dotenv \
    uvicorn \
    ultralytics \
    torch
    
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

WORKDIR /app

COPY . /app

COPY ../middleware.py /app/middleware.py

EXPOSE 8000

CMD ["uvicorn", "image_processing.segmentation:app", "--host", "0.0.0.0", "--port", "8000"]
