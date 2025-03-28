FROM python:3.10-bookworm

RUN pip install fastapi \
    opencv-python-headless python-multipart \
    python-dotenv \
    uvicorn \
    ultralytics \
    pytorch
    
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

WORKDIR /app

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "segmentation:app", "--host", "0.0.0.0", "--port", "8000"]
