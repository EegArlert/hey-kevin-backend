FROM python:3.10-bookworm

RUN pip install fastapi \
    opencv-python-headless python-multipart \
    python-dotenv \
    #Dependencies for Ultralytics 
    uvicorn ultralytics \
    # Dependencies for SAM
    torch torchvision torchaudio \
    matplotlib pycocotools onnxruntime onnx numpy
    
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

WORKDIR /app

# RUN pip install git+https://github.com/facebookresearch/segment-anything.git
    
COPY ./sam /app/sam
    
COPY ./segmentation.py /app

EXPOSE 8000

CMD ["uvicorn", "segmentation:app", "--host", "0.0.0.0", "--port", "8000"]
