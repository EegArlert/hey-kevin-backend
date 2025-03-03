FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

RUN pip install fastapi \
    uvicorn ultralytics \
    opencv-python-headless python-multipart \
    python-dotenv

WORKDIR /app

COPY ./segmentation.py /app

EXPOSE 8000

CMD ["uvicorn", "segmentation:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
