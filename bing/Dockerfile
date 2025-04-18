FROM python:3.10-bookworm

RUN pip install python-multipart requests fastapi uvicorn python-dotenv

WORKDIR /app

COPY . /app

COPY ../middleware.py /app/middleware.py

EXPOSE 8002

CMD ["uvicorn", "bing.bing:app", "--host", "0.0.0.0", "--port", "8001"]