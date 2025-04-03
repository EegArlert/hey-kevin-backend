FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install fastapi python-multipart openai uvicorn python-dotenv

EXPOSE 8002

CMD ["uvicorn", "chat_gpt.gpt_service:app", "--host", "0.0.0.0", "--port", "8002"]
