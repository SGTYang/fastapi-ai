FROM pytorch/pytorch:latest

RUN python -m pip install --no-cache-dir mlflow \
    efficientnet-pytorch \
    requests \
    pillow \
    fastapi \
    pydantic \
    typing \
    uvicorn[standard]

COPY fastapi-ai-api/ /app

WORKDIR /app
#CMD python -m uvicorn --host 0.0.0.0 --port 8000 main:app
CMD ["sleep", "1000"]