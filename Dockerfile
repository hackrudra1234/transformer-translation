FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir --disable-pip-version-check -r requirements.txt

COPY app ./app
COPY model ./model
COPY inference ./inference
COPY data ./data

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]