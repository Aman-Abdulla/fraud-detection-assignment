# Optional: containerized inference service (Docker is optional per assignment FAQ)
FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r api/requirements.txt

COPY api/main.py ./api/main.py
COPY src ./src
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]