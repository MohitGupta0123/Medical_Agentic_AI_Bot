# Src/Dockerfile - Backend API for DockerCompose

FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY ../requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source files, database, and artifacts
COPY . .
COPY Artifacts ./Artifacts/
COPY models ./models/

EXPOSE 7860

CMD ["uvicorn", "Src.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "7860"]