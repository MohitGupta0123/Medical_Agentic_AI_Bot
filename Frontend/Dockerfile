# Frontend Dockerfile - Frontend API for DockerCompose

FROM python:3.12-slim

WORKDIR /app

# Install frontend-specific dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy only frontend code
COPY . .
# COPY DataBase ./DataBase/

EXPOSE 8501

# Run Streamlit on container startup
CMD ["streamlit", "run", "App.py", "--server.port=8501", "--server.address=0.0.0.0"]