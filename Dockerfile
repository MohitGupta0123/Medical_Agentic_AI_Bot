# Backend Dockerfile

FROM python:3.12-slim

WORKDIR /app

# Install backend requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy backend files
COPY Src ./Src
COPY DataBase ./DataBase
COPY Artifacts ./Artifacts
COPY models ./models/

EXPOSE 7860

# Run FastAPI app using Uvicorn
CMD ["uvicorn", "Src.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "7860"]

# FROM python:3.12-slim

# # Set working directory
# WORKDIR /app

# # Copy requirements and install
# COPY requirements.txt ./requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the full app
# COPY Frontend ./Frontend
# COPY Src ./Src
# COPY DataBase ./DataBase
# COPY Artifacts ./Artifacts

# # Expose Streamlit port (Hugging Face Spaces requirement)
# EXPOSE 7860

# # Start both FastAPI and Streamlit
# CMD ["bash", "-c", "\
# uvicorn Src.api.fastapi_app:app --host 0.0.0.0 --port 8000 & \
# streamlit run Frontend/App.py --server.port 7860 --server.enableCORS false"]
