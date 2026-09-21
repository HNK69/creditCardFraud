FROM python:3.12-slim

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt
COPY src/ ./src/
COPY models/ ./models/
COPY app/ ./app/
COPY config/ ./config/

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]