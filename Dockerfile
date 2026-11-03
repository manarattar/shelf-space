FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SHELFSPACE_RESULTS=/app/results

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[app]"

COPY app ./app
COPY results ./results
COPY .streamlit ./.streamlit

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
