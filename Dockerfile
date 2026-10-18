FROM python:3.11-slim

# System deps for opencv, rasterio, geopandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p checkpoints outputs dataset/images dataset/masks

EXPOSE 8000 8501

# Default: run the FastAPI backend. Override CMD to run the Streamlit
# dashboard instead: `streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0`
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
