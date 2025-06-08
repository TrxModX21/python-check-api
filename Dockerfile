# Gunakan base image yang ringan
FROM python:3.11-slim

# Buat direktori kerja
WORKDIR /app

# Salin file dependencies dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh isi project ke dalam container
COPY . .

# Jalankan server FastAPI pakai uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]