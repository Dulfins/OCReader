# ---------- Builder Stage ----------
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt

# Download ComicTextDetector model
RUN mkdir -p app/data \
    && curl -L \
      https://huggingface.co/Dulfins/comictextdectector/resolve/main/comictextdetector.pt \
      -o app/data/comictextdetector.pt

# Download & extract Sugoi Translator model
RUN mkdir -p app/sugoi_translator/model \
    && curl -L \
      https://huggingface.co/Dulfins/sugoi_translator/resolve/main/model.tar.gz \
      -o /tmp/sugoi_model.tar.gz \
    && tar -xzf /tmp/sugoi_model.tar.gz -C app/sugoi_translator \
    && rm /tmp/sugoi_model.tar.gz

COPY . .

# ---------- Final Runtime Stage ----------
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY --from=builder /app /app

ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "app.ocreader:app", "--host", "0.0.0.0", "--port", "8000"]
