FROM python:3.13

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . /app

# Download ComicTextDetector model
RUN mkdir -p data \
 && curl -L -o data/comictextdetector.pt \
    https://huggingface.co/Dulfins/comictextdectector/resolve/main/comictextdetector.pt

# Download Sugoi Model
RUN mkdir -p sugoi_translator/model \
 && curl -L https://huggingface.co/Dulfins/sugoi_translator/resolve/main/model.tar.gz | tar -xz -C sugoi_translator/model



EXPOSE 8000

CMD ["uvicorn", "app.ocreader:app", "--host", "0.0.0.0", "--port", "8000"]
