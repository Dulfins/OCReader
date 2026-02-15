FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download Model
RUN python -c "import os; \
from huggingface_hub import snapshot_download; \
model_path = './sugoi_translator/model'; \
if not os.path.exists(model_path) or not os.listdir(model_path): \
    snapshot_download(repo_id='Dulfins/sugoi_translator', cache_dir=model_path

EXPOSE 8080


CMD ["uvicorn", "ocreader:app", "--host", "0.0.0.0", "--port", "8080"]