from fastapi import FastAPI, UploadFile
from schemas import BatchResponse, ImageResult, TextBox
from typing import List
from PIL import Image
import numpy as np

from inference import model2annotations, load_text_detector
from sugoi_translator.translator import translate_ja_to_en
from manga_ocr import MangaOcr

app = FastAPI()

model_path = r'data/comictextdetector.pt'
model = load_text_detector(model_path)

ocr = MangaOcr()

def detect_text(image_np):
    _, _, boxes = model2annotations(model, image_np)
    return boxes

def ocr_boxes(img, boxes):
    ocr_texts = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cropped_image = img.crop((x1, y1, x2, y2))
        ocr_texts.append(ocr(cropped_image))

    return ocr_texts

def translate(texts):
    return translate_ja_to_en(texts)

# For json output, convert numpy int to python int
def to_python_boxes(boxes):
    out = []
    for b in boxes:
        out.append([
            int(b[0]),
            int(b[1]),
            int(b[2]),
            int(b[3]),
        ])
    return out



@app.post("/process_image", response_model=BatchResponse)
async def process_image(files: List[UploadFile]):
    output = []
    
    for file in files:
        img_pil = Image.open(file.file).convert("RGB")
        img_np = np.array(img_pil)

        boxes = detect_text(img_np)
        boxes = to_python_boxes(boxes)

        texts = ocr_boxes(img_pil, boxes)
        translations = translate(texts)

        results = [
            TextBox(box=box, jp=jp, en=en)
            for box, jp, en in zip(boxes, texts, translations)
        ]

        output.append(
            ImageResult(
                filename=file.filename,
                width=img_pil.width,
                height=img_pil.height,
                results=results
            )
        )

    return {"images": output}