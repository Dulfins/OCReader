from fastapi import FastAPI, UploadFile
from schemas import BatchResponse, ImageResult, TextBox
from typing import List
from io import BytesIO
from PIL import Image
import numpy as np
import base64
import cv2


from inference import model2annotations, load_text_detector
from sugoi_translator.translator import translate_ja_to_en
from manga_ocr import MangaOcr

app = FastAPI()

model_path = r'data/comictextdetector.pt'
model = load_text_detector(model_path)

ocr = MangaOcr()

def detect_text(image_np):
    _, mask, boxes = model2annotations(model, image_np)
    return mask, boxes

def ocr_boxes(img, boxes):
    ocr_texts = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cropped_image = img.crop((x1, y1, x2, y2))
        ocr_texts.append(ocr(cropped_image))

    return ocr_texts

def translate(texts):
    return translate_ja_to_en(texts)

def cv2_to_base64(img_bgr: np.ndarray) -> str:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

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

def inpaint_image(image_np, mask):
    inpainted = cv2.inpaint(image_np, mask, 3, cv2.INPAINT_TELEA)
    return inpainted


@app.post("/process_image", response_model=BatchResponse)
async def process_image(files: List[UploadFile]):
    output = []
    
    for file in files:
        img_pil = Image.open(file.file).convert("RGB")
        img_np = np.array(img_pil)

        mask, boxes = detect_text(img_np)
        boxes = to_python_boxes(boxes)

        texts = ocr_boxes(img_pil, boxes)
        translations = translate(texts)

        inpainted_image = inpaint_image(img_np, mask)
        image_base64 = cv2_to_base64(inpainted_image)

        results = [
            TextBox(box=box, jp=jp, en=en)
            for box, jp, en in zip(boxes, texts, translations)
        ]

        output.append(
            ImageResult(
                width=img_pil.width,
                height=img_pil.height,
                image_base64=image_base64,
                results=results
            )
        )

    return {"images": output}