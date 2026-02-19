from fastapi import FastAPI, UploadFile
from .schemas import BatchResponse, ImageResult, TextBox
from typing import List
from io import BytesIO
from PIL import Image
from pathlib import Path
import numpy as np
import base64
import cv2


from .inference import model2annotations, load_text_detector
from .draw_image import render_translation_on_image
from .sugoi_translator.translator import translate_ja_to_en
from manga_ocr import MangaOcr

app = FastAPI()

TEXT_DET_MODEL = Path(__file__).parent / 'data/comictextdetector.pt'
model = load_text_detector(str(TEXT_DET_MODEL))

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

def to_base64(img: np.ndarray) -> str:
    pil_img = Image.fromarray(img)
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

def inpaint_image(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Clean mask (helps residue)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    inpainted_bgr = cv2.inpaint(
        image_bgr,
        mask,
        inpaintRadius=5,
        flags=cv2.INPAINT_TELEA
    )

    # Back to RGB for PIL rendering
    return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)



@app.post("/process_image", response_model=BatchResponse)
async def process_image(files: List[UploadFile]):
    output = []

    for file in files:
        img_pil = Image.open(file.file).convert("RGB")
        img_rgb = np.array(img_pil)

        mask, boxes = detect_text(img_rgb)
        boxes = to_python_boxes(boxes)

        texts = ocr_boxes(img_pil, boxes)
        translations = translate(texts)

        # --- inpaint ---
        inpainted_rgb = inpaint_image(img_rgb, mask)

        # --- render translations ON inpainted image ---
        translated_img = render_translation_on_image(
            image_np=inpainted_rgb,
            boxes=boxes,
            translations=translations,
            font_path="app/fonts/animeace2_bld.ttf"
        )

        # --- encode final image ---
        image_base64 = to_base64(np.array(translated_img))

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
