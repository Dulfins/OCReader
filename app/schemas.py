from pydantic import BaseModel
from typing import List

class TextBox(BaseModel):
    box: List[int]
    jp: str
    en: str

class ImageResult(BaseModel):
    width: int
    height: int
    image_base64: str
    results: List[TextBox]

class BatchResponse(BaseModel):
    images: List[ImageResult]
