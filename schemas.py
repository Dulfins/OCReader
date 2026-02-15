from pydantic import BaseModel
from typing import List

class TextBox(BaseModel):
    box: List[int]
    jp: str
    en: str

class ImageResult(BaseModel):
    filename: str
    width: int
    height: int
    results: List[TextBox]

class BatchResponse(BaseModel):
    images: List[ImageResult]
