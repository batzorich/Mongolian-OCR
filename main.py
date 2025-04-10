from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import base64
import numpy as np
import cv2
from src.service import extract_text

app = FastAPI()

class DataPayload(BaseModel):
    image_base64: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/extract-text")
async def extract_text_api(payload: DataPayload):
    try:
        img_data = base64.b64decode(payload.image_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        text = extract_text(image)
        return {"message": "Text extracted successfully", "text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message": "Error occurred", "details": str(e)})
