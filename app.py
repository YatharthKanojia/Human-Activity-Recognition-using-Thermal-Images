from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from predict import predict_sequence

app = FastAPI()   # ⭐ THIS MUST BE PRESENT

@app.get("/")
def home():
    return {"message": "HAR API Running"}

@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):

    if len(files) != 8:
        return {"error": "Upload exactly 8 images"}

    frames = []

    for file in files:
        content = await file.read()
        npimg = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

        frames.append(img)

    result = predict_sequence(frames)

    return {"prediction": result}