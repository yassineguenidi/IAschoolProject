import pathlib

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import numpy as np
import easyocr
from io import BytesIO

pathlib.PosixPath = pathlib.WindowsPath

app = FastAPI()

@app.on_event("startup")
def load_model():
    global model
    model_path = r'C:\Users\yassi\PycharmProjects\PfeProject\model\cv\bestCvMoyenV3.pt'
    model = torch.hub.load(r'C:\Users\yassi\PycharmProjects\PfeProject\yolov5\yolov5',
                           'custom',
                           path=model_path,
                           source='local',
                           force_reload=True
                           )

@app.post("/detect-objects/")
async def detect_objects(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    results = model(image)

    detections = []
    for det in results.pred[0]:
        bbox = det[:4].tolist()
        conf = det[4].item()
        class_id = int(det[5].item())
        class_names = {
            0: "Adresse",
            1: "Certifications",
            2: "Education",
            3: "Email",
            4: "Experience",
            5: "Github",
            6: "Image",
            7: "Interests",
            8: "Languages",
            9: "LinkedIn",
            10: "Naissance",
            11: "Name",
            12: "Phone",
            13: "Profil",
            14: "Projects",
            15: "Resume",
            16: "Skills",
        }
        class_name = class_names.get(class_id, f'Class {class_id}')
        detections.append({
            'bbox': bbox,
            'confidence': round(conf, 5),
            'class_name': class_name
        })

    return JSONResponse(content={"detections": detections})

@app.post("/perform-ocr/")
async def perform_ocr(file: UploadFile = File(...), language: str = 'en'):
    reader = easyocr.Reader([language])
    image = Image.open(BytesIO(await file.read()))
    result = reader.readtext(np.array(image))
    text = [detection[1] for detection in result]

    return JSONResponse(content={"text": text})

