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
    model_path = r'C:\Users\yassi\PycharmProjects\PfeProject\model\invoice\bestMoyenV3.pt'
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
            0: "Discount_Percentage",
            1: "Due_Date",
            2: "Email_Client",
            3: "Name_Client",
            4: "Products",
            5: "Remise",
            6: "Subtotal",
            7: "Tax",
            8: "Tax_Precentage",
            9: "Tel_Client",
            10: "billing address",
            11: "header",
            12: "invoice date",
            13: "invoice number",
            14: "shipping address",
            15: "total"
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
