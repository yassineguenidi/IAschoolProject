import streamlit as st
import torch
from PIL import Image, ImageDraw
from PIL import ImageFont
import pathlib
import numpy as np
import easyocr
import json

# Function to save data to JSON file
def save_to_json(data):
    file_path = r'C:\Users\yassi\PycharmProjects\PfeProject\savedData\detections.json'
    with open(file_path, 'w') as f:
        json.dump(data, f)
    st.success("Data saved to detections.json")

# Load EasyOCR reader
reader = easyocr.Reader(['en'])

def perform_ocr_on_image(image):
    result = reader.readtext(np.array(image))
    text = []
    for detection in result:
        text.append(detection[1])
    return text


@st.cache(allow_output_mutation=True)
def load_model():
    model_path = r'C:\Users\yassi\PycharmProjects\PfeProject\model\best.pt'
    model = torch.hub.load(r'C:\Users\yassi\PycharmProjects\PfeProject\yolov5\yolov5',
                           'custom',
                           path=model_path,
                           source='local',
                           force_reload=True
                           )
    return model

def detect_objects(model, image):
    results = model(image)
    return results


def draw_boxes(image, detections):
    draw = ImageDraw.Draw(image)

    font = ImageFont.load_default()

    for det in detections:
        bbox = det['bbox']
        class_name = det['class_name']
        confidence = det['confidence']

        draw.rectangle(bbox, outline='green', width=2)
        draw.rectangle([bbox[0], bbox[1], bbox[0], bbox[1]])
        draw.text((bbox[0]-5, bbox[1]-20), str(class_name)+' '+str(round(confidence, 3)), fill='green')

    return image

import streamlit as st
import json

def save_to_json(data):
    file_path = r"C:\Users\yassi\PycharmProjects\PfeProject\savedData\detections.json"
    with open(file_path, 'w') as f:
        json.dump(data, f)
    st.success("Data saved to data.json")

def main():
    st.title("Data Saving App")

    # Input fields for name and age
    name = st.text_input("Enter your name:")
    age = st.number_input("Enter your age:")

    # Button to save data
    if st.button("Save Data"):
        data = {"name": name, "age": age}
        save_to_json(data)

if __name__ == "__main__":
    main()
