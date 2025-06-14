import cv2
import streamlit as st
import torch
from PIL import Image, ImageDraw
from PIL import ImageFont
import pathlib
import numpy as np
import easyocr
import json
import pandas as pd

from pdf2image import convert_from_bytes
from io import BytesIO
pathlib.PosixPath = pathlib.WindowsPath

def save_to_json_file(detections, image_filename):
    # Extract the name part of the image filename
    image_name = image_filename.split('.')[0]
    json_filename = f"savedData/cv/{image_name}.json"

    with open(json_filename, 'w',encoding="utf-8") as f:
        json.dump(detections, f,ensure_ascii=False, indent=4)
    st.success(f"Detection results saved to {json_filename}")

def perform_ocr_on_image(image, language):
    reader = easyocr.Reader([language])
    result = reader.readtext(np.array(image))
    text = []
    for detection in result:
        text.append(detection[1])
    return text

@st.cache(allow_output_mutation=True)
# def load_model():
#     model_path = r'C:\Users\yassi\PycharmProjects\PfeProject\model\cv\bestCvMoyenV3.pt'
#     # model_path = r'C:\Users\yassi\PycharmProjects\PfeProject\model\cv\bestMoyenCvV1.pt'
#     # model_path = r'C:\Users\yassi\PycharmProjects\PfeProject\model\cv\bestLargeCVLast70.pt'
#     model = torch.hub.load(r'C:\Users\yassi\PycharmProjects\PfeProject\yolov5\yolov5',
#                            'custom',
#                            path=model_path,
#                            source='local',
#                            force_reload=True
#                            )

#     # model.conf = 0.5
#     # model.iou = 0.1
#     return model

def load_model():
    # ID du fichier Google Drive
    file_id = '1yPkUMVGGCOMb3lEdVbz3x7gWBhcyV7Ei'
    download_url = f'https://drive.google.com/uc?id={file_id}'

    # Chemin local pour sauvegarder le modèle
    model_filename = 'bestCvMoyenV3.pt'

    # Télécharger le modèle s’il n'existe pas déjà
    if not os.path.exists(model_filename):
        response = requests.get(download_url)
        with open(model_filename, 'wb') as f:
            f.write(response.content)

    # Charger le modèle avec torch.hub
    model = torch.hub.load('ultralytics/yolov5',  # dépôt GitHub officiel
                           'custom',
                           path=model_filename,
                           source='github',
                           force_reload=True)

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
        draw.rectangle([bbox[0] , bbox[1] , bbox[0] , bbox[1]])
        draw.text((bbox[0]-5, bbox[1]-20), str(class_name)+' '+str(round(confidence, 3)), fill='green')

    return image

def preprocess_cropped_image(cropped_image):
    img_array = np.array(cropped_image)

    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)


# Function to convert PDF to images
def pdf_to_images(pdf_file):
    images = []
    with st.spinner("Converting PDF to images..."):
        pdf_images = convert_from_bytes(pdf_file.read())
        for page in pdf_images:
            images.append(page)
    return images

# Function to concatenate images into one single image
def concatenate_images(images):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_image = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    return new_image


import spacy
from transformers import pipeline
import json
import pandas as pd
import streamlit as st
from PIL import Image

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Load pre-trained model for summarization
custom_summarizer = pipeline("summarization", model="t5-base", max_length=300, min_length=30, do_sample=True,
                             temperature=0.7)


def generate_resume_summary(detections):
    # Define the desired class names
    desired_classes = {
        "Certifications",
        "Education",
        "Experience",
        "Interests",
        "Languages",
        "Profil",
        "Projects",
        "Resume",
        "Skills",
    }

    # Initialize the summary dictionary
    summary = {}

    # Iterate through the detections
    for detection in detections:
        class_name = detection['class_name']
        extracted_text = detection['extracted_text']

        # Check if the class name is in the desired classes
        if class_name in desired_classes:
            if class_name in summary:
                summary[class_name] += " " + extracted_text
            else:
                summary[class_name] = extracted_text

    # Generate the summarized text using the custom summarizer
    summarized_text = custom_summarizer(" ".join(summary.values()), max_length=150, min_length=50, do_sample=False)

    # Add the summarized text to the summary dictionary
    summary['Resume Summary'] = summarized_text[0]['summary_text']

    return summary


def save_to_json_file(detections, image_filename):
    # Extract the name part of the image filename
    image_name = image_filename.split('.')[0]
    json_filename = f"savedData/{image_name}.json"

    with open(json_filename, 'w', encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False, indent=4)
    st.success(f"Detection results saved to {json_filename}")


def mainCV():
    st.title("Object Detection from Resumes")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "pdf"])
    # Radio button to select language for EasyOCR
    language = st.radio("Select language for OCR:", ("en", "fr", "ar"))

    if 'detections' not in st.session_state:
        st.session_state.detections = []

    if uploaded_image is not None:
        if uploaded_image.type == "application/pdf":
            # Convert PDF to images
            pdf_images = pdf_to_images(uploaded_image)
            # Concatenate images into one single image
            concatenated_image = concatenate_images(pdf_images)
            st.session_state.image = concatenated_image
        else:
            # Read uploaded image directly
            image = Image.open(uploaded_image)
            st.session_state.image = image

    if 'image' in st.session_state and st.session_state.image is not None:
        model = load_model()
        image = st.session_state.image
        col1, col2, col3 = st.columns([1.75, 5, 0.5])
        with col2:
            st.image(image, caption='Uploaded Image')

        col1, col2, col3 = st.columns([1.1, 0.3, 1])
        with col2:
            if st.button('Detect Objects'):
                with st.spinner('Detecting objects...'):
                    results = detect_objects(model, image)
                    print(results)

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
                    cropped_image = image.crop(bbox)
                    preprocessed_cropped_image = cropped_image
                    cropped_text = perform_ocr_on_image(preprocessed_cropped_image, language.lower())
                    text_to_display = ' '.join(cropped_text)

                    detections.append({
                        'bbox': bbox,
                        'confidence': round(conf, 5),
                        'class_name': class_name,
                        'extracted_text': text_to_display
                    })

                st.session_state.detections = detections

        # Display detection results
        if st.session_state.detections:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader('Image with BBOX')
                st.subheader('')
                st.subheader('')

                image_with_boxes = draw_boxes(image.copy(), st.session_state.detections)
                st.image(image_with_boxes, caption='Image with Bounding Boxes', use_column_width=True)
            with col2:
                st.subheader('Detection Results')
                detections_table = {
                    'Class': [det["class_name"] for det in st.session_state.detections],
                    'Confidence': [det["confidence"] for det in st.session_state.detections],
                    'Extracted_text': [det["extracted_text"] for det in st.session_state.detections],
                }
                df = pd.DataFrame(detections_table)

                # Modify and delete entries
                modified_classes = []
                modified_texts = []
                class_names2 = {
                    "Adresse": 0,
                    "Certifications": 1,
                    "Education": 2,
                    "Email": 3,
                    "Experience": 4,
                    "Github": 5,
                    "Image": 6,
                    "Interests": 7,
                    "Languages": 8,
                    "LinkedIn": 9,
                    "Naissance": 10,
                    "Name": 11,
                    "Phone": 12,
                    "Profil": 13,
                    "Projects": 14,
                    "Resume": 15,
                    "Skills": 16,
                }

                deleted_indices = []
                for i, det in enumerate(st.session_state.detections):
                    with st.expander(f"Item {i + 1} : " + det['class_name']):
                        modified_class = st.selectbox(f"Class Name {i}", options=list(class_names2.keys()),
                                                      index=class_names2.get(det['class_name'], 0))

                        modified_text = st.text_area(f"Extracted Text {i}", value=det['extracted_text'],
                                                     key=f"text_area_{i}")
                        modified_classes.append(modified_class)
                        modified_texts.append(modified_text)

                        if st.button(f"Delete Item {i}"):
                            deleted_indices.append(i)

                remaining_detections = []
                for i, det in enumerate(st.session_state.detections):
                    if i not in deleted_indices:
                        remaining_detections.append(det)

                st.session_state.detections = remaining_detections

                df['Class'] = modified_classes
                df['Extracted_text'] = modified_texts

                st.table(df)

            col1, col2, col3 = st.columns([1, 0.5, 1])
            with col2:
                # Button for saving to JSON file
                if st.button('Save to JSON file'):
                    # Generate the resume summary
                    resume_summary = generate_resume_summary(st.session_state.detections)
                    st.session_state.detections.append({
                        'class_name': 'Resume Summary',
                        'extracted_text': resume_summary['Resume Summary']
                    })

                    # Save to JSON file
                    save_to_json_file(st.session_state.detections, uploaded_image.name)


if __name__ == "__main__":
    mainCV()

