import streamlit as st
import torch
from PIL import Image, ImageDraw
from PIL import ImageFont
import pathlib
import numpy as np
# import easyocr
import json
import pandas as pd
import fitz
import os
import requests
import pytesseract
# import cv2 as cv

from scipy.ndimage import median_filter

# from pdf2image import convert_from_bytes
from io import BytesIO
pathlib.PosixPath = pathlib.WindowsPath

def save_to_json_file(detections, image_filename):
    # Extract the name part of the image filename
    image_name = image_filename.split('.')[0]
    json_filename = f"savedData/cv/{image_name}.json"

    with open(json_filename, 'w',encoding="utf-8") as f:
        json.dump(detections, f,ensure_ascii=False, indent=4)
    st.success(f"Detection results saved to {json_filename}")

# def perform_ocr_on_image(image, language):
#     reader = easyocr.Reader([language])
#     result = reader.readtext(np.array(image))
#     text = []
#     for detection in result:
#         text.append(detection[1])
#     return text

def perform_ocr_on_image(image, language):
    image = image.convert('RGB')
    extracted_text = pytesseract.image_to_string(image, lang=language)
    return extracted_text.splitlines()

@st.cache(allow_output_mutation=True)
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

# def preprocess_cropped_image(cropped_image):
#     img_array = np.array(cropped_image)

#     gray = cv.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

#     filtered = cv.medianBlur(gray, 3)

#     return filtered

def preprocess_cropped_image(cropped_image):
    gray = cropped_image.convert("L")  

    gray_array = np.array(gray)

    filtered = median_filter(gray_array, size=3)

    return filtered

# Function to convert PDF to images
# def pdf_to_images(pdf_file):
#     images = []
#     with st.spinner("Converting PDF to images..."):
#         pdf_images = convert_from_bytes(pdf_file.read())
#         for page in pdf_images:
#             images.append(page)
#     return images


def pdf_to_images(uploaded_pdf):
    # Ouvre le PDF depuis les bytes uploadés
    pdf_bytes = uploaded_pdf.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=150)  # ajuste la qualité
        img_data = pix.tobytes("png")
        image = Image.open(BytesIO(img_data))
        images.append(image)

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

def mainCV():
    st.title("Object Detection from Resumes")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","pdf"])
    # Radio button to select language for EasyOCR
    language = st.radio("Select language for OCR:", ("en", "fr", "ar"))

    if 'detections' not in st.session_state:
        st.session_state.detections = []

    # if uploaded_image is not None:
    #     st.session_state.image = Image.open(uploaded_image)

    if uploaded_image is not None:
        if uploaded_image.type == "application/pdf":
            # Convert PDF to images
            pdf_images = pdf_to_images(uploaded_image)
            # Concatenate images into one single image
            concatenated_image = concatenate_images(pdf_images)
            # st.image(concatenated_image, caption="Concatenated Image")
            # Assign concatenated image to session state
            st.session_state.image = concatenated_image
        else:
            # Read uploaded image directly
            image = Image.open(uploaded_image)
            # st.image(image, caption="Uploaded Image")
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
                        # 2: "Community",
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
                        # 16: "References",
                        15: "Resume",
                        # 16: "Sexe",
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
                    # 'bound_box': [det["bbox"] for det in st.session_state.detections],
                    # 'Bounding Box': [det["bbox"] for det in st.session_state.detections],
                }
                df = pd.DataFrame(detections_table)

                # Modify and delete entries
                modified_classes = []
                modified_texts = []
                class_names2 = {
                "Adresse": 0,
                "Certifications": 1,
                # "Community": 2,
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
                # "References": 16,
                "Resume": 15,
                # "Sexe": 17,
                "Skills": 16,

            }

                deleted_indices = []
                for i, det in enumerate(st.session_state.detections):
                    with st.expander(f"Item {i + 1} : " + det['class_name']):
                        # Dropdown menu to select the class name
                        modified_class = st.selectbox(f"Class Name {i}", options=list(class_names2.keys()),
                                                      index=class_names2.get(det['class_name'], 0))

                        modified_text = st.text_area(f"Extracted Text {i}", value=det['extracted_text'],
                                                 key=f"text_area_{i}")
                        modified_classes.append(modified_class)
                        modified_texts.append(modified_text)

                    # Check if the delete button is clicked
                        if st.button(f"Delete Item {i}"):
                            deleted_indices.append(i)

                # Create a new list to store remaining items after deletion
                remaining_detections = []

                # Iterate through detections and remove selected items
                for i, det in enumerate(st.session_state.detections):
                    if i not in deleted_indices:
                        remaining_detections.append(det)

                # Update the detections list with remaining items
                st.session_state.detections = remaining_detections

                # Update DataFrame with remaining entries
                df['Class'] = modified_classes
                df['Extracted_text'] = modified_texts

                # Display the updated DataFrame
            st.table(df)
            # Filtrer les lignes correspondant aux produits et obtenir la colonne "Extracted_text"
            products_extracted_text = df.loc[df['Class'] == 'Products', 'Extracted_text']
            header_extracted_text = df.loc[df['Class'] == 'header', 'Extracted_text']

            # Afficher les résultats
            print(products_extracted_text)
            print(header_extracted_text)

            col1, col2, col3 = st.columns([1, 0.5, 1])
            with col2:
                # Button for saving to JSON file
                if st.button('Save to JSON file'):
                    save_to_json_file(df.to_dict(orient='records'), uploaded_image.name)

def mainCV2():
    st.title("Object Detection from Resumes")

    uploaded_image = st.file_uploader("Choose an image or PDF...", type=["jpg", "jpeg", "png", "pdf"])
    # Radio button to select language for OCR
    language = st.radio("Select language for OCR:", ("en", "fr", "ar"))

    if 'detections' not in st.session_state:
        st.session_state.detections = []

    if uploaded_image is not None:
        if uploaded_image.type == "application/pdf":
            # Convert PDF to images
            pdf_images = pdf_to_images(uploaded_image)
            # Concatenate images into one single image
            concatenated_image = concatenate_images(pdf_images)
            # st.image(concatenated_image, caption="Concatenated Image")
            # Assign concatenated image to session state
            st.session_state.image = concatenated_image
        else:
            # Read uploaded image directly
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image")
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
                        2: "Community",
                        3: "Education",
                        4: "Email",
                        5: "Experience",
                        6: "Github",
                        7: "Image",
                        8: "Interests",
                        9: "Languages",
                        10: "LinkedIn",
                        11: "Naissance",
                        12: "Name",
                        13: "Phone",
                        14: "Profil",
                        15: "Projects",
                        16: "References",
                        17: "Resume",
                        18: "Sexe",
                        19: "Skills",
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
                modified_class_names = []
                class_names2 = {
                    "Adresse": 0,
                    "Certifications": 1,
                    "Community": 2,
                    "Education": 3,
                    "Email": 4,
                    "Experience": 5,
                    "Github": 6,
                    "Image": 7,
                    "Interests": 8,
                    "Languages": 9,
                    "LinkedIn": 10,
                    "Naissance": 11,
                    "Name": 12,
                    "Phone": 13,
                    "Profil": 14,
                    "Projects": 15,
                    "References": 16,
                    "Resume": 17,
                    "Sexe": 18,
                    "Skills": 19,
                }

                deleted_indices = []


                for i, det in enumerate(st.session_state.detections):
                    with st.expander(f"Item {i + 1} : {modified_classes[i]}"):  # Use modified class name here
                        # Dropdown menu to select the class name
                        modified_class = st.selectbox(f"Class Name {i}", options=list(class_names2.keys()),
                                                      index=class_names2.get(det['class_name'], 0))
                        modified_class_names.append(modified_class)  # Store modified class name
                        modified_text = st.text_area(f"Extracted Text {i}", value=det['extracted_text'],
                                                     key=f"text_area_{i}")
                        modified_classes.append(modified_class)
                        modified_texts.append(modified_text)

                # Create a new list to store remaining items after deletion
                remaining_detections = []

                # Iterate through detections and remove selected items
                for i, det in enumerate(st.session_state.detections):
                    if i not in deleted_indices:
                        remaining_detections.append(det)

                # Update the detections list with remaining items
                st.session_state.detections = remaining_detections

                # Update DataFrame with remaining entries
                df['Class'] = modified_classes
                df['Extracted_text'] = modified_texts

                # Display the updated DataFrame
            st.table(df)

if __name__ == "__main__":
    mainCV()






