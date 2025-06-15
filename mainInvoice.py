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
from io import BytesIO
import sys
# import cv2 as cv

from scipy.ndimage import median_filter

pathlib.PosixPath = pathlib.WindowsPath


def save_to_json_file(detections, image_filename):
    # Extract the name part of the image filename
    image_name = image_filename.split('.')[0]
    json_filename = f"savedData/invoice/{image_name}.json"

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
    # https://drive.google.com/file/d/1mzm7SWktezu7po1kkhLN6r0SE54bUIQB/view?usp=drive_link
    # ID du fichier Google Drive
    # file_id = '1yPkUMVGGCOMb3lEdVbz3x7gWBhcyV7Ei'
    file_id = '1mzm7SWktezu7po1kkhLN6r0SE54bUIQB'
    download_url = f'https://drive.google.com/uc?id={file_id}'

    # Chemin local pour sauvegarder le modèle
    model_filename = 'bestLV3.pt'

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

#     gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

#     filtered = cv.medianBlur(gray, 3)

#     return filtered

def preprocess_cropped_image(cropped_image):
    
    gray = cropped_image.convert("L")  

    gray_array = np.array(gray)

    filtered = median_filter(gray_array, size=3)

    return filtered

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

def mainInvoice():
    st.title("Object Detection from invoices")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "pdf"])
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
                "Discount_Percentage": 0,
                "Due_Date": 1,
                "Email_Client": 2,
                "Name_Client": 3,
                "Products": 4,
                "Remise": 5,
                "Subtotal": 6,
                "Tax": 7,
                "Tax_Precentage": 8,
                "Tel_Client": 9,
                "billing address": 10,
                "header": 11,
                "invoice date": 12,
                "invoice number": 13,
                "shipping address": 14,
                "total": 15
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

# if __name__ == "__main__":
#     mainInvoice()






