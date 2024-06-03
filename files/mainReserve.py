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

pathlib.PosixPath = pathlib.WindowsPath


# Function to save data to JSON file
# def save_to_json(data):
#     file_path = r'C:\Users\yassi\PycharmProjects\PfeProject\savedData\detections.json'
#     with open(file_path, 'w') as f:
#         json.dump(data, f)
#     st.success("Data saved to detections.json")
#
# def save_to_json(data):
#     file_path = r'C:\Users\yassi\PycharmProjects\PfeProject\savedData\detections.json'
#     if os.path.exists(file_path):
#         with open(file_path, 'r') as f:
#             existing_data = json.load(f)
#     else:
#         existing_data = []
#     existing_data.append(data)
#     with open(file_path, 'w') as f:
#         json.dump(existing_data, f)
#     st.success("Data appended to detections.json")
# def save_to_json_file(detections, filename):
#
#   with open(filename, 'w') as f:
#     json.dump(detections, f, indent=4)
#     st.success(f"Detection results saved to {filename}")

def save_to_json_file(detections, image_filename):
    # Extract the name part of the image filename
    image_name = image_filename.split('.')[0]
    json_filename = f"savedData/invoice/{image_name}.json"

    with open(json_filename, 'w') as f:
        json.dump(detections, f, indent=4)
    st.success(f"Detection results saved to {json_filename}")


# reader = easyocr.Reader(['en'])
#
# def perform_ocr_on_image(image):
#     result = reader.readtext(np.array(image))
#     text = []
#     for detection in result:
#         text.append(detection[1])
#     return text

def perform_ocr_on_image(image, language):
    reader = easyocr.Reader([language])
    result = reader.readtext(np.array(image))
    text = []
    for detection in result:
        text.append(detection[1])
    return text


@st.cache(allow_output_mutation=True)
def load_model():
    model_path = r'C:\Users\yassi\PycharmProjects\PfeProject\model\bestv3.pt'
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
        draw.text((bbox[0] - 5, bbox[1] - 20), str(class_name) + ' ' + str(round(confidence, 3)), fill='green')

    return image


def preprocess_cropped_image(cropped_image):
    img_array = np.array(cropped_image)

    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    filtered = cv2.medianBlur(gray, 3)

    return filtered


def main():
    st.title("Object Detection from invoices")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    # Radio button to select language for EasyOCR
    language = st.radio("Select language for OCR:", ("en", "fr", "ar"))

    if 'image' not in st.session_state:
        st.session_state.image = None

    if uploaded_image is not None:
        st.session_state.image = Image.open(uploaded_image)

    if st.session_state.image is not None:
        model = load_model()
        image = st.session_state.image

        st.image(image, caption='Uploaded Image', use_column_width=True)
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
        valid = False
        if st.button('Detect Objects'):
            valid = True
            with st.spinner('Detecting objects...'):
                results = detect_objects(model, image)
                print(results)

            detections = []
            for det in results.pred[0]:
                # print(det)
                bbox = det[:4].tolist()
                conf = det[4].item()
                class_id = int(det[5].item())

                class_name = class_names.get(class_id, f'Class {class_id}')
                cropped_image = image.crop(bbox)

                preprocessed_cropped_imag = preprocess_cropped_image(cropped_image)

                # cropped_text = perform_ocr_on_image(cropped_image)
                cropped_text = perform_ocr_on_image(preprocessed_cropped_imag, language.lower())
                text_to_display = '\n'.join(cropped_text)
                print(class_id)
                print(class_name)
                print(conf)
                print(text_to_display)

                detections.append({
                    'bbox': bbox,
                    'confidence': round(conf, 5),
                    'class_name': class_name,
                    'extracted_text': text_to_display
                })

            image_with_boxes = draw_boxes(image.copy(), detections)

            st.image(image_with_boxes, caption='Image with Bounding Boxes', use_column_width=True)
            st.subheader('Detection Results')
            detections_table = {
                'Class': [det["class_name"] for det in detections],
                'Confidence': [det["confidence"] for det in detections],
                'Extracted_text': [det["extracted_text"] for det in detections],
                'Bounding Box': [det["bbox"] for det in detections]
            }
            st.table(detections_table)
            # path = r"C:\Users\yassi\PycharmProjects\PfeProject\savedData\detections.json"

        if valid:
            if st.button('Save to json file '):
                save_to_json_file(detections, uploaded_image.name)


def main2():
    st.title("Object Detection from invoices")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    # Radio button to select language for EasyOCR
    language = st.radio("Select language for OCR:", ("en", "fr", "ar"))

    if 'image' not in st.session_state:
        st.session_state.image = None

    if uploaded_image is not None:
        st.session_state.image = Image.open(uploaded_image)

    if st.session_state.image is not None:
        model = load_model()
        image = st.session_state.image
        st.image(image, caption='Uploaded Image', use_column_width=True)

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

                preprocessed_cropped_imag = preprocess_cropped_image(cropped_image)

                cropped_text = perform_ocr_on_image(preprocessed_cropped_imag, language.lower())
                text_to_display = '\n'.join(cropped_text)

                detections.append({
                    'bbox': bbox,
                    'confidence': round(conf, 5),
                    'class_name': class_name,
                    'extracted_text': text_to_display
                })
            col1, col2 = st.columns([1, 1])
            with col1:
                image_with_boxes = draw_boxes(image.copy(), detections)
                st.image(image_with_boxes, caption='Image with Bounding Boxes', use_column_width=True)
            with col2:
                # Display text areas for modifying extracted text
                st.subheader('Extracted Text')
                for det in detections:
                    num_lines = det['extracted_text'].count('\n')
                    height = min(200, max(50, num_lines * 20))
                    width = min(800, max(200, len(det['extracted_text']) * 10))
                    st.markdown(
                        f"""
                        <div style="display: flex; width: 100%;">
                            <div style="margin-right: 10px; width: 150px;">{det['class_name']} (Confidence: {det['confidence']})</div>
                            <div style="flex: 1;">
                                <textarea style="width: {width}px; height: {height}px;">{det['extracted_text']}</textarea>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # st.markdown(
                    #     f'<textarea style="width: {width}px; height: {height}px;">{det["extracted_text"]}</textarea>',
                    #     unsafe_allow_html=True)

                    # # Compter le nombre de lignes dans le texte
                    # num_lines = det['extracted_text'].count('\n')
                    #
                    # # Calculer la hauteur en fonction du nombre de lignes
                    # height = min(200, max(50, num_lines * 5))
                    # extracted_text_editable = st.text_area(f"{det['class_name']} (Confidence: {det['confidence']})",
                    #                                    value=det['extracted_text'], height=height)

            if st.button('Save '):
                st.subheader('Detection Results')
                detections_table = {
                    'Class': [det["class_name"] for det in detections],
                    'Confidence': [det["confidence"] for det in detections],
                    'Extracted_text': [det["extracted_text"] for det in detections],
                    'Bounding Box': [det["bbox"] for det in detections]
                }
                st.table(detections_table)

            # Save to JSON button
            if st.button('Save to JSON file'):
                save_to_json_file(detections, uploaded_image.name)


def main3():
    st.title("Object Detection from invoices")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    # Radio button to select language for EasyOCR
    language = st.radio("Select language for OCR:", ("en", "fr", "ar"))

    if 'image' not in st.session_state:
        st.session_state.image = None

    if uploaded_image is not None:
        st.session_state.image = Image.open(uploaded_image)

    if st.session_state.image is not None:
        model = load_model()
        image = st.session_state.image
        st.image(image, caption='Uploaded Image', use_column_width=True)

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

                # preprocessed_cropped_imag = preprocess_cropped_image(cropped_image)
                preprocessed_cropped_imag = cropped_image

                cropped_text = perform_ocr_on_image(preprocessed_cropped_imag, language.lower())
                text_to_display = '\t'.join(cropped_text)

                detections.append({
                    'bbox': bbox,
                    'confidence': round(conf, 5),
                    'class_name': class_name,
                    'extracted_text': text_to_display
                })

            image_with_boxes = draw_boxes(image.copy(), detections)

            st.image(image_with_boxes, caption='Image with Bounding Boxes', use_column_width=True)

            st.subheader('Detection Results')
            detections_table = {
                'Class': [det["class_name"] for det in detections],
                'Confidence': [det["confidence"] for det in detections],
                'Extracted_text': [det["extracted_text"] for det in detections],
                # 'Action': [""] * len(detections)
            }
            df = pd.DataFrame(detections_table)

            # Add modify and delete buttons using Streamlit widgets
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

            for i, det in enumerate(detections):
                with st.expander(f"Item {i + 1} : " + det['class_name']):
                    # Dropdown menu to select the class name
                    modified_class = st.selectbox(f"Class Name {i}", options=list(class_names2.keys()),
                                                  index=class_names2.get(det['class_name'], 0))

                    modified_text = st.text_area("Extracted Text", value=det['extracted_text'], key=f"text_area_{i}")
                    modified_classes.append(modified_class)
                    modified_texts.append(modified_text)

                    if st.button(f"Delete Item {i}"):
                        detections.pop(i)
                        modified_classes.pop(i)
                        modified_texts.pop(i)

            # Update DataFrame with modified entries
            df['Class'] = modified_classes
            df['Extracted_text'] = modified_texts

            # Render table with modified entries
            st.table(df)

            # Save to JSON button
            if st.button('Save to JSON file'):
                save_to_json_file(detections, uploaded_image.name)


def main4():
    st.title("Object Detection from invoices")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    # Bouton radio pour sélectionner la langue pour EasyOCR
    language = st.radio("Select language for OCR:", ("en", "fr", "ar"))

    if 'detections' not in st.session_state:
        st.session_state.detections = []

    if uploaded_image is not None:
        st.session_state.image = Image.open(uploaded_image)

    if 'image' in st.session_state and st.session_state.image is not None:
        model = load_model()
        image = st.session_state.image
        st.image(image, caption='Uploaded Image', use_column_width=True)

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
                text_to_display = '\t'.join(cropped_text)

                detections.append({
                    'bbox': bbox,
                    'confidence': round(conf, 5),
                    'class_name': class_name,
                    'extracted_text': text_to_display
                })

            st.session_state.detections = detections

        # Affichage des résultats de la détection
        if st.session_state.detections:
            image_with_boxes = draw_boxes(image.copy(), st.session_state.detections)
            st.image(image_with_boxes, caption='Image with Bounding Boxes', use_column_width=True)

            st.subheader('Detection Results')
            detections_table = {
                'Class': [det["class_name"] for det in st.session_state.detections],
                'Confidence': [det["confidence"] for det in st.session_state.detections],
                'Extracted_text': [det["extracted_text"] for det in st.session_state.detections],
            }
            df = pd.DataFrame(detections_table)

            # Modifier et supprimer les entrées
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

            for i, det in enumerate(st.session_state.detections):
                with st.expander(f"Item {i + 1} : " + det['class_name']):
                    # Dropdown pour sélectionner le nom de la classe
                    modified_class = st.selectbox(f"Class Name {i}", options=list(class_names2.keys()),
                                                  index=class_names2.get(det['class_name'], 0))

                    modified_text = st.text_area(f"Extracted Text {i}", value=det['extracted_text'],
                                                 key=f"text_area_{i}")
                    modified_classes.append(modified_class)
                    modified_texts.append(modified_text)

                    if st.button(f"Delete Item {i}"):
                        st.session_state.detections.pop(i)

            # Mise à jour du DataFrame avec les entrées modifiées
            df['Class'] = modified_classes
            df['Extracted_text'] = modified_texts

            # Affichage du tableau avec les entrées modifiées
            st.table(df)

            # Bouton pour enregistrer dans un fichier JSON
            if st.button('Save to JSON file'):
                save_to_json_file(st.session_state.detections, uploaded_image.name)


def main5():
    st.title("Object Detection from invoices")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    # Radio button to select language for EasyOCR
    language = st.radio("Select language for OCR:", ("en", "fr", "ar"))

    if 'detections' not in st.session_state:
        st.session_state.detections = []

    if uploaded_image is not None:
        st.session_state.image = Image.open(uploaded_image)

    if 'image' in st.session_state and st.session_state.image is not None:
        model = load_model()
        image = st.session_state.image
        st.image(image, caption='Uploaded Image', use_column_width=True)

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
                text_to_display = '\t'.join(cropped_text)

                detections.append({
                    'bbox': bbox,
                    'confidence': round(conf, 5),
                    'class_name': class_name,
                    'extracted_text': text_to_display
                })

            st.session_state.detections = detections

        # Display detection results
        if st.session_state.detections:
            image_with_boxes = draw_boxes(image.copy(), st.session_state.detections)
            st.image(image_with_boxes, caption='Image with Bounding Boxes', use_column_width=True)

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
                    # Dropdown to select class name
                    modified_class = st.selectbox(f"Class Name {i}", options=list(class_names2.keys()),
                                                  index=class_names2.get(det['class_name'], 0))

                    modified_text = st.text_area(f"Extracted Text {i}", value=det['extracted_text'],
                                                 key=f"text_area_{i}")
                    modified_classes.append(modified_class)
                    modified_texts.append(modified_text)

                    if st.button(f"Delete Item {i}"):
                        deleted_indices.append(i)

            # Remove elements from detections, modified_classes, and modified_texts
            for idx in reversed(deleted_indices):
                st.session_state.detections.pop(idx)
                modified_classes.pop(idx)
                modified_texts.pop(idx)

            # Update DataFrame with modified entries
            df['Class'] = modified_classes
            df['Extracted_text'] = modified_texts

            # Display table with modified entries
            st.table(df)

            # Bouton pour enregistrer dans un fichier JSON
            if st.button('Save to JSON file'):
                save_to_json_file(st.session_state.detections, uploaded_image.name)


def main6():
    st.title("Object Detection from invoices")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    # Radio button to select language for EasyOCR
    language = st.radio("Select language for OCR:", ("en", "fr", "ar"))

    if 'detections' not in st.session_state:
        st.session_state.detections = []

    if uploaded_image is not None:
        st.session_state.image = Image.open(uploaded_image)

    if 'image' in st.session_state and st.session_state.image is not None:
        model = load_model()
        image = st.session_state.image
        st.image(image, caption='Uploaded Image', use_column_width=True)

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
            image_with_boxes = draw_boxes(image.copy(), st.session_state.detections)
            st.image(image_with_boxes, caption='Image with Bounding Boxes', use_column_width=True)

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

                    if st.button(f"Delete Item {i}"):
                        deleted_indices.append(i)

            # Remove items from detections, modified_classes, and modified_texts
            for idx in reversed(deleted_indices):
                st.session_state.detections.pop(idx)
                modified_classes.pop(idx)
                modified_texts.pop(idx)

            # Update DataFrame with modified entries
            df['Class'] = modified_classes
            df['Extracted_text'] = modified_texts

            # Update detections with modified data
            for i, det in enumerate(st.session_state.detections):
                det['class_name'] = modified_classes[i]
                # print(det['class_name'])
                det['extracted_text'] = modified_texts[i]
                # print(det['extracted_text'])

            st.table(df)

            # Button for saving to JSON file
            if st.button('Save to JSON file'):
                save_to_json_file(st.session_state.detections, uploaded_image.name)


def main7():
    st.title("Object Detection from invoices")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    # Radio button to select language for EasyOCR
    language = st.radio("Select language for OCR:", ("en", "fr", "ar"))

    if 'detections' not in st.session_state:
        st.session_state.detections = []

    if uploaded_image is not None:
        st.session_state.image = Image.open(uploaded_image)

    if 'image' in st.session_state and st.session_state.image is not None:
        model = load_model()
        image = st.session_state.image
        st.image(image, caption='Uploaded Image', use_column_width=True)

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
            image_with_boxes = draw_boxes(image.copy(), st.session_state.detections)
            st.image(image_with_boxes, caption='Image with Bounding Boxes', use_column_width=True)

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

                    # Vérifiez si le bouton de suppression est cliqué
                    if st.button(f"Delete Item {i}"):
                        deleted_indices.append(i)

            # Créez une nouvelle liste pour stocker les éléments restants après la suppression
            remaining_detections = []

            # Parcourez les détections et supprimez les éléments sélectionnés
            for i, det in enumerate(st.session_state.detections):
                if i not in deleted_indices:
                    remaining_detections.append(det)

            # Mettez à jour la liste detections avec les éléments restants
            st.session_state.detections = remaining_detections

            # Mettez à jour DataFrame avec les entrées restantes
            df['Class'] = modified_classes
            df['Extracted_text'] = modified_texts

            # Affichez le DataFrame mis à jour
            st.table(df)

            # Button for saving to JSON file
            if st.button('Save to JSON file'):
                save_to_json_file(st.session_state.detections, uploaded_image.name)



def main77():
    st.title("Object Detection from invoices")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    # Radio button to select language for EasyOCR
    language = st.radio("Select language for OCR:", ("en", "fr", "ar"))

    if 'detections' not in st.session_state:
        st.session_state.detections = []

    if uploaded_image is not None:
        st.session_state.image = Image.open(uploaded_image)

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

                    # Vérifiez si le bouton de suppression est cliqué
                        if st.button(f"Delete Item {i}"):
                            deleted_indices.append(i)

                # Créez une nouvelle liste pour stocker les éléments restants après la suppression
                remaining_detections = []

                # Parcourez les détections et supprimez les éléments sélectionnés
                for i, det in enumerate(st.session_state.detections):
                    if i not in deleted_indices:
                        remaining_detections.append(det)

                # Mettez à jour la liste detections avec les éléments restants
                st.session_state.detections = remaining_detections

                # Mettez à jour DataFrame avec les entrées restantes
                df['Class'] = modified_classes
                df['Extracted_text'] = modified_texts

                # Affichez le DataFrame mis à jour
            st.table(df)

            col1, col2, col3 = st.columns([1, 0.5, 1])
            with col2:
                # Button for saving to JSON file
                if st.button('Save to JSON file'):
                    save_to_json_file(st.session_state.detections, uploaded_image.name)



if __name__ == "__main__":
    main7()






