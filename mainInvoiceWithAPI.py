import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from io import BytesIO
import json
from pdf2image import convert_from_bytes

def draw_boxes(image, detections):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for det in detections:
        bbox = det['bbox']
        class_name = det['class_name']
        confidence = det['confidence']
        draw.rectangle(bbox, outline='green', width=2)
        draw.text((bbox[0]-5, bbox[1]-20), str(class_name) + ' ' + str(round(confidence, 3)), fill='green', font=font)

    return image

def save_to_json_file(detections, image_filename):
    image_name = image_filename.split('.')[0]
    json_filename = f"savedData/invoice/{image_name}.json"

    with open(json_filename, 'w', encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False, indent=4)
    st.success(f"Detection results saved to {json_filename}")


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

#
# def mainInvoiceNot():
#     st.title("Object Detection from Invoices")
#
#     uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
#     language = st.radio("Select language for OCR:", ("en", "fr", "ar"))
#
#     if 'detections' not in st.session_state:
#         st.session_state.detections = []
#
#     if uploaded_image is not None:
#         st.session_state.image = Image.open(uploaded_image)
#
#     if 'image' in st.session_state and st.session_state.image is not None:
#         image = st.session_state.image
#         col1, col2, col3 = st.columns([1.75, 5, 0.5])
#         with col2:
#             st.image(image, caption='Uploaded Image')
#
#         col1, col2, col3 = st.columns([1.1, 0.3, 1])
#         with col2:
#             if st.button('Detect Objects'):
#                 with st.spinner('Detecting objects...'):
#                     img_buffer = BytesIO()
#                     image.save(img_buffer, format='PNG')
#                     img_buffer.seek(0)
#
#                     files = {'file': ('image.png', img_buffer, 'image/png')}
#                     response = requests.post("http://127.0.0.1:8084/detect-objects/", files=files)
#                     response_data = response.json()
#
#                     detections = response_data["detections"]
#                     for det in detections:
#                         bbox = det['bbox']
#                         class_name = det['class_name']
#                         cropped_image = image.crop(bbox)
#
#                         img_buffer = BytesIO()
#                         cropped_image.save(img_buffer, format='PNG')
#                         img_buffer.seek(0)
#
#                         files = {'file': ('cropped_image.png', img_buffer, 'image/png')}
#                         data = {'language': language.lower()}
#                         ocr_response = requests.post("http://127.0.0.1:8084/perform-ocr/", files=files, data=data)
#                         ocr_response_data = ocr_response.json()
#
#                         text_to_display = ' '.join(ocr_response_data["text"])
#
#                         det['extracted_text'] = text_to_display
#
#                     st.session_state.detections = detections
#
#         if st.session_state.detections:
#             col1, col2 = st.columns([1, 1])
#             with col1:
#                 st.subheader('Image with BBOX')
#                 st.subheader('')
#                 st.subheader('')
#                 image_with_boxes = draw_boxes(image.copy(), st.session_state.detections)
#                 st.image(image_with_boxes, caption='Image with Bounding Boxes', use_column_width=True)
#             with col2:
#                 st.subheader('Detection Results')
#                 detections_table = {
#                     'Class': [det["class_name"] for det in st.session_state.detections],
#                     'Confidence': [det["confidence"] for det in st.session_state.detections],
#                     'Extracted_text': [det["extracted_text"] for det in st.session_state.detections],
#                 }
#                 df = pd.DataFrame(detections_table)
#
#                 modified_classes = []
#                 modified_texts = []
#                 class_names2 = {
#                     "Discount_Percentage": 0,
#                     "Due_Date": 1,
#                     "Email_Client": 2,
#                     "Name_Client": 3,
#                     "Products": 4,
#                     "Remise": 5,
#                     "Subtotal": 6,
#                     "Tax": 7,
#                     "Tax_Precentage": 8,
#                     "Tel_Client": 9,
#                     "billing address": 10,
#                     "header": 11,
#                     "invoice date": 12,
#                     "invoice number": 13,
#                     "shipping address": 14,
#                     "total": 15
#                 }
#
#                 deleted_indices = []
#                 for i, det in enumerate(st.session_state.detections):
#                     with st.expander(f"Item {i + 1} : " + det['class_name']):
#                         modified_class = st.selectbox(f"Class Name {i}", options=list(class_names2.keys()),
#                                                       index=class_names2.get(det['class_name'], 0))
#
#                         modified_text = st.text_area(f"Extracted Text {i}", value=det['extracted_text'],
#                                                      key=f"text_area_{i}")
#                         modified_classes.append(modified_class)
#                         modified_texts.append(modified_text)
#
#                         if st.button(f"Delete Item {i}"):
#                             deleted_indices.append(i)
#
#                 remaining_detections = [det for i, det in enumerate(st.session_state.detections) if i not in deleted_indices]
#
#                 st.session_state.detections = remaining_detections
#
#                 df['Class'] = modified_classes
#                 df['Extracted_text'] = modified_texts
#
#             st.table(df)
#
#             products_extracted_text = df.loc[df['Class'] == 'Products', 'Extracted_text']
#             header_extracted_text = df.loc[df['Class'] == 'header', 'Extracted_text']
#
#             col1, col2, col3 = st.columns([1, 0.5, 1])
#             with col2:
#                 if st.button('Save to JSON file'):
#                     save_to_json_file(df.to_dict(orient='records'), uploaded_image.name)
#


def mainInvoice():
    st.title("Object Detection from Invoices")

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
        # model = load_model()
        image = st.session_state.image
        col1, col2, col3 = st.columns([1.75, 5, 0.5])
        with col2:
            st.image(image, caption='Uploaded Image')

        col1, col2, col3 = st.columns([1.1, 0.3, 1])
        with col2:
            if st.button('Detect Objects'):
                with st.spinner('Detecting objects...'):
                    img_buffer = BytesIO()
                    image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)

                    files = {'file': ('image.png', img_buffer, 'image/png')}
                    response = requests.post("http://127.0.0.1:8084/detect-objects/", files=files)
                    response_data = response.json()

                    detections = response_data["detections"]
                    for det in detections:
                        bbox = det['bbox']
                        class_name = det['class_name']
                        cropped_image = image.crop(bbox)

                        img_buffer = BytesIO()
                        cropped_image.save(img_buffer, format='PNG')
                        img_buffer.seek(0)

                        files = {'file': ('cropped_image.png', img_buffer, 'image/png')}
                        data = {'language': language.lower()}
                        ocr_response = requests.post("http://127.0.0.1:8084/perform-ocr/", files=files, data=data)
                        ocr_response_data = ocr_response.json()

                        text_to_display = ' '.join(ocr_response_data["text"])

                        det['extracted_text'] = text_to_display

                    st.session_state.detections = detections

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
                        modified_class = st.selectbox(f"Class Name {i}", options=list(class_names2.keys()),
                                                      index=class_names2.get(det['class_name'], 0))

                        modified_text = st.text_area(f"Extracted Text {i}", value=det['extracted_text'],
                                                     key=f"text_area_{i}")
                        modified_classes.append(modified_class)
                        modified_texts.append(modified_text)

                        if st.button(f"Delete Item {i}"):
                            deleted_indices.append(i)

                remaining_detections = [det for i, det in enumerate(st.session_state.detections) if i not in deleted_indices]

                st.session_state.detections = remaining_detections

                df['Class'] = modified_classes
                df['Extracted_text'] = modified_texts

            st.table(df)

            products_extracted_text = df.loc[df['Class'] == 'Products', 'Extracted_text']
            header_extracted_text = df.loc[df['Class'] == 'header', 'Extracted_text']

            col1, col2, col3 = st.columns([1, 0.5, 1])
            with col2:
                if st.button('Save to JSON file'):
                    save_to_json_file(df.to_dict(orient='records'), uploaded_image.name)


if __name__ == "__main__":
    mainInvoice()
