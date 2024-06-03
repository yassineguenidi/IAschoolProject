import streamlit as st
import pandas as pd



def delete_entry(detections, index):
    del detections[index]

def main2():
    st.title("Object Detection from invoices")

    # Mocking detection results for demonstration
    detections = [
        {'class_name': 'Class 1', 'confidence': 0.7, 'extracted_text': 'Text 1'},
        {'class_name': 'Class 2', 'confidence': 0.8, 'extracted_text': 'Text 2'},
        {'class_name': 'Class 3', 'confidence': 0.9, 'extracted_text': 'Text 3'}
    ]

    st.subheader('Detection Results')
    detections_table = {
        'Class': [det["class_name"] for det in detections],
        'Confidence': [det["confidence"] for det in detections],
        'Extracted_text': [det["extracted_text"] for det in detections]
    }
    df = pd.DataFrame(detections_table)

    modified_classes = [det['class_name'] for det in detections]
    modified_texts = [det['extracted_text'] for det in detections]

    deleted = False

    for i, det in enumerate(detections):
        with st.expander(f"Entry {i + 1}"):
            modified_class = st.text_input(f"Class (Confidence: {det['confidence']})", value=det['class_name'])
            modified_text = st.text_area("Extracted Text", value=det['extracted_text'])
            if st.button(f"Delete Entry {i}"):
                del detections[i]
                del modified_classes[i]
                del modified_texts[i]
                deleted = True

    if deleted:
        # Rebuild DataFrame with modified entries
        df = pd.DataFrame({
            'Class': modified_classes,
            'Extracted_text': modified_texts,
            'Confidence': [det['confidence'] for det in detections]
        })

    # Render table with modified entries
    st.table(df)


if __name__ == "__main__":
    main2()
