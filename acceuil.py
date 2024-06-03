from PIL import Image
import streamlit as st

def principale():
    st.title("Welcome Back")
    st.subheader("This application is designed to help to extract automatically informations from invoices and Resumes:")
    espace()
    firstPart()
    espace()
    secondPart()
    espace()
    about()
    espace()
    display_form()
    espace()
    footer()

def footer():
    # Add page footer
    footer = """
        <style>
            .footer {
                font-size: 0.7em;
                text-align: center;
                padding: 1em;
            }
        </style>
        <div class="footer">
            <p>Made by Yassine GUENIDI</p>
        </div>
    """
    st.write("___")
    st.markdown(footer, unsafe_allow_html=True)

def espace():
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("------------------------------------------------------------")

def about():
    # Add an "About" section to provide more information about the application
    st.header("About")

    # Add a call to action to encourage users to get started
    st.write("To get started, simply upload a resume or an invoice and select which part of the application you'd like to use....")

def display_contact_section():
    st.header("Contact Us")
    st.write("If you have any questions or feedback about this application, please feel free to contact us at:")
    st.write("- Email: yassinegunidi99@email.com")
    st.write("- Phone: +216 22 344 203")

def display_form():

    # ---- CONTACT ----
    with st.container():
        # st.write("---")
        contact_form = """
                <style>
                    input[type=text], input[type=email], textarea {
                        width: 100%;
                        padding: 12px;
                        border: 1px solid #ccc;
                        border-radius: 4px;
                        resize: vertical;
                    }
                    label {
                        display: block;
                        font-weight: bold;
                    }
                    button[type=submit] {
                        background-color: #4CAF50;
                        color: white;
                        padding: 12px 20px;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                    }
                    button[type=submit]:hover {
                        background-color: #45a049;
                    }
                    .container {
                        border-radius: 5px;
                        background-color: #f2f2f2;
                        padding: 20px;
                    }
                </style>
                <div class="container">
                    <form action="https://formsubmit.co/yassineguenidi99@gmail.com" 
                          method="POST" autocomplete="off">
                        <input type="hidden" name="_captcha" value="false">
                        <label for="name">Nom:</label>
                        <input type="text" name="name" placeholder="Your name" required>
                        <label for="email">Email:</label>
                        <input type="email" name="email" placeholder="Your email" required>
                        <label for="message">Message:</label>
                        <textarea name="message" placeholder="Your message" required></textarea>
                        <button type="submit">Submit</button>
                    </form>
                </div>
                """
        left_column,right_column = st.columns((3.5, 0.5))
        with left_column:
            st.header("Do you have any question ?")
            st.markdown(contact_form, unsafe_allow_html=True)

def firstPart():
    with st.container():
        image_column, text_column = st.columns((1, 2))
        with image_column:
            image = Image.open(r'C:\Users\yassi\PycharmProjects\PfeProject\images\cvcv.png')
            centered_container = st.container()
            with centered_container:
                st.write("##")
                st.image(image, width=300, use_column_width=True)

        with text_column:
            st.subheader("Resume Parser")
            st.write("1. Resume Parser: Extracting important information from a candidate's resume")
            st.write(
                """
                Our resume parser now incorporates cutting-edge technologies to revolutionize candidate evaluation. 
                Leveraging advanced object detection models and OCR, our system extracts text seamlessly from resumes, 
                ensuring no detail goes unnoticed. Powered by NLP, the extracted content is meticulously analyzed 
                to identify key information such as skills, experiences, and qualifications. 
                What sets us apart is our ability to match this information with job descriptions, 
                providing invaluable insights into candidate suitability. 
                With this enhanced functionality, recruiters can swiftly and accurately identify the best candidates for any role, 
                streamlining the hiring process like never before.
                """
            )

def secondPart():
    with st.container():
        image_column, text_column = st.columns((1, 2))

        with image_column:
            image = Image.open(r'C:\Users\yassi\PycharmProjects\PfeProject\images\image_invoice.JPG')
            centered_container = st.container()
            with centered_container:
                st.write("##")
                st.image(image, width=300, use_column_width=True)

        with text_column:
            st.subheader("Invoice Parser")
            st.write("1. Invoice Parser:: Extracting important information from Invoices")
            st.write(
                """
               Our invoice data extraction process employs state-of-the-art object detection techniques coupled 
               with powerful OCR technology to streamline and automate the extraction of crucial information from invoices. 
               By utilizing object detection models, we accurately identify key regions of interest within the invoice document, 
               such as vendor details, invoice number, dates, and line items. Once these regions are identified, 
               our OCR engine swiftly extracts text from each segment, converting it into machine-readable data. 
               This process ensures the precise capture of information regardless of invoice format or layout, 
               optimizing efficiency and accuracy in invoice processing. Through the seamless integration of object detection and OCR, 
               our solution empowers businesses to digitize their invoicing workflows, saving time and resources while minimizing errors.
                 """
            )

# principale()