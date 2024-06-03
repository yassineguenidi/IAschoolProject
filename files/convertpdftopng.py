import os
from PyPDF2 import PdfFileReader, PdfReader
from PIL import Image

def pdf_to_images(pdf_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all PDF files in the input folder
    for filename in os.listdir(pdf_path):
        if filename.endswith('.pdf'):
            pdf_file_path = os.path.join(pdf_path, filename)
            output_file_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')

            # Convert PDF to images
            images = []
            with WandImage(filename=pdf_file_path, resolution=300) as img:
                img.compression_quality = 99
                img.format = 'png'
                img_background = WandImage(width=img.width, height=img.height, background="white")
                img_background.composite(img, 0, 0)
                img_blob = img_background.make_blob('png')
                images.append(img_blob)

            # If there are multiple pages, concatenate them into one image
            if len(images) > 1:
                with open(output_file_path, 'wb') as f:
                    f.write(b"".join(images))
            else:
                with open(output_file_path, 'wb') as f:
                    f.write(images[0])

            print(f"{filename} converted to image(s) and saved as {output_file_path}")

            # # Convert PDF to images
            # images = []
            # with open(pdf_file_path, 'rb') as file:
            #     pdf_reader = PdfReader(file)
            #     for page_num in range(len(pdf_reader.pages)):
            #         page = pdf_reader.pages[page_num]
            #         image = page.to_pil()
            #         images.append(image)
            #
            # # If there are multiple pages, concatenate them into one image
            # if len(images) > 1:
            #     images[0].save(output_file_path, save_all=True, append_images=images[1:])
            # else:
            #     images[0].save(output_file_path)
            #
            # print(f"{filename} converted to image(s) and saved as {output_file_path}")

# Example usage
input_folder = r"C:\Users\yassi\PycharmProjects\PfeProject\PFE 2023"
output_folder = r"C:\Users\yassi\PycharmProjects\PfeProject\pfe images"
pdf_to_images(input_folder, output_folder)
