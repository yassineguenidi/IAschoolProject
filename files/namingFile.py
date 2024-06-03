import os
import os
import fitz  # PyMuPDF
from PIL import Image
# def rename_files(input_folder, output_folder):
#     # Create the output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # List all files in the input folder
#     files = os.listdir(input_folder)
#
#     # Sort files to ensure consistent numbering
#     files.sort()
#
#     # Iterate over each file and rename it sequentially
#     for i, filename in enumerate(files, start=1):
#         # Construct the input and output paths
#         input_path = os.path.join(input_folder, filename)
#         output_path = os.path.join(output_folder, f"{i}.png")
#
#         # Rename the file
#         os.rename(input_path, output_path)
#         print(f"Renamed {filename} to {i}.png")
#
# if __name__ == "__main__":
#     input_folder = r"C:\Users\yassi\PycharmProjects\PfeProject\images"
#     output_folder = r"C:\Users\yassi\PycharmProjects\PfeProject\images sorted"
#     rename_files(input_folder, output_folder)
#




def pdf_folder_to_png(input_folder, output_folder, dpi=300):

        # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

        # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
                # Path to input PDF file
            input_pdf_path = os.path.join(input_folder, filename)

                # Open the PDF file
            pdf_document = fitz.open(input_pdf_path)

                # Iterate through each page in the PDF
            for page_number in range(len(pdf_document)):
                    # Get the page
                page = pdf_document.load_page(page_number)

                    # Render the page to an image
                image = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))

                    # Create a PIL image object
                pil_image = Image.frombytes("RGB", [image.width, image.height], image.samples)

                    # Save the PIL image as PNG
                png_filename = os.path.splitext(filename)[0] + f'_page{page_number + 1}.png'
                output_png_path = os.path.join(output_folder, png_filename)
                pil_image.save(output_png_path, format="PNG")
                print(f"{filename} - Page {page_number + 1} saved as {png_filename}")

                # Close the PDF file
            pdf_document.close()


import os

def rename_images(input_folder, output_folder, new_base_name):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize counter
    counter = 0

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            old_file_path = os.path.join(input_folder, filename)
            new_file_name = f"{new_base_name}_{counter}.png"
            new_file_path = os.path.join(output_folder, new_file_name)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed {filename} to {new_file_name}")
            counter += 1

# # Example usage
# input_folder = r"C:\Users\yassi\PycharmProjects\PfeProject\pfe images 1"
# output_folder = r"C:\Users\yassi\PycharmProjects\PfeProject\pfe images 2"
# new_base_name = "new_image"
# rename_images(input_folder, output_folder, new_base_name)

    # Example usage:
input_folder = r"C:\Users\yassi\PycharmProjects\PfeProject\PFE 2023"
output_folder = r"C:\Users\yassi\PycharmProjects\PfeProject\pfe_2023_images"
pdf_folder_to_png(input_folder, output_folder)
