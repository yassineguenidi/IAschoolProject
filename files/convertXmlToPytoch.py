# import os
# import xml.etree.ElementTree as ET
#
# def convert_voc_to_yolo(xml_path, output_path, classes):
#     with open(output_path, 'w') as output_file:
#         for xml_file in os.listdir(xml_path):
#             if xml_file.endswith('.xml'):
#                 xml_file_path = os.path.join(xml_path, xml_file)
#                 tree = ET.parse(xml_file_path)
#                 root = tree.getroot()
#
#                 image_width = int(root.find('size').find('width').text)
#                 image_height = int(root.find('size').find('height').text)
#
#                 for obj in root.findall('object'):
#                     class_name = obj.find('name').text
#                     if class_name not in classes:
#                         continue
#
#                     class_id = classes.index(class_name)
#
#                     bbox = obj.find('bndbox')
#                     xmin = float(bbox.find('xmin').text)
#                     ymin = float(bbox.find('ymin').text)
#                     xmax = float(bbox.find('xmax').text)
#                     ymax = float(bbox.find('ymax').text)
#
#                     x_center = (xmin + xmax) / 2.0
#                     y_center = (ymin + ymax) / 2.0
#                     width = xmax - xmin
#                     height = ymax - ymin
#
#                     x_center /= image_width
#                     y_center /= image_height
#                     width /= image_width
#                     height /= image_height
#
#                     output_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
#
# # Example usage:
# xml_folder = r'C:\Users\yassi\PycharmProjects\PfeProject\pytorchAnn'
# output_folder = r'C:\Users\yassi\PycharmProjects\PfeProject\pytorchLabels'
# classes = ['delivrance', 'naissance']  # Specify the class names in the same order as YOLO model
#
# convert_voc_to_yolo(xml_folder, os.path.join(output_folder, 'annotations.txt'), classes)


import os
import xml.etree.ElementTree as ET


def convert_voc_to_yolo(xml_path, output_folder, classes):
    for xml_file in os.listdir(xml_path):
        if xml_file.endswith('.xml'):
            xml_file_path = os.path.join(xml_path, xml_file)
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            image_width = int(root.find('size').find('width').text)
            image_height = int(root.find('size').find('height').text)

            output_file_path = os.path.join(output_folder, xml_file.replace('.xml', '.txt'))

            with open(output_file_path, 'w') as output_file:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name not in classes:
                        continue

                    class_id = classes.index(class_name)

                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)

                    x_center = (xmin + xmax) / 2.0
                    y_center = (ymin + ymax) / 2.0
                    width = xmax - xmin
                    height = ymax - ymin

                    x_center /= image_width
                    y_center /= image_height
                    width /= image_width
                    height /= image_height

                    output_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


# # Example usage:
xml_folder = r'C:\Users\yassi\PycharmProjects\PfeProject\recto\labelsXml'
output_folder = r'C:\Users\yassi\PycharmProjects\PfeProject\recto\labels'

classes = ['naissance']

convert_voc_to_yolo(xml_folder, output_folder, classes)
