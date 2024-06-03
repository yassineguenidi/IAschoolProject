import os
import xml.etree.ElementTree as ET
import cv2


def convert_voc_to_yolo(xml_path, output_folder, classes, target_size=(640, 640)):
    for xml_file in os.listdir(xml_path):
        if xml_file.endswith('.xml'):
            xml_file_path = os.path.join(xml_path, xml_file)
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            image_width = int(root.find('size').find('width').text)
            image_height = int(root.find('size').find('height').text)

            # Resize the image dimensions to 640x640
            resize_factor_w = target_size[0] / image_width
            resize_factor_h = target_size[1] / image_height

            image_width = target_size[0]
            image_height = target_size[1]

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

                    # Resize bounding box coordinates
                    xmin *= resize_factor_w
                    xmax *= resize_factor_w
                    ymin *= resize_factor_h
                    ymax *= resize_factor_h

                    # Normalize bounding box coordinates
                    x_center = (xmin + xmax) / 2.0 / target_size[0]
                    y_center = (ymin + ymax) / 2.0 / target_size[1]
                    width = (xmax - xmin) / target_size[0]
                    height = (ymax - ymin) / target_size[1]

                    output_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


# Example usage:
xml_folder = r'C:\Users\yassi\PycharmProjects\PfeProject\verso final\1\labelsXml'
output_folder = r'C:\Users\yassi\PycharmProjects\PfeProject\verso final\1\labelsResized'
classes = ['delivrance', 'naissance']

convert_voc_to_yolo(xml_folder, output_folder, classes)
