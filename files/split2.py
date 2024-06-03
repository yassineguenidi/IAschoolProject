import os
import shutil
import random

# Define paths
final_dir = r'C:\Users\yassi\PycharmProjects\PfeProject\final'
train_dir = r'C:\Users\yassi\PycharmProjects\PfeProject\train_data'
test_dir = r'C:\Users\yassi\PycharmProjects\PfeProject\test_data'
valid_dir = r'C:\Users\yassi\PycharmProjects\PfeProject\valid_data'

# Create folders if they don't exist
for folder in [train_dir, test_dir, valid_dir]:
    os.makedirs(os.path.join(folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'labels'), exist_ok=True)

# List all image files in the final images directory
image_files = os.listdir(os.path.join(final_dir, 'images'))

# Shuffle the list of image files while maintaining correspondence with labels
random.shuffle(image_files)

# Define proportions for train, test, and validation sets
train_split = 0.7
test_split = 0.1
valid_split = 0.2

# Calculate the number of images for each split
num_images = len(image_files)
num_train = int(train_split * num_images)
num_test = int(test_split * num_images)
num_valid = num_images - num_train - num_test

# Split the dataset
train_images = image_files[:num_train]
test_images = image_files[num_train:num_train + num_test]
valid_images = image_files[num_train + num_test:]

# Function to move images and corresponding labels to their respective folders
def move_files(images, source_dir, dest_dir):
    for img in images:
        img_path = os.path.join(source_dir, 'images', img)
        label_path = os.path.join(source_dir, 'labels', img.replace('.jpg', '.txt'))  # Assuming label files have the same name as image files but with .txt extension
        shutil.move(img_path, os.path.join(dest_dir, 'images'))
        shutil.move(label_path, os.path.join(dest_dir, 'labels'))

# Move images and labels to their respective folders
move_files(train_images, final_dir, train_dir)
move_files(test_images, final_dir, test_dir)
move_files(valid_images, final_dir, valid_dir)

print("Dataset split and shuffled successfully!")


# 1_png_jpg.rf.910a42237590b8324f51319b05796302
# 1_png_jpg.rf.910a42237590b8324f51319b05796302
