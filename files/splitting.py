import os
import shutil
import random

# Define paths
dataset_dir = r'C:\Users\yassi\PycharmProjects\PfeProject\final\images'
train_dir = r'C:\Users\yassi\PycharmProjects\PfeProject\train_data'
test_dir = r'C:\Users\yassi\PycharmProjects\PfeProject\test_data'
valid_dir = r'C:\Users\yassi\PycharmProjects\PfeProject\valid_data'

# Create folders if they don't exist
for folder in [train_dir, test_dir, valid_dir]:
    os.makedirs(folder, exist_ok=True)

# List all image files in the dataset directory
image_files = [file for file in os.listdir(dataset_dir) if file.endswith('.jpg') or file.endswith('.png')]

# Shuffle the list of image files
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

# Function to move images to their respective folders
def move_images(images, destination):
    for img in images:
        shutil.move(os.path.join(dataset_dir, img), os.path.join(destination, img))

# Move images to their respective folders
move_images(train_images, train_dir)
move_images(test_images, test_dir)
move_images(valid_images, valid_dir)

print("Dataset split and shuffled successfully!")
