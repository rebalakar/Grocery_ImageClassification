# sample_dataset.py

import os
import pandas as pd
import numpy as np
import shutil

# Set up directories
data_dir = 'data/SKU110K_fixed'
annotations_dir = os.path.join(data_dir, 'annotations')
images_dir = os.path.join(data_dir, 'images')  # Directory where original images are stored
COLUMNS_TO_SAVE = ['image_name', 'x1', 'y1', 'x2', 'y2', 'class', 'image_width', 'image_height']

# Load only the original train annotations
train_annotations = pd.read_csv(os.path.join(annotations_dir, 'annotations_train.csv'), header=None)

# Assign column names
columns = ['image_name', 'x1', 'y1', 'x2', 'y2', 'class', 'image_width', 'image_height']
train_annotations.columns = columns

# Use only train_annotations
all_annotations = train_annotations.copy()

# Compute bounding box attributes
all_annotations['bbox_width'] = all_annotations['x2'] - all_annotations['x1']
all_annotations['bbox_height'] = all_annotations['y2'] - all_annotations['y1']
all_annotations['bbox_area'] = all_annotations['bbox_width'] * all_annotations['bbox_height']
all_annotations['aspect_ratio'] = all_annotations['bbox_width'] / all_annotations['bbox_height']

# Get unique images from the annotations
all_images = all_annotations['image_name'].unique()

# Determine sample size
sample_size = 2000  # Adjust as needed

# Set random seed for repeatability
np.random.seed(1)

# Randomly sample images
sampled_images = np.random.choice(all_images, size=sample_size, replace=False)

# Extract annotations for sampled images
sampled_annotations = all_annotations[all_annotations['image_name'].isin(sampled_images)]

# Ensure that object height/width diversity is maintained
# Since we are sampling images randomly from the original dataset, which already has a diverse range of object sizes,
# the sampled dataset will naturally inherit this diversity.

# However, to confirm that the diversity is maintained, we can compare the distributions of bounding box widths and heights
# in the sampled dataset to those in the original dataset (optional).

# Split into training, validation, and test sets
train_split = 0.7
val_split = 0.2
test_split = 0.1

# Shuffle the sampled images
np.random.shuffle(sampled_images)

# Calculate the number of images for each split
num_train = int(len(sampled_images) * train_split)
num_val = int(len(sampled_images) * val_split)
num_test = len(sampled_images) - num_train - num_val

# Get image lists for each split
sampled_train_images = sampled_images[:num_train]
sampled_val_images = sampled_images[num_train:num_train+num_val]
sampled_test_images = sampled_images[num_train+num_val:]

# Create annotations for each split
sampled_train_annotations = sampled_annotations[sampled_annotations['image_name'].isin(sampled_train_images)]
sampled_val_annotations = sampled_annotations[sampled_annotations['image_name'].isin(sampled_val_images)]
sampled_test_annotations = sampled_annotations[sampled_annotations['image_name'].isin(sampled_test_images)]

# Create directory for sampled annotations
sampled_data_dir = 'sampled_annotations'
sampled_images_dir = 'sampled_images'
os.makedirs(sampled_data_dir, exist_ok=True)
os.makedirs(sampled_images_dir, exist_ok=True)

# Create subdirectories for images in each split
train_images_dir = os.path.join(sampled_images_dir, 'train')
val_images_dir = os.path.join(sampled_images_dir, 'val')
test_images_dir = os.path.join(sampled_images_dir, 'test')
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)

# Save annotations to CSV files
def save_annotations(annotations_df, file_path):
    annotations_df = annotations_df.copy()
    # Select only the required columns
    annotations_df = annotations_df[COLUMNS_TO_SAVE]
    # Save to CSV without headers
    annotations_df.to_csv(file_path, index=False, header=False)

# Save the annotations for each split
save_annotations(sampled_train_annotations, os.path.join(sampled_data_dir, 'train.csv'))
save_annotations(sampled_val_annotations, os.path.join(sampled_data_dir, 'val.csv'))
save_annotations(sampled_test_annotations, os.path.join(sampled_data_dir, 'test.csv'))

# Function to copy sampled images to their respective directories
def copy_images(image_list, dest_dir):
    for image_name in image_list:
        source_path = os.path.join(images_dir, image_name)
        dest_path = os.path.join(dest_dir, image_name)
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)
        else:
            print(f"Warning: Image {image_name} not found in the source directory.")

# Copy images for each split
copy_images(sampled_train_images, train_images_dir)
copy_images(sampled_val_images, val_images_dir)
copy_images(sampled_test_images, test_images_dir)

print("Sampled annotations are saved in the 'sampled_annotations' directory.")
print("Sampled images are saved in the 'sampled_images/train', 'sampled_images/val', and 'sampled_images/test' directories.")
