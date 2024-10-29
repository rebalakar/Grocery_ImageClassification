# eda_combined.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

# Set up directories
data_dir = 'data/SKU110K_fixed'
annotations_dir = os.path.join(data_dir, 'annotations')
images_dir = os.path.join(data_dir, 'images')

# Create a directory to save plots
plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)

# Load annotations
annotations = pd.read_csv(os.path.join(annotations_dir, 'annotations_train.csv'), header=None)

# Assign column names as per the readme
columns = ['image_name', 'x1', 'y1', 'x2', 'y2', 'class', 'image_width', 'image_height']
annotations.columns = columns


print("Total number of annotations after merging:", len(annotations))
print("Total number of unique images:", annotations['image_name'].nunique())

# Basic statistics
def basic_stats(annotations):
    images = annotations['image_name'].unique()
    num_images = len(images)
    num_annotations = len(annotations)
    print(f"Number of images: {num_images}")
    print(f"Number of annotations (objects): {num_annotations}")
    # Objects per image
    objects_per_image = annotations.groupby('image_name').size()
    mean_objects = objects_per_image.mean()
    median_objects = objects_per_image.median()
    max_objects = objects_per_image.max()
    min_objects = objects_per_image.min()
    print(f"Average number of objects per image: {mean_objects:.2f}")
    print(f"Median number of objects per image: {median_objects}")
    print(f"Maximum number of objects in an image: {max_objects}")
    print(f"Minimum number of objects in an image: {min_objects}")

    plt.figure(figsize=(10,6))
    sns.histplot(objects_per_image, bins=50, kde=False)
    plt.title('Distribution of Number of Objects per Image')
    plt.xlabel('Number of Objects')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'objects_per_image_distribution.png'))
    # plt.show()
    plt.close()

# Perform basic statistics
basic_stats(annotations)

# Compute bounding box attributes
annotations['bbox_width'] = annotations['x2'] - annotations['x1']
annotations['bbox_height'] = annotations['y2'] - annotations['y1']
annotations['bbox_area'] = annotations['bbox_width'] * annotations['bbox_height']
annotations['aspect_ratio'] = annotations['bbox_width'] / annotations['bbox_height']

# Print basic statistics of bounding boxes
print("\nBounding Box Width Statistics:")
print(annotations['bbox_width'].describe())
print("\nBounding Box Height Statistics:")
print(annotations['bbox_height'].describe())
print("\nBounding Box Area Statistics:")
print(annotations['bbox_area'].describe())
print("\nBounding Box Aspect Ratio Statistics:")
print(annotations['aspect_ratio'].describe())

# Plot distributions
def plot_distribution(data, column, title, xlabel, filename, log_scale=False):
    plt.figure(figsize=(10,6))
    if log_scale:
        data = np.log(data + 1e-6)  # Add small value to avoid log(0)
        xlabel = 'Log ' + xlabel
    sns.histplot(data, bins=50, kde=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename))
    # plt.show()
    plt.close()

# Plot bounding box width distribution
plot_distribution(annotations['bbox_width'], 'bbox_width', 'Distribution of Bounding Box Width', 'Bounding Box Width (pixels)', 'bbox_width_distribution.png')

# Plot bounding box height distribution
plot_distribution(annotations['bbox_height'], 'bbox_height', 'Distribution of Bounding Box Height', 'Bounding Box Height (pixels)', 'bbox_height_distribution.png')

# Plot bounding box area distribution (log scale)
plot_distribution(annotations['bbox_area'], 'bbox_area', 'Distribution of Bounding Box Area', 'Bounding Box Area', 'bbox_area_distribution_log.png', log_scale=True)

# Plot aspect ratio distribution
plot_distribution(annotations['aspect_ratio'], 'aspect_ratio', 'Distribution of Bounding Box Aspect Ratios', 'Aspect Ratio (Width / Height)', 'aspect_ratio_distribution.png')

# Class frequency analysis
def class_frequency(annotations):
    class_counts = annotations['class'].value_counts()
    num_classes = len(class_counts)
    print(f"\nNumber of unique classes: {num_classes}")
    print("Class counts:")
    print(class_counts)

    plt.figure(figsize=(12,6))
    sns.barplot(x=class_counts.index.astype(str), y=class_counts.values)
    plt.title('Class Frequency Distribution')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'class_frequency_distribution.png'))
    # plt.show()
    plt.close()

# Perform class frequency analysis
if annotations['class'].nunique() > 1:
    class_frequency(annotations)
else:
    print("\nOnly one class present in the dataset.")

# Analyze object sizes relative to image sizes
annotations['relative_width'] = annotations['bbox_width'] / annotations['image_width']
annotations['relative_height'] = annotations['bbox_height'] / annotations['image_height']
annotations['relative_area'] = annotations['bbox_area'] / (annotations['image_width'] * annotations['image_height'])

print("\nRelative Bounding Box Width Statistics:")
print(annotations['relative_width'].describe())
print("\nRelative Bounding Box Height Statistics:")
print(annotations['relative_height'].describe())
print("\nRelative Bounding Box Area Statistics:")
print(annotations['relative_area'].describe())

# Plot relative bounding box area distribution
plot_distribution(annotations['relative_area'], 'relative_area', 'Distribution of Relative Bounding Box Area', 'Relative Bounding Box Area', 'relative_bbox_area_distribution.png')

# Correlation between bounding box dimensions and image dimensions
print("\nCorrelation between Bounding Box Width and Image Width:")
print(annotations['bbox_width'].corr(annotations['image_width']))

print("\nCorrelation between Bounding Box Height and Image Height:")
print(annotations['bbox_height'].corr(annotations['image_height']))

# Visualize sample images with bounding boxes
def plot_image_with_bboxes(image_name, annotations_df, images_dir, save_dir):
    image_path = os.path.join(images_dir, image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image {image_name} not found.")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes = annotations_df[annotations_df['image_name'] == image_name]
    for idx, row in bboxes.iterrows():
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 2)
        # Optionally, add class labels if multiple classes exist
        if annotations_df['class'].nunique() > 1:
            class_label = str(row['class'])
            cv2.putText(image, class_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    plt.figure(figsize=(12,8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Image: {image_name}")
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{os.path.splitext(image_name)[0]}_bboxes.png")
    plt.savefig(save_path)
    # plt.show()
    plt.close()

# Create a directory to save sample images with bounding boxes
sample_images_dir = os.path.join(plots_dir, 'sample_images')
os.makedirs(sample_images_dir, exist_ok=True)

# Display and save sample images with bounding boxes
sample_images = np.random.choice(annotations['image_name'].unique(), size=5, replace=False)
for img_name in sample_images:
    plot_image_with_bboxes(img_name, annotations, images_dir, sample_images_dir)

print("\nEDA completed. Plots are saved in the 'plots' directory.")
