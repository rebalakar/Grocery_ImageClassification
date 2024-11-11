import os
import pandas as pd

# Define paths
annotations_dir = './sampled_annotations'
images_dir = './data/SKU110K_fixed/images'
output_base_dir = './data/SKU110_txt'


# Ensure output base directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Define column names for your CSV files
column_names = ['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name', 'image_width', 'image_height']

# Process CSV files (train, val, test)
csv_files = {
    'train': 'train.csv',
    'val': 'val.csv',
    'test': 'test.csv'
}

for dataset_type, csv_file in csv_files.items():
    csv_path = os.path.join(annotations_dir, csv_file)
    output_dir = os.path.join(output_base_dir, dataset_type)  # Create subdirectory for each dataset type
    os.makedirs(output_dir, exist_ok=True)

    # Load the CSV file into a pandas DataFrame without a header row
    df = pd.read_csv(csv_path, names=column_names, header=None)

    # Print column names to verify
    print(f"Columns in {csv_file}: {df.columns}")

    # Iterate over each row in the CSV file
    for _, row in df.iterrows():
        # Access the image name and bounding box information using the defined column names
        image_name = row['image_name']
        x_min, y_min, x_max, y_max = row['x_min'], row['y_min'], row['x_max'], row['y_max']
        class_id = 0  # Assuming a single class 'object'

        # Calculate YOLO format (x_center, y_center, width, height)
        image_width = row['image_width']
        image_height = row['image_height']
        x_center = ((x_min + x_max) / 2) / image_width
        y_center = ((y_min + y_max) / 2) / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        # Create label file path in the respective directory
        label_file_name = os.path.splitext(image_name)[0] + '.txt'
        label_file_path = os.path.join(output_dir, label_file_name)

        # Write the annotation to a txt file
        with open(label_file_path, 'w') as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

print("Annotation conversion complete.")
