from ultralytics import YOLO
from clearml import Task, Dataset
import torch
import tempfile
import os

# Initialize ClearML task
task = Task.init(project_name='5CCSAGAP Group D', task_name='YOLOv5 Training and Tracking')

task.connect_configuration({"requirements_file": "requirements.txt"})

# Enqueue the task for remote execution
task.execute_remotely(queue_name='default', exit_process=True)


try:
    import numpy as np
    print("NumPy version:", np.__version__)
except ImportError:
    print("NumPy is not installed.")


# Get the dataset from ClearML
dataset = Dataset.get(dataset_name='Sampled Images', dataset_project='5CCSAGAP Group D')
dataset_path = dataset.get_local_copy()

# Fetch the 'data_template.yaml' artifact
artifact_task = Task.get_task(project_name='5CCSAGAP Group D', task_name='Upload Data Template')
artifact_path = artifact_task.artifacts['data_template.yaml'].get_local_copy()

# Read the data_template.yaml content
with open(artifact_path, 'r') as f:
    data_yaml_content = f.read()

# Replace the placeholder with the actual dataset path
data_yaml_content = data_yaml_content.replace('{dataset_path}', dataset_path)

# Write the updated data.yaml to a temporary file
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
    tmp.write(data_yaml_content)
    data_yaml_path = tmp.name

# Load a YOLOv5 model (e.g., pretrained on COCO)
model = YOLO('yolov5s.pt')  

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set augmentation parameters
augmentation_params = {
    'hsv_h': 0.03,       # Hue augmentation (fraction)
    'hsv_s': 0.8,        # Saturation augmentation (fraction)
    'hsv_v': 0.5,        # Value augmentation (fraction)
    'degrees': 20,       # Image rotation (degrees)
    'translate': 0.1,    # Image translation (fraction)
    'scale': 0.0,        # Image scaling (fraction)
    'shear': 0.0,        # Image shearing (degrees)
    'perspective': 0.1,  # Perspective transformation (fraction)
    'flipud': 0.5,       # Vertical flip probability
    'fliplr': 0.5,       # Horizontal flip probability
    'mosaic': 1.0,       # Mosaic augmentation probability
    'mixup': 0.2,        # Image mixup probability
    'copy_paste': 0.1    # Segment copy-paste probability
}

# Log augmentation parameters to ClearML
task.connect(augmentation_params)

# Train the model
model.train(
    data=data_yaml_path,   # Path to the YAML file with dataset information
    epochs=16,             # Number of epochs to train for (adjust based on training needs)
    imgsz=640,             # Image size
    batch=16,              # Batch size
    device=device,         # Use 'cpu' or 'cuda' (if you have a GPU)
    # optimizer='AdamW',   # Use AdamW for regularization
    **augmentation_params  # Unpack the augmentation parameters
)

# Save the trained model
model.save('./yolov5_trained_model.pt')

# Validate the model
print("Validating the model...")
metrics_val = model.val()  # Validate the model performance on the validation set
print("Validation Metrics:", metrics_val)

# Test the model
print("Testing the model...")
metrics_test = model.val(split='test')  # Validate the model performance on the test set
print("Test Metrics:", metrics_test)

# Inference (optional)
# sample_image_path = "./images/sample_image.jpg"  # Replace with a valid test image path
# results = model.predict(source=sample_image_path, save=True)
# print(results)  # Print predictions