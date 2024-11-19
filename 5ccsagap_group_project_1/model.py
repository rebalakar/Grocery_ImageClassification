from ultralytics import YOLO
from clearml import Task, Dataset
import tempfile
import os

# Initialize ClearML task
task = Task.init(project_name='5CCSAGAP Group D', task_name='YOLOv5 Training and Tracking')

# Enqueue the task for remote execution
task.execute_remotely(queue_name='default', exit_process=True)

# Get the dataset from ClearML
dataset = Dataset.get(dataset_name='Sampled Images', dataset_project='5CCSAGAP Group D')
dataset_path = dataset.get_local_copy()

# Get the directory where model.py is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to data_template.yaml
data_template_path = os.path.join(script_dir, 'data_template.yaml')

# Debugging statements
print("Script directory:", script_dir)
print("Contents of script directory:", os.listdir(script_dir))
print("Data template path:", data_template_path)

# Read the data.yaml template
with open(data_template_path, 'r') as f:
    data_yaml_content = f.read()

# Replace the placeholder with the actual dataset path
data_yaml_content = data_yaml_content.replace('{dataset_path}', dataset_path)

# Write the updated data.yaml to a temporary file
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
    tmp.write(data_yaml_content)
    data_yaml_path = tmp.name

# Load a YOLOv5 model (e.g., pretrained on COCO)
model = YOLO('yolov5s.pt')  


# Train the model
model.train(
    data=data_yaml_path,  # Path to the YAML file with dataset information
    epochs=16,            # Number of epochs to train for (adjust based on training needs)
    imgsz=640,            # Image size
    batch=16,             # Batch size
    device='cuda'          # Use 'cpu' or 'cuda' (if you have a GPU)
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
