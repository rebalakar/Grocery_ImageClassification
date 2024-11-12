from ultralytics import YOLO

# Load a YOLOv5 model (e.g., pretrained on COCO)
model = YOLO('yolov5s.pt')  

# Path to the dataset YAML file
data_yaml_path = './data/data.yaml'

# Train the model
model.train(
    data=data_yaml_path,  # Path to the YAML file with dataset information
    epochs=16,            # Number of epochs to train for (adjust based on training needs)
    imgsz=640,            # Image size
    batch=16,             # Batch size
    device='cpu'          # Use 'cpu' or 'cuda' (if you have a GPU)
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
sample_image_path = "./images/sample_image.jpg"  # Replace with a valid test image path
results = model.predict(source=sample_image_path, save=True)
print(results)  # Print predictions
