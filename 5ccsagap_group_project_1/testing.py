# predict_with_clearml.py

from ultralytics import YOLO
from clearml import Task, Logger
import os

def main():
    # Initialize ClearML Task
    task = Task.init(
        project_name='YOLOv5_Custom_Project',
        task_name='YOLOv5_Inference_With_ClearML',
        task_type=Task.TaskTypes.testing
    )

    # Define inference parameters
    hyperparams = {
        'weights': '/home/ethan/PycharmProjects/5CCSAGAP-group-project-1/runs/detect/train15/weights/best.pt',  # Update with your path
        'source': './temp/',  # Path to images or video for inference
        'img_size': 640,
        'conf_threshold': 0.25,
        'iou_threshold': 0.45,
        'save_txt': True,
        'save_conf': True,
        'save_img': True,
        'device': 'cuda:0',  # or 'cpu'
    }

    # Log hyperparameters
    task.connect(hyperparams)

    # Load the trained model
    model = YOLO(hyperparams['weights'])

    # Perform inference
    results = model.predict(
        source=hyperparams['source'],
        imgsz=hyperparams['img_size'],
        conf=hyperparams['conf_threshold'],
        iou=hyperparams['iou_threshold'],
        save_txt=hyperparams['save_txt'],
        save=True,  # Saves annotated images
        device=hyperparams['device']
    )

    '''# Define paths to artifacts
    detected_images_dir = model.results.save_dir  # Directory where images are saved
    results_txt_dir = os.path.join(detected_images_dir, 'labels')  # Directory for .txt files

    # Upload detected images as an artifact
    if os.path.exists(detected_images_dir):
        task.upload_artifact(name='detected_images', artifact_object=detected_images_dir)
        task.get_logger().report_text(f"Uploaded detected images from {detected_images_dir}", level=Logger.REPORT_LEVEL_INFO)
    else:
        task.get_logger().report_text(f"Detected images directory not found: {detected_images_dir}", level=Logger.REPORT_LEVEL_WARNING)

    # Upload results.txt files as an artifact
    if os.path.exists(results_txt_dir):
        task.upload_artifact(name='detection_results', artifact_object=results_txt_dir)
        task.get_logger().report_text(f"Uploaded detection results from {results_txt_dir}", level=Logger.REPORT_LEVEL_INFO)
    else:
        task.get_logger().report_text(f"Detection results directory not found: {results_txt_dir}", level=Logger.REPORT_LEVEL_WARNING)'''

    # Close the ClearML task
    task.close()

if __name__ == "__main__":
    main()
