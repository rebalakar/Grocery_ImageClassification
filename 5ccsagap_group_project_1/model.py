# train_with_clearml.py

from ultralytics import YOLO
from clearml import Task, Dataset
import os
import yaml


def main():
    # Initialize ClearML Task
    task = Task.init(
        project_name='YOLOv5_Custom_Project',
        task_name='YOLOv5_Training_With_ClearML',
        task_type=Task.TaskTypes.training
    )

    task.connect_configuration({"requirements_file": "requirements.txt"})
    # Enqueue the task for remote execution
    task.execute_remotely(queue_name='default', exit_process=True)

    # Get the dataset from ClearML
    dataset = Dataset.get(dataset_name='Sampled Images', dataset_project='5CCSAGAP Group D')
    dataset_path = dataset.get_local_copy()



    # Define paths based on the downloaded dataset
    train_images = os.path.join(dataset_path, 'images', 'train')
    val_images = os.path.join(dataset_path, 'images', 'val')

    # Create a data configuration dictionary
    data_config = {
        'train': train_images,
        'val': val_images,
        'nc': 1,  # Number of classes
        'names': ['object']  # Replace with your actual class name
    }

    # Save the data configuration to 'data.yaml'
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f)


    # Define training parameters
    model = YOLO('yolov5s.pt')  # Specify the model architecture

    # Optional: Connect hyperparameters to ClearML
    hyperparams = {
        'epochs': 1,
        'batch_size': 2,
        'img_size': 640,
        'data': 'data_yaml_path',  # Path to your data configuration
        'optimizer': 'AdamW',
        'learning_rate': 0.01,
        # Add other hyperparameters as needed
    }
    task.connect(hyperparams)

    # Start training
    results = model.train(
        data=hyperparams['data'],
        epochs=hyperparams['epochs'],
        batch=hyperparams['batch_size'],
        imgsz=hyperparams['img_size'],
        optimizer=hyperparams['optimizer'],
        lr0=hyperparams['learning_rate'],
        # Add other training arguments as needed
    )

    # Upload the best model as an artifact
    '''best_model_path = os.path.join(model.results.save_dir, 'best.pt')
    task.upload_artifact(name='best_model', artifact_object=best_model_path)'''

    # Close the ClearML task
    task.close()

if __name__ == "__main__":
    main()
