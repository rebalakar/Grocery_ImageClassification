# scripts/model.py

from clearml import Task, Dataset
import subprocess
from datetime import datetime
import logging
import yaml
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Initialize ClearML Task
    task = Task.init(
        project_name='5CCSAGAP Group D',
        task_name='Training',
        task_type=Task.TaskTypes.training
    )

    task.connect_configuration({"requirements_file": "requirements.txt"})
    task.execute_remotely(queue_name='default', exit_process=True)

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

    # Load and verify data.yaml using the temporary file path
    with open(data_yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)
    logger.info(f"Loaded data.yaml: {data_yaml}")

    # Define hyperparameters
    params = {
        'img': 640,
        'batch': 16,
        'epochs': 16,
        'data': data_yaml_path,
        'weights': 'yolov5s.pt',
        'project': '5CCSAGAP Group D',
        'name': f'Training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
    }

    # Log hyperparameters
    task.connect(params)
    logger.info(f"Hyperparameters: {params}")

    # Define the training command
    cmd = [
        'python3', '../yolov5/train.py',
        '--img', str(params['img']),
        '--batch', str(params['batch']),
        '--epochs', str(params['epochs']),
        '--data', params['data'],
        '--weights', params['weights'],
        '--project', params['project'],
        '--name', params['name'],
    ]

    # Run the training process
    subprocess.run(cmd, check=True)
    logger.info("Training completed successfully.")

    # Close the ClearML task
    task.close()
    logger.info("Closed ClearML task.")


if __name__ == '__main__':
    main()
