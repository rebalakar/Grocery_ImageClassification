# scripts/model.py

from clearml import Task, Dataset
import subprocess
from datetime import datetime
import logging
import yaml
import tempfile
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_requirements(requirements_path):
    """
    Installs the required Python packages from a given requirements.txt file.
    """
    logger.info(f"Installing dependencies from {requirements_path}...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
        logger.info("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        sys.exit(1)

def get_git_commit_hash():
    """
    Retrieves the current Git commit hash.
    """
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        return commit_hash
    except subprocess.CalledProcessError:
        return "Unknown"

def main():
    # Initialize ClearML Task
    task = Task.init(
        project_name='5CCSAGAP Group D',
        task_name='Training_New',  # Use a unique task name to avoid cached environments
        task_type=Task.TaskTypes.training
    )

    task.connect_configuration({"requirements_file": "requirements.txt"})
    task.execute_remotely(queue_name='default', exit_process=True)

    # Get the dataset
    dataset = Dataset.get(dataset_name='Sampled Images', dataset_project='5CCSAGAP Group D')
    dataset_path = dataset.get_local_copy()

    # Fetch the 'data_template.yaml' artifact
    artifact_task = Task.get_task(project_name='5CCSAGAP Group D', task_name='Upload Data Template')
    artifact = artifact_task.artifacts.get('data_template.yaml')
    if not artifact:
        logger.error("Artifact 'data_template.yaml' not found.")
        task.close()
        sys.exit(1)
    artifact_path = artifact.get_local_copy()

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
        'batch': 32,
        'epochs': 100,
        'data': data_yaml_path,
        'weights': 'yolov5s.pt',
        'project': '5CCSAGAP Group D',
        'name': f'Training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
    }

    # Log hyperparameters
    task.connect(params)
    logger.info(f"Hyperparameters: {params}")

    # Log the current Git commit hash
    commit_hash = get_git_commit_hash()
    logger.info(f"Current Git Commit Hash: {commit_hash}")

    # Define YOLOv5 directory and train.py path using absolute paths
    script_dir = os.path.dirname(os.path.realpath(__file__))  # Directory containing model.py
    yolov5_dir = os.path.abspath(os.path.join(script_dir, '..', 'yolov5'))
    yolov5_train_script = os.path.join(yolov5_dir, 'train.py')

    # Clone YOLOv5 repository if it doesn't exist
    if not os.path.isdir(yolov5_dir):
        logger.info("Cloning YOLOv5 repository...")
        try:
            subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git', yolov5_dir], check=True)
            logger.info("YOLOv5 cloned successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone YOLOv5 repository: {e.stderr}")
            task.mark_failed()
            sys.exit(1)

    # Install YOLOv5 dependencies
    requirements_path = os.path.join(yolov5_dir, 'requirements.txt')
    if os.path.isfile(requirements_path):
        install_requirements(requirements_path)
        # Log installed packages for verification
        logger.info("Installed Python packages:")
        installed_packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode('utf-8')
        logger.info(installed_packages)
    else:
        logger.error(f"Requirements file not found at: {requirements_path}")
        task.mark_failed()
        sys.exit(1)

    # Verify that train.py exists
    if not os.path.isfile(yolov5_train_script):
        logger.error(f"YOLOv5 training script not found at: {yolov5_train_script}")
        task.close()
        sys.exit(1)

    # Define the training command using sys.executable for consistency
    cmd = [
        sys.executable, yolov5_train_script,
        '--img', str(params['img']),
        '--batch', str(params['batch']),
        '--epochs', str(params['epochs']),
        '--data', params['data'],
        '--weights', params['weights'],
        '--project', params['project'],
        '--name', params['name'],
    ]

    logger.info(f"Running training command: {' '.join(cmd)}")

    # Run the training process
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.info(f"Training stdout: {result.stdout}")
        logger.info("Training completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with error: {e.stderr}")
        task.mark_failed()
        sys.exit(1)

    # Close the ClearML task
    task.close()
    logger.info("Closed ClearML task.")

    # Optionally, clean up the temporary data.yaml file
    try:
        os.remove(data_yaml_path)
        logger.info(f"Removed temporary data.yaml at: {data_yaml_path}")
    except Exception as e:
        logger.warning(f"Could not remove temporary data.yaml: {e}")

if __name__ == '__main__':
    main()
