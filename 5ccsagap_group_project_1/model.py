from clearml import Task
import subprocess


def main():
    # Initialize ClearML Task
    task = Task.init(
        project_name='5CCSAGAP Group D',
        task_name='Training',
        task_type=Task.TaskTypes.training
    )

    # Set up parameters (you can customize these)
    params = {
        'img': 640,
        'batch': 16,
        'epochs': 50,
        'data': '../data/data.yaml',
        'weights': 'yolov5s.pt',
        'cache': True,
        'project': '5CCSAGAP Group D',
        'name': 'Training'
    }

    # Log hyperparameters
    task.connect(params)

    # Execute YOLOv5 training via subprocess
    cmd = [
        'python', '../yolov5/train.py',
        '--img', str(params['img']),
        '--batch', str(params['batch']),
        '--epochs', str(params['epochs']),
        '--data', params['data'],
        '--weights', params['weights'],
        '--cache',
        '--project', params['project'],
        '--name', params['name']
    ]

    # Run the command
    subprocess.run(cmd, check=True)

    # Optionally, upload artifacts or models
    task.upload_artifact(name='model', artifact_object='path/to/best.pt')

    # Mark the task as completed
    task.close()


if __name__ == '__main__':
    main()
