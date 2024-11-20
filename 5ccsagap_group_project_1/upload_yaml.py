from clearml import Task

# Uploads data_template.yaml to ClearML as an artifact
task = Task.init(project_name='5CCSAGAP Group D', task_name='Upload Data Template')

# Path to your local 'data_template.yaml' file
local_yaml_path = '5ccsagap_group_project_1/data_template.yaml'

# Correctly upload the file as an artifact
task.upload_artifact(name='data_template.yaml', artifact_object=local_yaml_path)

task.close()
