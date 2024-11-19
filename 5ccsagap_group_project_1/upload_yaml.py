from clearml import Task
#uploads data_template_yaml to clearml as an artifact
task = Task.init(project_name='5CCSAGAP Group D', task_name='Upload Data Template')

task.upload_artifact(name='data_template.yaml', artifact_object='data_template.yaml')

task.close()
