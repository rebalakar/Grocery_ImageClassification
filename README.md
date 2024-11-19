# 5CCSAGAP-group-project-1

## Environment Setup

### Prerequisites

- [Python 3.12](https://www.python.org/downloads/release/python-3124/)
- [Poetry](https://python-poetry.org/docs/)

### Steps

- Initialize a virtual environment using Poetry:
```shell
poetry shell
```

- Install the dependencies:
```shell
poetry install
```

- Add new dependencies:
```shell
poetry add <package-name>
```


## Data Setup

Create a folder named `data` in the root directory of the project and extract the contents of [this archive](https://drive.google.com/file/d/1iq93lCdhaPUN0fWbLieMtzfB1850pKwd/edit) into the `data` folder.

## Scripts

### Generating stats for EDA

- Run the following command to generate statistics in the terminal save plots in `plots` folder for exploratory data analysis (EDA):
```shell
python 5ccsagap_group_project_1/eda.py
```

### Sampling

- Run the following command to sample the dataset, saves images in `dataset` folder and annotations in `sampled_annotation` folder (should be done only once as the results are saved to a csv file):
```shell
python 5ccsagap_group_project_1/sample.py    
```

### Converting Annotations to YOLO Format

- Run the following command to convert the CSV annotation files into YOLO format. The resulting `.txt` files are saved in `dataset/labels` with subfolders for `train`, `val`, and `test`:
```shell
python 5ccsagap_group_project_1/to_txt.py
```

## ClearML Configuration

To use ClearML for task management and dataset artifact uploads, you need to configure your ClearML credentials. Follow these steps:

1. Locate or create the ClearML configuration file at `~/.clearml.conf`.
2. Add your credentials in the following format:
```plaintext
api {
    # ClearML server URL
    api_server: https://app.clearml.example.com
    web_server: https://web.clearml.example.com
    files_server: https://files.clearml.example.com

    # Your ClearML credentials
    credentials {
        access_key: "<your-access-key>"
        secret_key: "<your-secret-key>"
    }
}
```
---

## Scripts for ClearML Integration

### Upload Dataset to ClearML

Use the following command to upload the `dataset` folder to ClearML as an artifact. This is essential for sharing datasets with remote tasks or agents:

```shell
python 5ccsagap_group_project_1/upload.py
```
---

## Data Template Configuration (`data_template.yaml`)

The `data_template.yaml` file is a template used to define dataset paths and other configurations dynamically. Ensure the file is configured correctly before use.

### Steps to Upload `data_template.yaml` as an Artifact to ClearML

1. Verify that the `data_template.yaml` file is located in the project directory:

5ccsagap_group_project_1/data_template.yaml


2. Use the following command to upload it:
```shell
python 5ccsagap_group_project_1/upload_yaml.py
```
