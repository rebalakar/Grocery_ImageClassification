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

- Run the following command to sample the dataset (should be done only once as the results are saved to a csv file):
```shell
python 5ccsagap_group_project_1/sample.py    
```
