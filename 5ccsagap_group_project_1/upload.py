from clearml import Dataset

# Create a new dataset
dataset = Dataset.create(
    dataset_name='Sampled Images', 
    dataset_project='5CCSAGAP Group D', 
    parent_datasets=None
)

# Add files to the dataset
dataset.add_files('./dataset/')

# Upload the dataset to ClearML server
dataset.upload()

# Finalize the dataset (make it read-only)
dataset.finalize()
