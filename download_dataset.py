import kagglehub
import shutil
import os

# Download dataset
path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")

print("Downloaded to:", path)

# Create local data folder
os.makedirs("data", exist_ok=True)

# Copy files into your project data folder
destination = "data/lgg-mri"

if os.path.exists(destination):
    print("Data already exists, skipping copy.")
else:
    shutil.copytree(path, destination)
    print("Data copied to:", destination)