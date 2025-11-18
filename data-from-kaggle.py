import kagglehub

# Download latest version
path = kagglehub.dataset_download("imtkaggleteam/breast-cancer")

print("Path to dataset files:", path)