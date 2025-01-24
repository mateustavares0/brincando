import kagglehub

# Download latest version
path = kagglehub.dataset_download("xregis/train_test_br_coins")

print("Path to dataset files:", path)
