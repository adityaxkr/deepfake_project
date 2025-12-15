import os

base_path = "model_training/data"
splits = ['train', 'val', 'test']
classes = ['real', 'fake']

for split in splits:
    for cls in classes:
        dir_path = os.path.join(base_path, split, cls)
        os.makedirs(dir_path, exist_ok=True)

print("✅ Directory structure created.")
