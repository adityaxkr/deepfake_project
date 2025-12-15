import os
import shutil

# Source directory where CelebA real images are stored
source_dir = r"C:\Users\adity\OneDrive\Desktop\deep_fake_assist\celeba\img_align_celeba\img_align_celeba"

# Destination base path
base_dest = os.path.join("model_training", "data")
splits = {
    "train": 1600,
    "val": 200,
    "test": 200
}

# Step 1: Clean real image folders
for split in splits:
    dest = os.path.join(base_dest, split, "real")
    if os.path.exists(dest):
        for file in os.listdir(dest):
            file_path = os.path.join(dest, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(dest)

# Step 2: Copy real images from CelebA to destination
print("📦 Copying real images...")
all_images = sorted(os.listdir(source_dir))
start_idx = 0

for split, count in splits.items():
    dest = os.path.join(base_dest, split, "real")
    for i in range(count):
        src_path = os.path.join(source_dir, all_images[start_idx + i])
        dst_path = os.path.join(dest, all_images[start_idx + i])
        shutil.copyfile(src_path, dst_path)
    start_idx += count

print("✅ Real image organization complete.")
