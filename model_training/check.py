import os

real_dir = "model_training/data/train/real"
fake_dir = "model_training/data/train/fake"

print("🔍 Real images:")
print(os.listdir(real_dir)[:5])  # Print first 5 filenames

print("🔍 Fake images:")
print(os.listdir(fake_dir)[:5])
