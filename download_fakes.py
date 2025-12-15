import os
import requests
import time
import random

# Paths
base_path = "model_training/data"
splits = {
    "train": 1,
    "val": 0,
    "test": 0
}

# Clean up and create folders
for split in splits:
    path = os.path.join(base_path, split, "fake")
    os.makedirs(path, exist_ok=True)
    # Optional: You can uncomment the cleaning loop if needed
    # for file in os.listdir(path):
    #     file_path = os.path.join(path, file)
    #     if os.path.isfile(file_path):
    #         os.remove(file_path)

# Function to download with delay and backoff
def download_image(url, dest_path, retries=5):
    headers = {
        'User-Agent': random.choice([
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
        ])
    }

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                with open(dest_path, "wb") as f:
                    f.write(response.content)
                print(f"✅ Saved: {dest_path}")
                return True
            else:
                print(f"⚠️ Status code {response.status_code} for {dest_path}")
        except Exception as e:
            print(f"⚠️ Error: {e} on {dest_path}")
        
        # Wait before retrying
        time.sleep(random.uniform(1, 3)) 
    
    print(f"❌ Failed after retries: {dest_path}")
    return False

# --- Main Download Loop ---
print("⬇️ Downloading fake images...")
base_url = "https://thispersondoesnotexist.com/"

for split, count in splits.items():
    folder = os.path.join(base_path, split, "fake")
    for i in range(count):
        # ✨ KEY CHANGE: Add a unique parameter to the URL to prevent caching
        unique_url = f"{base_url}?{time.time()}{random.random()}"
        
        file_name = f"{split}_799.jpg"
        path = os.path.join(folder, file_name)
        
        if download_image(unique_url, path):
            # Add a small delay between successful downloads to be respectful to the server
            time.sleep(random.uniform(0.5, 1.5))

print("✅ All downloads attempted.")