import os
import numpy as np

def aggregate_features(features_dir="features_local/"):
    feature_list = []
    for file in os.listdir(features_dir):
        if file.endswith(".npy"):
            data = np.load(os.path.join(features_dir, file))
            feature_list.append(data)

    if not feature_list:
        print("❌ No features to aggregate.")
        return

    # Simulated aggregation (e.g., averaging)
    global_feature = np.mean(feature_list, axis=0)
    np.save("federated_model/global_feature_avg.npy", global_feature)
    print("✅ Federated model updated (average feature vector).")

# Example:
# aggregate_features()
