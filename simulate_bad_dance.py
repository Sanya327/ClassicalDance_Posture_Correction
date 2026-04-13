import json
import numpy as np
import os
import random

def load_keypoints(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_keypoints(wrapped_data, file_path):
    with open(file_path, 'w') as f:
        json.dump(wrapped_data, f)

def add_noise_to_keypoints(data, noise_strength=0.05, missing_chance=0.1):
    """Creates a modified copy of pose data with noise and possible keypoint removal."""
    new_data = []
    for frame in data:
        noisy_frame = {}
        for k, v in frame.items():
            if random.random() < missing_chance:
                noisy_frame[k] = [0.0, 0.0, 0.0]  # simulate lost keypoint
            else:
                noise = np.random.normal(0, noise_strength, 3)
                noisy = np.clip(np.array(v) + noise, 0.0, 1.0)
                noisy_frame[k] = noisy.tolist()
        new_data.append(noisy_frame)
    return new_data

def calculate_fake_score(noise_strength, missing_chance):
    """
    Simulate a 'performance score' based on how much noise and keypoint loss is present.
    You can adjust the formula as you like.
    """
    # Higher noise and missing = lower score
    score = max(0.0, 1.0 - (noise_strength * 5 + missing_chance * 2))
    return round(score, 2)

def generate_bad_versions(original_file, output_dir, count=5):
    base_name = os.path.basename(original_file).replace("-keypoints.json", "")
    data = load_keypoints(original_file)

    os.makedirs(output_dir, exist_ok=True)

    for i in range(count):
        noise_strength = 0.05 + 0.02 * i
        missing_chance = 0.1 + 0.02 * i

        modified = add_noise_to_keypoints(data, noise_strength, missing_chance)
        score = calculate_fake_score(noise_strength, missing_chance)

        wrapped_output = {
            "score": score,
            "frames": modified
        }

        out_path = os.path.join(output_dir, f"{base_name}_bad_{i}-keypoints.json")
        save_keypoints(wrapped_output, out_path)
        print(f"✅ Generated: {out_path} (Score: {score})")

# CONFIG
input_json = "keypoint-extractor/output/pv na-keypoints.json"  # <- your original clean file
output_folder = "keypoint-extractor/output/bad_versions"

generate_bad_versions(input_json, output_folder, count=5)
