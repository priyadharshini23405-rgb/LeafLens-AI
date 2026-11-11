
from pathlib import Path
import json
import shutil

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "dataset" / "test"
OUTPUT_DIR = BASE_DIR / "dataset" / "test_labeled"
CLASS_FILE = BASE_DIR / "models" / "class_names.json"

with open(CLASS_FILE, "r") as file:
    class_names = json.load(file)

# Filename prefixes mapped to their actual dataset class names
prefix_mapping = {
    "AppleCedarRust": "Apple___Cedar_apple_rust",
    "AppleScab": "Apple___Apple_scab",
    "CornCommonRust": "Corn_(maize)___Common_rust",
    "PotatoEarlyBlight": "Potato___Early_blight",
    "PotatoHealthy": "Potato___healthy",
    "TomatoEarlyBlight": "Tomato___Early_blight",
    "TomatoHealthy": "Tomato___healthy",
    "TomatoYellowCurlVirus":
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}

# Match labels to the exact names used by your model
for prefix, label_part in prefix_mapping.items():
    matches = [
        name for name in class_names
        if label_part.lower() in name.lower()
    ]

    if len(matches) != 1:
        print(f"CHECK LABEL: {prefix} -> {matches}")
        continue

    label = matches[0]
    output_folder = OUTPUT_DIR / label
    output_folder.mkdir(parents=True, exist_ok=True)

    for image_path in TEST_DIR.iterdir():
        if image_path.is_file() and image_path.name.startswith(prefix):
            destination = output_folder / image_path.name
            shutil.copy2(image_path, destination)
            print(f"Copied: {image_path.name} -> {label}")

print("\nFinished preparing test images.")
print("Please verify the labels before evaluating the model.")
