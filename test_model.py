
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "dataset" / "test"
MODEL_PATH = BASE_DIR / "models" / "leaflens.keras"
CLASS_PATH = BASE_DIR / "models" / "class_names.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

# Convert filename prefixes into expected disease labels
def get_expected_label(filename):
    name = filename.lower()

    labels = {
        "applecedarrust": "Apple___Cedar_apple_rust",
        "applescab": "Apple___Apple_scab",
        "corncommonrust": "Corn_(maize)___Common_rust_",
        "potatoearlyblight": "Potato___Early_blight",
        "potatohealthy": "Potato___healthy",
        "tomatoearlyblight": "Tomato___Early_blight",
        "tomatohealthy": "Tomato___healthy",
        "tomatoyellowcurlvirus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    }

    for prefix, label in labels.items():
        if name.startswith(prefix):
            return label

    return None

extensions = {".jpg", ".jpeg", ".png"}
test_images = sorted(
    p for p in TEST_DIR.iterdir()
    if p.suffix.lower() in extensions
)

correct = 0
evaluated = 0

for path in test_images:
    expected = get_expected_label(path.name)

    if expected is None or expected not in class_names:
        print(f"Skipped: {path.name} (label not mapped)")
        continue

    image = Image.open(path).convert("RGB").resize((160, 160))
    image_array = np.array(image, dtype=np.float32)[None, ...]

    probabilities = model.predict(image_array, verbose=0)[0]
    predicted = class_names[int(np.argmax(probabilities))]

    evaluated += 1
    is_correct = predicted == expected
    correct += int(is_correct)

    status = "CORRECT" if is_correct else "INCORRECT"
    print(f"{path.name}: {status}")
    if not is_correct:
        print(f"  Expected: {expected}")
        print(f"  Predicted: {predicted}")

print("\n--- TEST RESULTS ---")
print(f"Images evaluated: {evaluated}")
print(f"Correct predictions: {correct}")

if evaluated:
    print(f"Test accuracy: {correct / evaluated * 100:.2f}%")
else:
    print("No images could be evaluated. Check the filename labels.")
