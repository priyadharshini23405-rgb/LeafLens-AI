
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# Project paths
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
VALID_DIR = BASE_DIR / "dataset" / "valid"
MODEL_DIR = BASE_DIR / "models"

# Training settings
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 5
SEED = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load training dataset
print("Loading training dataset...", flush=True)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
)

# Load validation dataset
valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = train_ds.class_names

# Save class names for the Streamlit app
with open(MODEL_DIR / "class_names.json", "w") as file:
    json.dump(class_names, file)

print(f"Found {len(class_names)} disease classes.", flush=True)

# Improve data loading performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)

# Data augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")

# Load pretrained MobileNetV2
print("Loading MobileNetV2 model...", flush=True)

base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet",
)

# Freeze pretrained layers
base_model.trainable = False

# Build LeafLens AI model
model = models.Sequential([
    layers.Input(shape=(160, 160, 3)),
    data_augmentation,
    layers.Rescaling(1.0 / 127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax"),
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Save the best model based on validation accuracy
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(MODEL_DIR / "leaflens.keras"),
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)

# Stop when validation loss stops improving
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
    verbose=1,
)

# Train model
print("Starting training...", flush=True)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop],
)

# Save final best model
model.save(MODEL_DIR / "leaflens.keras")

print("\nTraining completed successfully!", flush=True)
print(f"Model saved at: {MODEL_DIR / 'leaflens.keras'}", flush=True)
print(f"Classes: {len(class_names)}", flush=True)
