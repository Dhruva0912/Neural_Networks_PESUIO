import os
import shutil
import random

# Paths
SOURCE_DIR = "data/archive/training_set/training_set"
BASE_DIR = "data/split"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

# Ratios
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Create directories
for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(folder, exist_ok=True)

# Loop through each class
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    train_end = int(len(images) * TRAIN_SPLIT)
    val_end = train_end + int(len(images) * VAL_SPLIT)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    # Create class folders
    os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, class_name), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, class_name), exist_ok=True)

    # Copy files
    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(TRAIN_DIR, class_name))

    for img in val_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(VAL_DIR, class_name))

    for img in test_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(TEST_DIR, class_name))

print("Dataset split complete!")
