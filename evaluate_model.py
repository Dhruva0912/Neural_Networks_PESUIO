import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# -------------------------
# PATHS
# -------------------------
test_dir = "data/split/test"
checkpoint_path = "checkpoints/epoch_06.keras"   # change if needed

# -------------------------
# DATA GENERATOR
# -------------------------
img_size = (224, 224)
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1/255.)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

labels = list(test_gen.class_indices.keys())

print(f"\nLoading model from: {checkpoint_path}")
model = load_model(checkpoint_path)

# -------------------------
# EVALUATE
# -------------------------
print("\nEvaluating model...")
test_loss, test_acc = model.evaluate(test_gen)
print(f"\nTest Accuracy: {test_acc:.4f}\n")

# -------------------------
# PREDICTIONS
# -------------------------
y_true = test_gen.classes
y_pred = np.argmax(model.predict(test_gen), axis=1)

# -------------------------
# CLASSIFICATION REPORT
# -------------------------
print("Classification Report:\n")
report = classification_report(
    y_true, y_pred, target_names=labels, digits=2
)
print(report)

# -------------------------
# TOP 15 BY F1-SCORE
# -------------------------
print("\nCalculating Top 15 Classes by F1-score...\n")

# Convert classification report to dict → DataFrame
report_dict = classification_report(
    y_true, y_pred, target_names=labels, output_dict=True
)

df = pd.DataFrame(report_dict).transpose()

# Remove non-class rows
df_classes = df.iloc[:-3]

# Sort by F1-score descending
df_sorted = df_classes.sort_values(by="f1-score", ascending=False)

# Top 15 rows
top_15 = df_sorted.head(15)

print("Top 15 Classes by F1-Score:\n")
print(top_15[["precision", "recall", "f1-score"]])

# -------------------------
# CONFUSION MATRIX
# -------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(15, 12))
sns.heatmap(cm, cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
