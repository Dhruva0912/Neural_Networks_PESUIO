import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# Load model (use your best checkpoint)
model_path = "checkpoints/epoch_06.keras"
model = tf.keras.models.load_model(model_path)

print(f"Loaded model from: {model_path}")

# Load class labels ***IMPORTANT***
train_dir = "data/split/train"
class_names = sorted(os.listdir(train_dir))

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    print("\n<------------------------------------------------------------------------->")
    print(f"Predicted Bird: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")
    print("<------------------------------------------------------------------------->")

# Loop for user input
while True:
    img_path = input("\nEnter image path (or 'q' to quit): ").strip()

    if img_path.lower() == 'q':
        print("Exiting prediction tool.")
        break

    if not os.path.exists(img_path):
        print("File not found. Try again.")
        continue

    predict_image(img_path)
