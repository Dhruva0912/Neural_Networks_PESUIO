import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import json
import os

# Paths
base_dir = "data/split"
train_dir = os.path.join(base_dir, "train")
val_dir   = os.path.join(base_dir, "val")

# Image settings
img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1/255.)
val_datagen   = ImageDataGenerator(rescale=1/255.)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

num_classes = len(train_gen.class_indices)

# Model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Checkpoint saving
checkpoint = ModelCheckpoint(
    filepath="checkpoints/epoch_{epoch:02d}.keras",
    save_weights_only=False,
    save_freq="epoch"
)

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[checkpoint]
)

# Save history
with open("training_history.json", "w") as f:
    json.dump(history.history, f)

print("Training completed and history saved.")
