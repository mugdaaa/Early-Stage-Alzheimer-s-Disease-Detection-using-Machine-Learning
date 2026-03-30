import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset folder
train_dir = r"C:\Users\nitis\Desktop\ALZHEIMER'S PROJECT\MRI_dataset"

# Image preprocessing
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(128,128),
    color_mode="grayscale",
    class_mode="binary"
)

# CNN model
model = Sequential([

    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")

])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(train_data, epochs=10)

# Save model
model.save(r"C:\Users\nitis\Desktop\ALZHEIMER'S PROJECT\MRI_dataset\mri_cnn_model.h5")

print("CNN model saved successfully")