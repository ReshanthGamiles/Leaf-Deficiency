import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image

# ✅ GPU Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU(s) detected: {gpus}")
    except RuntimeError as e:
        print(f"❌ Error setting GPU memory growth: {e}")
else:
    print("❌ No GPU detected. Running on CPU.")

# ✅ 1. Dataset path
data_dir = r"D:/mlops/FinalDataset"

# ✅ 2. Image size and training config
img_height = 512
img_width = 512
batch_size = 32
epochs = 20

# ✅ 3. Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2
)

# ✅ 4. Load training data
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# ✅ 5. Load validation data
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# ✅ 6. Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

# ✅ 7. Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ 8. Save best model (.keras)
checkpoint = ModelCheckpoint('best_leaf_model.keras', monitor='val_accuracy', save_best_only=True)

# ✅ 9. Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[checkpoint]
)

# ✅ 10. Save final model
model.save("leaf_deficiency_classifier.h5")
print("✅ Model training complete and saved!")

# ✅ 11. Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# ✅ 12. Get class names for prediction
class_names = os.listdir(data_dir)

# ✅ 13. Prediction function
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    print(f"🧠 Predicted: {predicted_class} ({confidence:.2f}%)")

# ✅ Example usage:
# predict_image(r"C:/Users/USER/OneDrive/Desktop/mangospot.png")
