from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt

# Load the trained model
model = load_model(r"D:/mlops/leaf_deficiency_classifier.h5")

# Option 2: Get class names from the directory structure
data_dir = r"D:/mlops/FinalDataset"
class_names = os.listdir(data_dir)

# Prediction function
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(512, 512))  # Resize as per training
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display the image with prediction text
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)", fontsize=14)
    plt.axis('off')
    plt.show()

# Loop until user enters 'q'
while True:
    img_path = input("Enter the image path (or 'q' to quit): ")
    if img_path.lower() == 'q':
        print("Exiting the prediction loop.")
        break
    if os.path.exists(img_path):
        try:
            predict_image(img_path)
        except Exception as e:
            print(f"Error processing image: {e}")
    else:
        print("Invalid path. Please try again.")
