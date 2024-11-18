# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:00:06 2024

@author: PIXEL
"""

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('chart_classification_model.h5')

# Define the image dimensions 
img_height = 150
img_width = 150

 # Replace class names
class_labels = ['uptrend', 'downtrend'] 

# Create the main window
root = tk.Tk()
root.title("Image Classification")

# Set window size
root.geometry("600x600")

image_label = tk.Label(root)
image_label.pack(pady=20)

result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack()

def preprocess_image(image_path):
    """Preprocess the image to the required size and shape."""
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    # Rescale as done during training
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_image():
    """Classify the selected image and display the result."""
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        try:
            # Preprocess the image
            processed_image = preprocess_image(file_path)
            
            prediction = model.predict(processed_image)
            predicted_class = np.where(prediction > 0.7, 1, 0)
            
            predicted_label = class_labels[predicted_class[0][0]]
            
            img = Image.open(file_path)
            img = img.resize((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            image_label.configure(image=img_tk)
            image_label.image = img_tk 
            
            # Display the prediction
            result_label.config(text=f"Predicted Class: {predicted_label}")
        except Exception as e:
            result_label.config(text=f"Error: {str(e)}")

classify_button = tk.Button(root, text="Select Image", command=classify_image)
classify_button.pack(pady=10)

root.mainloop()
