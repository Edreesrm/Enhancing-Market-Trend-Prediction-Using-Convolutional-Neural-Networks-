# -*- coding: utf-8 -*-
"""
Created on Fri May 31 07:30:42 2024

@author: edree
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix

# Define the path to your dataset
dataset_path = 'C:/Users/PIXEL/Desktop/test/C_i5_2/chart_images5_2/'  # Change this to the path where your data is stored
# Initialize the ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)

# Load images and labels
def load_images(data_path):
    generator = datagen.flow_from_directory(
        data_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',  
        shuffle=True
    )
    num_samples = generator.samples
    images = np.zeros((num_samples, 150, 150, 3))
    labels = np.zeros((num_samples,))
    i = 0
    for x_batch, y_batch in generator:
        images[i * 32:(i + 1) * 32] = x_batch
        labels[i * 32:(i + 1) * 32] = y_batch
        i += 1
        if i >= num_samples / 32:
            break
    return images, labels

X, y = load_images(dataset_path)

# Define the CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model
# K-fold Cross-Validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
acc_per_fold = []
loss_per_fold = []
histories = []
confusion_matrices = []
classification_reports = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = create_model()  

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        verbose=2
    )
    histories.append(history)

    # Evaluate the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    acc_per_fold.append(scores[1])
    loss_per_fold.append(scores[0])

    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)

    cm = confusion_matrix(y_test, predicted_classes)
    cr = classification_report(y_test, predicted_classes)
    confusion_matrices.append(cm)
    classification_reports.append(cr)

# Output the scores per fold
print('Score per fold:')
for i in range(len(acc_per_fold)):
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')

for i in range(len(confusion_matrices)):
    print(f'Confusion Matrix for Fold {i+1}:')
    print(confusion_matrices[i])
    print(f'Classification Report for Fold {i+1}:')
    print(classification_reports[i])

# Plot the training and validation accuracy and loss
plt.figure(figsize=(15, 10))
for i, history in enumerate(histories):
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label=f'Train Fold {i+1}')
    plt.plot(history.history['val_accuracy'], label=f'Val Fold {i+1}')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label=f'Train Fold {i+1}')
    plt.plot(history.history['val_loss'], label=f'Val Fold {i+1}')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
plt.show()
