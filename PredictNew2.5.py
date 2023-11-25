# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 00:47:37 2023

@author: 15127
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image as img
import numpy as np


#data augmentation such as re-scaling, shear range and zoom range
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
#Create the convolutional base and Maxpooling
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), input_shape=(100, 100,3)))
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), input_shape=(100, 100,3)))
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4))
model.add(layers.Activation('softmax'))


model.summary()
# Define the data directories for train and validation
test_data_dir = "Project 2 Data\\Data\\Test"  # Test


# Define the image target size, batch size, and class mode
image_target_size = (100,100)

# Create train and validation data generators
test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=10,
    shear_range=0.2,
    height_shift_range=0.1,
    width_shift_range=0.1
    )

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_target_size,
    batch_size=32,
    class_mode="categorical"  # Set to "categorical" for classification tasks
)

#3 predition 
predictions = model.predict(test_generator)

predicted_classes = np.argmax(predictions, axis=1)

batch_images, batch_labels = next(test_generator)

# Display the images and labels
for i in range(len(batch_images)):
    plt.imshow(batch_images[i])
    plt.title(f"Label: {batch_labels[i]}")  # Adjust based on your label format
    plt.show()

# dirtt1="Project 2 Data\\Data\\Test\\Medium\\Crack__20180419_06_19_09,915.bmp"
# dirtt2="Project 2 Data\\Data\\Test\\Large\\Crack__20180419_13_29_14,846.bmp"
# samp1 = img.open(dirtt1)
# samp2 = img.open(dirtt2)
# index1 = test_generator.filenames.index(dirtt1)
# index2 = test_generator.filenames.index(dirtt2)

# overlay_text1 = f"Predicted Class: {predicted_classes[index1]}"
# overlay_text2 = f"Predicted Class: {predicted_classes[index2]}"

# plt.imshow(samp1)
#     plt.text(10, 10, overlay_text1, color='red', fontsize=12, b
#              box=dict(facecolor='white', alpha=0.7))
#     plt.axis('off') 
    
#  plt.imshow(samp2)
#      plt.text(10, 10, overlay_text2, color='red', fontsize=12, b
#               box=dict(facecolor='white', alpha=0.7))
#      plt.axis('off') 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 