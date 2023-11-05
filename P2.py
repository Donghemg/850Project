# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 18:25:54 2023

@author: 15127
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image as img

desired_shape = (100, 100, 3)

#data augmentation such as re-scaling, shear range and zoom range
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=10,
    shear_range=0.2,
    height_shift_range=0.1,
    width_shift_range=0.1
)

# Specify the input and output folders
input_folder = "Project 2 Data\\Data\\Train"
output_folder = "filtered_images"  # Change this to your desired output folder

os.makedirs(output_folder, exist_ok=True)

#Create the train and validation generator
generator = datagen.flow_from_directory(
    directory=input_folder,
    target_size=(desired_shape[0], desired_shape[1]),
    batch_size=32,
    class_mode=None,  # We don't need class labels
    shuffle=False  # Disable shuffling for predictable order
)

# #Iterate through the generator and save filtered images
while True:
    try:
        image_array = next(generator)
        if image_array.shape[1:] == desired_shape:
            image_path = os.path.join(input_folder, generator.filenames[generator.batch_index - 1])
            image_array = (image_array[0] * 255).astype('uint8')
            filtered_image = img.fromarray(image_array)
            filtered_image.save(os.path.join(output_folder, os.path.basename(image_path)))
        else:
            print(f"Dropped image {generator.filenames[generator.batch_index - 1]} with undesired dimension.")
    except StopIteration:
        break