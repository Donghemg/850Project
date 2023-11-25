# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 18:25:54 2023

@author: 15127
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image as img


#data augmentation such as re-scaling, shear range and zoom range


# Define the data directories for train and validation
train_data_dir = "Project 2 Data\\Data\\Train"  # Replace with the path to your training data
validation_data_dir = "Project 2 Data\\Data\\Validation"  # Replace with the path to your validation data

# Define the image target size, batch size, and class mode
image_target_size = (100,100)
batch_size_0 = 32

# Create train and validation data generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=10,
    shear_range=0.2,
    height_shift_range=0.1,
    width_shift_range=0.1
    )

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_target_size,
    batch_size=32,
    class_mode="categorical"  # Set to "categorical" for classification tasks
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=10,
    shear_range=0.2,
    height_shift_range=0.1,
    width_shift_range=0.1
    )

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=image_target_size,
    batch_size=32,
    class_mode="categorical"
)

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt




# #Validate the data class
# class_names = list(generator.class_indices.keys())
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     train_images= image_array in generator
#     plt.imshow(train_images[i])
#     # The CIFAR labels happen to be arrays, 
#     # which is why you need the extra index
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()

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

#Compile and train the model
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit(train_generator, epochs=40, validation_data=validation_generator)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()


