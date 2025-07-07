#### Load python libraries ####
import matplotlib.pyplot as plt # you're just importing this library under an alias 
import seaborn as sns


import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf

import cv2
import os

import numpy as np



#### Loading the data ####
lables = ['tumor', 'normal'] # Creating labels
img_size = 224 # Defining the size of the image you want
def get_data(data_dir): # Creating a function to load the data
  data = [] # creating an empty 
  for label in labels:
    path = ps.path.join(data_dir, label) # creating path where files are located 
    class_num = labels.index(label)
    for img in os.listdir(path):
      try:
        img_arr = cv2.imread(os.path.join(path, img))[...,::-1] # convert BGR to RGB format
        resized_arr = cv2.resize(iimg_arr, (img_size, img_size)) # Reshaping images 
        data.append([resized_arr, class_num])
     except Exception as e:
        print(e)
  return np.array(data)

### There are 4,602 images total (both tumor and normal). Save 3202 of those images for training (create a folder named 'training'), and the last 1400 for validation (create a folder named 'testing').
train = get_data(r"\\ifs.win.uthscsa.edu\M1509-AhujaS\MainShare\Lois\Lois_Local\Dr_M\Projects\pipeline_development\Rashmi\Image_Classification\Training")
test = get_data(r"\\ifs.win.uthscsa.edu\M1509-AhujaS\MainShare\Lois\Lois_Local\Dr_M\Projects\pipeline_development\Rashmi\Image_Classification\Testing")

### Example of what the train and test data will look like:
### train: [image_array, label]
### test: [image_array, label]

### Python indices used
### 0 - extracts first column.
### 1 - extracts 2nd column.
### -1 - extracts last column.
### -2 - extracts 2nd to last column.


#### Visualize data (optional) ####
l = [] # creating empty list
for i in train:
  if(i[1] == 0: # indexing the 2nd column from 'train'
    l.append("tumor") # adding all images labeled 'tumor' to list named 'l'
  else
    l.append("normal") # adding all images labeled 'normal' to list named 'l'
sns.set_style('darkgrid') # setting the background of the plot dark gray
sns.counterplot(l) # visualizing list l

plt.figure(figsize = (5,5)) # setting the size of your image 5 x 5 inches (heigth, width)
plt.imshow(train[1][0])
plt.title(labels[train[0][1]])



#### Data preprocessing and Data Augmentation ####
x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train: 
  x_train.append(feature) # adding feature data to 'x_train' (e.g., colors, edges, textures all represented as pixel values)
  y_train.append(label) # adding labels to y_train (e.g., 'tumor', 'normal')

for feature, label in train in val:
  x_val.append(feature)
  y_val.append(label)

#### Normalize the data ####
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)


#### Data augmentation on the train data ####
datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range = 30,
        zoom_range = 0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip = True,
        vertical_flip=False)

datagen.fit(x_train)


#### Define the model ####
model = Sequential () # creating a new neural network model
model.add(Conv2D(32, 3, pading="same", activation="relu", input_shape=(224,224,3))) # adding a convolutional layer 
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="SoftMax"))

model.summary()


#### Compiling model ####
opt = Adam(lr=0.000001)
model.compile(optimizer = opt, loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])


#### Train model ####
history = model.fit(x_train, y_train, epochs = 500, validation_data = (x_val, y_val))

acc = history.history['accuracy']
val_acc = history.history[val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(500) # creating a range from 0-499

#### Plotting model performance ####
plt.figure(figsize=(15,15)) # plot will be 15x15 inches
plt.subplot(2, 2, 1) # plt.subplot(nrows, ncols, index); using a 2x2 layout and plotting in the upper left corner
plt.plot(epochs_range, acc, label='Training Accuracy') # plotting accuracy measures for training data across all epochs (500 total)
plt.plot(epochs_range, val_acc='Validation Accuracy') # plotting accuracy measures for validation data across all epochs (500 total)
plt.legend(loc='lower right') # the location where you want you legend
plt.title('Training and Validation Accuracy') # title of your plot

plt.subplot(2, 2, 2) # using 2x2 layout and plotting in upper right hand corner
plt.plot(epochs_range, loss, label='Training Loss') # plotting loss measures for training data across all epochs (500 total)
plt.plot(epochs_range, val_loss, label='Validation Loss') # plotting loss measures for validation data across all epochs (500 total)
plt.legend(loc='upper right') # the location where you want your legend
plt.title('Training and Validation Loss') # title of your plot
plt.show() # show the plot 


#### print out predictions ####
predictions = model.predict_classes(x_val)
predictions = predictions.reshape(1,-1)[0]
