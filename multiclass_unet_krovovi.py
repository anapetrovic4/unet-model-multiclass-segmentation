from simple_multi_unet_model_krovovi import multi_unet_model
from keras.utils import normalize
import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import cv2

# Resizing images, if needed
SIZE_X = 256
SIZE_Y = 256 
n_classes = 10

images_dir = '/mnt/c/projects/unet-multiclass/krovovi/Standarised/patches2/images'
masks_dir = '/mnt/c/projects/unet-multiclass/krovovi/Standarised/patches2/labels'

# Capture training image and masks info as lists
images_list = [f for f in os.listdir(images_dir) if f.endswith('.png')]
masks_list = [f for f in os.listdir(masks_dir) if f.endswith('.png')]

# Open images and masks with OpenCV
images_data = []
for img in images_list:
    images_path = os.path.join(images_dir, img)
    # images_data = cv2.imread(images_path) # !!!!!! ovo svaki put pamti poslednju sliku i zato su dimenzije 3D, moras da uradis append u listu da bi dobila 4D
    images_data.append(cv2.imread(images_path))

masks_data = []
for msk in masks_list:
    masks_path = os.path.join(masks_dir, msk)
    # masks_data = cv2.imread(masks_path) !!!!!! ovo svaki put pamti poslednju sliku i zato su dimenzije 3D, moras da uradis append u listu da bi dobila 4D
    masks_data.append(cv2.imread(masks_path, cv2.IMREAD_GRAYSCALE))
    
# Convert lists to arrays for ML processing
images_data = np.array(images_data[:100])
masks_data = np.array(masks_data[:100])

print('Type of images ', type(images_data)) # numpy.ndarray
print('Type of masks ', type(masks_data)) # numpy.ndarray

print('Shape of images ', images_data.shape) # (50, 256, 256, 3)
print('Shape of masks ', masks_data.shape) # (50, 256, 256, 3) !!!!! podesi da maske imaju kanal 1

#############################################################

# Create subset of data

# Normalize train images from uint8 to float64 for easier processing
train_images = images_data
train_images = normalize(train_images, axis=1)
print('Shape of train images normalized ', train_images.shape)
train_masks_input = np.expand_dims(masks_data, axis=3)
print('Shape of train masks input ', train_masks_input.shape)

# Create a subset of data for quick testing
from sklearn.model_selection import train_test_split
x1, x_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size=0.10, random_state=0)

# Further split training data to a smaller subset for quick testing
x_train, x_do_not_use, y_train, y_do_not_use = train_test_split(x1, y1, test_size=0.2, random_state=0)

print('Shape of final train and test sets: ', x_train.shape, y_test.shape)
print('Shape of x test set: ', x_test.shape)
print("Class values in the dataset are", np.unique(y_train))

#############################################################

# One-hot-encoding: convert to categorical variables
from keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes)

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes)) # (160, 128, 128, 4)

print('Shape of train masks categorical ', y_train_cat.shape)
print('Shape of test masks categorical ', y_test_cat.shape)

#############################################################

# Since our dataset is not balanced, we use class_weight utility to balance our dataset. It looks how many pixels are assigned to each class, and based on that it gives weights
from sklearn.utils import class_weight

# Make 1D array from masks_data 
train_masks_reshaped = masks_data.reshape(-1)
print('Shape of train masks reshaped ', train_masks_reshaped.shape)
print('Unique values of train masks reshaped ', np.unique(train_masks_reshaped))

class_weights = class_weight.compute_class_weight('balanced', classes=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), y=train_masks_reshaped)
print('Class weights are: ', class_weights)

#############################################################

# Load the model
IMG_HEIGHT = x_train.shape[1]
IMG_WIDTH = x_train.shape[2]
IMG_CHANNELS = x_train.shape[3]

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH)

model = get_model()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', 
                       tf.keras.metrics.IoU(num_classes=n_classes, target_class_ids=[0]),
                       tf.keras.metrics.IoU(num_classes=n_classes, target_class_ids=[1]),
                       tf.keras.metrics.IoU(num_classes=n_classes, target_class_ids=[2]),
                       tf.keras.metrics.IoU(num_classes=n_classes, target_class_ids=[3]),
                       tf.keras.metrics.IoU(num_classes=n_classes, target_class_ids=[4]),
                       tf.keras.metrics.IoU(num_classes=n_classes, target_class_ids=[5]),
                       tf.keras.metrics.IoU(num_classes=n_classes, target_class_ids=[6]),
                       tf.keras.metrics.IoU(num_classes=n_classes, target_class_ids=[7]),
                       tf.keras.metrics.IoU(num_classes=n_classes, target_class_ids=[8]),
                       tf.keras.metrics.IoU(num_classes=n_classes, target_class_ids=[9]),
                       tf.keras.metrics.MeanIoU(num_classes=n_classes)
                      ]) 

model.summary()

history = model.fit(x_train, y_train_cat,
                    batch_size=2,
                    verbose=1,
                    epochs=10,
                    validation_data=(x_test, y_test_cat),
                    shuffle=False)

model.save('model.keras')

#############################################################

# Evaluate metrics

_, acc, iou_all_classes, a, b, c, d, e, f, g, h, i, j = model.evaluate(x_test, y_test_cat)
print('Accuracy is = ', (acc * 100.0), '%')

print(iou_all_classes)
print(a)
print(b)
print(c)
print(d)
print(e)
print(f)
print(g)
print(h)
print(i)
print(j)

#############################################################

# Plot the training and validation accuracy and loss at each point

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'y', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#############################################################

# Load the weights

model.load_weights('model.keras')

# Predict

y_pred = model.predict(x_test)
y_pred_argmax = np.argmax(y_pred, axis=3) # returns a class with the highest probability

# Calculate IoU

from keras.metrics import MeanIoU
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print('Mean IoU: ', IOU_keras.result().numpy())

plt.imshow(train_images[0,:,:,0], cmap='gray')
plt.imshow(masks_data[0], cmap = 'gray')

m1 = tf.keras.metrics.IoU(num_classes=n_classes, target_class_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

confusion_matrix = m1.variables[0].numpy()

class_iou = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix))
for i in range(n_classes):
    print(f"IoU for class {i+1} is: {class_iou[i]}")

#############################################################

# Predict on a few images

import random

test_img_number = random.randint(0, len(x_test) - 1) # Create random number
test_img = x_test[test_img_number] # Find test image in this position from x test
ground_truth = y_test[test_img_number] # Find mask in this position from y test
test_img_norm = test_img / 255.0
test_img_input = np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input))
predicted_img = np.argmax(prediction, axis=3)[0,:,:]

plt.figure(figsize=(12,8))
plt.subplot(231)
plt.title('Testing image')
plt.imshow(test_img[:,:,0], cmap='gray') 
plt.subplot(232)
plt.title('Testing label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='jet')
plt.show()

#############################################################
