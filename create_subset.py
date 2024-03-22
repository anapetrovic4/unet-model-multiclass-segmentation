import os
import numpy as np

# Create paths for images and labels
path_images = '/mnt/c/projects/unet-multiclass/krovovi/Standarised/images/train'
path_labels = '/mnt/c/projects/unet-multiclass/krovovi/Standarised/labels/train'

# Store rotation angles in a list
numbers = [45,90,135,180,225,270,315]

# Delete images that end with '45.png', etc.
def delete_images():
    for filename in os.listdir(path_images):
        for num in numbers:
            if filename.endswith(f'{num}.png'):
                os.remove(os.path.join(path_images, filename))
delete_images()

# Delete labels that end with '45.png', etc.
def delete_labels():
    for filename in os.listdir(path_labels):
        for num in numbers:
            if filename.endswith(f'{num}.png'):
                os.remove(os.path.join(path_labels, filename))
delete_labels()


