import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Create dictionary
single_channel = {
    (0, 0, 0) : 0,
    (1, 1, 1) : 1,
    (2, 2, 2) : 2,
    (3, 3, 3) : 3,
    (4, 4, 4) : 4,
    (5, 5, 5) : 5,
    (6, 6, 6) : 6,
    (7, 7, 7) : 7,
    (8, 8, 8) : 8,
    (9, 9, 9) : 9,
}

image = cv2.imread('/mnt/c/projects/unet-multiclass/krovovi/Standarised/patches/labels_1_channel/mask_1_0_0.png')
plt.imshow(image)
plt.show()

input_dir = '/mnt/c/projects/unet-multiclass/krovovi/Standarised/patches/labels'
output_dir = '/mnt/c/projects/unet-multiclass/krovovi/Standarised/patches/labels_1_channel'

def convert_to_single_channel(input_dir, output_dir, map):
    
    # Iterate through files of masks
    for filename in os.listdir(input_dir):
        # Only include files that end with .png and create paths for input and output 
        if filename.endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Load each image
            image = cv2.imread(input_path)
            
            # If there is an image, create new matrix with single channel in 3rd dimension, and make it store integers
            if image is not None:
                single_channel_image = np.zeros_like(image[:,:,0], dtype=np.uint8)
                # Iterate through dictionary
                for key, value in single_channel.items():
                    # For all occurences check if image is equal to dict. key (we convert it in numpy array), and if it is True, we store boolean values in mask array
                    mask = np.all(image == np.array(key), axis=2)
                    # When the condition is met, assing the value to the array (Boolean indexing NumPy)
                    single_channel_image[mask] = value
                    
                cv2.imwrite(output_path, single_channel_image)
            
            else:
                print(f'Failed to load image from {input_path}')
    
convert_to_single_channel(input_dir, output_dir, single_channel)