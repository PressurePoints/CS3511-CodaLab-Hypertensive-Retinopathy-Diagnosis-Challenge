import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

class CLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        # apply CLAHE to the Y channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])

        # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output

def test_image_transform(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Apply the CLAHE transformation
    clahe_transform = CLAHE()
    img_clahe = clahe_transform(img)

    # Convert the images to RGB format for displaying
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_clahe_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)

    # Display the original and transformed images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(img_clahe_rgb)
    plt.title('CLAHE Image')

    plt.show()

# Test the function
images_dir = os.path.join('./data-raw', '1-Images', '1-Training Set', '00000ce9.png')
test_image_transform(images_dir)