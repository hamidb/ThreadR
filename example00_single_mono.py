"""example00_single_mono.py creates a simple circular frame with one Monochrome
   layer.
"""

from __future__ import absolute_import, division, print_function

import cv2

import image_utils
import utils
from circular_frame import CircularFrame
from layers.monochrome_layer import MonochromeLayer

radius = 250
num_pins = 288

# Reading input image.
image = cv2.imread('data/images/pearl-girl-fg.png', cv2.IMREAD_UNCHANGED)
#  image[image[:, :, 3] == 0] = [255, 255, 255, 255]
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (2 * radius, 2 * radius))
cv2.imshow('Original image', image)

# Building the main circular frame.
frame = CircularFrame(radius, display_scale=3.0)

# Creating pins and adding a single circular layer.
pins = utils.compute_circular_pins(num_pins, radius, offset=(radius, radius))
frame.add_new_layer(
    MonochromeLayer(image,
                    radius,
                    layer_name='black',
                    image_origin=(radius, radius),
                    max_threads=3500,
                    thread_color=(0, 0, 0, 100),
                    correct_contrast=False,
                    pins=pins))
frame.run()
frame.clear()
print('Process finished. Press any key to exit.')
cv2.waitKey()
