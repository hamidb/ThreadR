"""simple_circular_frame.py creates a simple circular frame with one layer."""

from __future__ import absolute_import, division, print_function

import cv2

import image_utils
import utils
from circular_frame import CircularFrame

radius = 250
num_pins = 288

# Reading input image.
image = cv2.imread('data/images/pearl-girl.png')
image = cv2.resize(image, (2 * radius, 2 * radius))
cv2.imshow('Original image', image)

# Building the main circular frame.
frame = CircularFrame(radius)

# Creating pins and adding a single circular layer.
pins = utils.compute_circular_pins(num_pins, radius, offset=(radius, radius))
# Shift the center with (50, 20) pixels for better appearance.
frame.add_new_layer(image,
                    radius,
                    image_origin=(radius + 50, radius + 20),
                    pins=pins)
frame.run()

print('Process finished. Press any key to exit.')
cv2.waitKey()
