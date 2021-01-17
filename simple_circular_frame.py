"""simple_circular_frame.py creates a simple circular frame with one layer."""

from __future__ import absolute_import, division, print_function

import cv2

import image_utils
import utils
from circular_frame import CircularFrame

radius = 250
num_pins = 288

# Reading input image (Grayscale).
image = cv2.imread('data/images/pearl-girl-red.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (500, 500))
cv2.imshow('Original image', image)

# Building the main circular frame.
frame = CircularFrame(radius)

# Creating pins and adding a single circular layer.
pins = utils.compute_circular_pins(num_pins, radius, offset=(radius, radius))
frame.add_new_layer(image, radius, (0, 0), (250, 250), pins)
frame.run()
cv2.waitKey()
