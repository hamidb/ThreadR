"""simple_circular_frame.py creates a simple circular frame with one layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from circular_frame import CircularFrame
import utils
import image_utils
import cv2

radius = 100
num_pins = 180

# Reading input image (Grayscale).
image = cv2.imread('data/images/pearl-girl-red.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (200, 500))
cv2.imshow('image', image)
cv2.waitKey()

crop = image_utils.circular_crop(image, 100, center=(100, 250))
cv2.imshow('crop', crop)
cv2.waitKey()

# Building the main circular frame.
frame = CircularFrame(radius)

# Creating pins and adding a single circular layer.
pins = utils.compute_circular_pins(num_pins, radius, offset=(radius, radius))
frame.add_new_layer(radius, (-10, 10), image, (100, 250), pins)

frame.display('Main Frame')
cv2.waitKey()
