"""color_circular_frame.py creates a circular frame with color layers."""

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np

import image_utils
import utils
from circular_frame import CircularFrame

radius = 250
num_pins = 288

# Reading input image.
image = cv2.imread('data/images/pearl-girl.png')
image = cv2.resize(image, (2 * radius, 2 * radius))
cv2.imshow('Original image', image)
cv2.waitKey(1)

# Building the main circular frame.
frame = CircularFrame(radius)

# Creating pins and adding a single circular layer.
pins = utils.compute_circular_pins(num_pins, radius, offset=(radius, radius))
layer_settings = [
    ('black', [5, 4, 11, 35], 2500),  # #05040b
    ('brown', [187, 157, 120, 35], 2500),  # #bb9d78
    ('blue', [0, 0, 255, 30], 1000),  #  #0000ff
    ('red', [255, 0, 0, 30], 1000),  # #FF0000
]

for name, color, max_thread in layer_settings:
  image_plane = image_utils.extract_color_plane(image, color)
  color[0], color[2] = color[2], color[0]  # opencv works with BGR format.

  frame.add_new_layer(image_plane,
                      radius,
                      layer_name=name,
                      image_origin=(radius, radius),
                      max_threads=max_thread,
                      thread_color=color,
                      thread_intensity=20,
                      correct_contrast=False,
                      pins=pins)
frame.run()
frame.clear()
print('Process finished. Press any key to exit.')
cv2.waitKey()
