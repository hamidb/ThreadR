"""example01_multi_mono.py creates a circular frame with multiple mono layers.
"""

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np

import image_utils
import utils
from circular_frame import CircularFrame
from layers.monochrome_layer import MonochromeLayer

radius = 250
num_pins = 288

# Reading input image.
image = cv2.imread('data/images/Lenna.jpg')
image = cv2.resize(image, (2 * radius, 2 * radius))
cv2.imshow('Original image', image)
cv2.waitKey(1)

# Building the main circular frame.
frame = CircularFrame(radius, background_color=(240, 240, 240))

# Creating pins and adding a single circular layer.
pins = utils.compute_circular_pins(num_pins, radius, offset=(radius, radius))

layer_settings = [
    ('black', [5, 4, 11, 30], 2000),  # #05040b
    ('yellow', [242, 229, 78, 30], 1500),  # #f2e54e
    ('blue', [13, 22, 189, 30], 2000),  #  #0d16bd
    ('red', [181, 2, 2, 30], 3000),  # #b50202
]

for name, color, max_thread in layer_settings:
  image_plane = image_utils.extract_color_plane(image, color)
  color[0], color[2] = color[2], color[0]  # opencv works with BGR format.

  frame.add_new_layer(
      MonochromeLayer(image_plane,
                      radius,
                      layer_name=name,
                      image_origin=(radius, radius),
                      max_threads=max_thread,
                      thread_color=color,
                      thread_intensity=20,
                      correct_contrast=True,
                      pins=pins))
frame.run()
frame.clear()
print('Process finished. Press any key to exit.')
cv2.waitKey()
exit()
