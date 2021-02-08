"""example02_single_color.py creates a circular frame with one color layer. """

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np

import typings as tp
import utils
from circular_frame import CircularFrame
from layers.shared_layer import SharedLayer

num_pins = 270
radius = 250

# Reading input image.
image = cv2.imread('data/images/Lenna.jpg', cv2.IMREAD_UNCHANGED)
image = utils.circular_crop(image, radius, center=(radius, radius))
cv2.imshow('Original image', image)
cv2.waitKey(1)

# Creating pins and adding a single circular layer.
pins = utils.compute_circular_pins(num_pins, radius, offset=(radius, radius))
# Building the main circular frame.
frame = CircularFrame(radius=radius,
                      image=image,
                      pins=pins,
                      display_scale=2.0,
                      background_color=(255, 255, 255))

layer_settings = [
    ('black', [5, 4, 11, 60], 2000),  # #05040b
    ('red', [181, 2, 2, 60], 1500),  # #b50202
    ('blue', [13, 22, 189, 60], 1000),  #  #0d16bd
    ('yellow', [242, 229, 78, 60], 1000), # #f2e54e
]

for name, color, max_thread in layer_settings:
  color[0], color[2] = color[2], color[0]  # opencv works with BGR format.
  layer = SharedLayer(type=tp.COLOR_RGB,
                      name=name,
                      radius=radius,
                      image_origin=(radius, radius),
                      max_threads=max_thread,
                      thread_intensity=20,
                      thread_color=color,
                      norm_factor=0.007)
  frame.add_new_layer(layer)
frame.run()
frame.clear()
print('Process finished. Press any key to exit.')
cv2.waitKey()
