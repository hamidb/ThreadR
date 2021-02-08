"""example00_single_mono.py creates a simple circular frame with one Monochrome
   layer.
"""

from __future__ import absolute_import, division, print_function

import cv2

import utils
from circular_frame import CircularFrame
from layers.simple_layer import SimpleLayer

num_pins = 270
radius = 250

# Reading input image.
image = cv2.imread('data/images/pearl-girl-fg.png', cv2.IMREAD_UNCHANGED)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (2 * radius, 2 * radius))
cv2.imshow('Original image', image)

# Building the main circular frame.
frame = CircularFrame(radius, display_scale=3.0)

# Creating pins and adding a single circular layer.
pins = utils.compute_circular_pins(num_pins, radius, offset=(radius, radius))
frame.add_new_layer(
    SimpleLayer(image=image,
                name='black',
                radius=radius,
                image_origin=(radius, radius),
                max_threads=4000,
                thread_intensity=25,
                thread_color=(0, 0, 0, 70),
                norm_factor=0.006,
                pins=pins))
frame.run()
frame.clear()
print('Process finished. Press any key to exit.')
cv2.waitKey()
