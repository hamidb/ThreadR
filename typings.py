"""typings.py contains user types and constants."""

from __future__ import absolute_import, division, print_function

from typing import List, NewType, Optional, Text, Tuple, Union

######## User defined type Annotations ########

# 2D point type.
Point = NewType('Point', Tuple[int, int])
# ROI type for bounding box.
Roi = NewType('Roi', Tuple[int, int, int, int])
# Color type for RGB or RGBA.
Color = NewType('Color', Union[Tuple[int, int, int], Tuple[int, int, int, int]])
# Opencv type for image (np.ndarray).
cvImage = NewType('cvImage', 'np.ndarray')
# PIL Image type.
PImage = NewType('PImage', 'Image')
# Different layer types.
Layer = NewType('Layer', Union['SimpleLayer', 'SharedLayer'])

######## Flags and constants ########

# Layer flags
MONOCHROME = 0 << 0
COLOR_RGB = 1 << 0
COST_COMPENSATE_CONTRAST = 1 << 2
COST_COMPENSATE_BORDERS = 1 << 3


