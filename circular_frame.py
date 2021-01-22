"""circular_frame.py implements Frame Object."""

from __future__ import absolute_import, division, print_function

import math
import os
import uuid
from collections import deque
from datetime import datetime
from functools import lru_cache
from time import time
from typing import List, NewType, Optional, Text, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw

import image_utils as image_utils
import utils

_DEFAULT_RADIUS = 250
_DEFAULT_MAX_THREADS = 3000
_DISPLAY_SCALE = 4.0

Color = NewType('Color', Union[Tuple[int, int, int], Tuple[int, int, int, int]])
Layer = NewType('Layer', Union['MonochromeLayer', 'ColorLayer'])


class CircularFrame:

  def __init__(
      self,
      radius: int = _DEFAULT_RADIUS,
      max_threads: int = _DEFAULT_MAX_THREADS,
      max_thread_length: Optional[float] = None,
      background_color: Color = (255, 255, 255)  # background color for display.
  ) -> None:
    self.radius = radius
    self.max_threads = max_threads
    self.max_thread_length = max_thread_length or (max_threads * 2 * radius)

    self.layers = []
    size = int(_DISPLAY_SCALE * 2 * radius)
    self.frame_image = Image.new('RGB', (size, size), background_color)
    self.drawable = ImageDraw.Draw(self.frame_image, 'RGBA')

  def add_new_layer(self, layer: Layer) -> None:
    layer.set_frame_drawable(self.drawable)
    self.layers.append(layer)

  def run(self) -> None:
    # Draw pins.
    for layer in self.layers:
      for px, py in layer.pins:
        px = int((px + layer.origin[0]) * _DISPLAY_SCALE)
        py = int((py + layer.origin[1]) * _DISPLAY_SCALE)
        r = int(1 * _DISPLAY_SCALE)
        self.drawable.ellipse((px - r, py - r, px + r, py + r),
                              fill=(0, 128, 0, 255))

    processed_layers = set()
    total_threads = sum([l.max_threads for l in self.layers])
    while len(processed_layers) < len(self.layers):
      for layer in self.layers:
        if layer in processed_layers:
          continue
        if layer.thread_count >= layer.max_threads:
          processed_layers.add(layer)
          continue
        # Draw each layers' threads evenly.
        layer.run(max(1, int(15 * layer.max_threads / total_threads)))
        display = image_utils.pil_to_opencv(self.frame_image)
        display = cv2.resize(display, (1000, 1000))
        cv2.imshow('Frame Image', display)
        cv2.waitKey(1)

  def clear(self) -> None:
    self.layers.clear()
