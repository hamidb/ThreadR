"""circular_frame.py implements Frame Object."""

from __future__ import absolute_import, division, print_function

import math
from typing import List, Optional, Text, Tuple

import cv2
import numpy as np

import image_utils as utils

_DEFAULT_RADIUS = 250
_DEFAULT_NUM_PINS = 180
_DEFAULT_MAX_THREADS = 5000


class CircularFrame:
  _layers = []

  def __init__(self,
               radius: int = _DEFAULT_RADIUS,
               max_threads: int = _DEFAULT_MAX_THREADS,
               max_thread_length: Optional[float] = None) -> None:
    self.radius = radius
    self.max_threads = max_threads
    self.max_thread_length = max_thread_length or (max_threads * 2 * radius)

  @classmethod
  def add_new_layer(cls, *args, **kwargs) -> '_CircularLayer':
    layer = _CircularLayer(*args, **kwargs)
    cls._layers.append(layer)
    return layer

  def display(self, window_name: Text = '') -> None:
    frame = np.zeros((2 * self.radius, 2 * self.radius, 3), dtype=np.uint8)
    layer = self._layers[0]
    x1, y1 = layer.origin
    x2, y2 = x1 + 2 * layer.radius, y1 + 2 * layer.radius
    for i in range(3):
      frame[:, :, i] = utils.copy_to_roi(layer.working_image, frame[:, :, i],
                                         [x1, y1, x2, y2])

    cv2.circle(frame, layer.get_center(), layer.radius, (255, 0, 0), 1)

    # Draw pins
    origin = layer.origin
    for px, py in layer.pins:
      cv2.circle(frame, (px + origin[0], py + origin[1]), 1, (0, 255, 0), -1)

    cv2.imshow(window_name, frame)
    cv2.waitKey()
    layer.run()


class _CircularLayer:

  def __init__(self,
               radius: int = _DEFAULT_RADIUS,
               origin: Tuple[int, int] = (0, 0),
               image_plane: Optional['numpy.ndarray'] = None,
               image_plane_origin: Tuple[int, int] = (0, 0),
               pins: List[Tuple[int, int]] = [],
               thread_width: int = 1,
               thread_color: Tuple[int, int, int] = (0, 0, 0),
               thread_intensity: int = 10,
               max_threads: int = _DEFAULT_MAX_THREADS,
               max_thread_length: Optional[float] = None,
               ignore_neighbor_ratio: float = 0.1) -> None:
    self.radius = radius
    self.origin = origin
    self.image_plane = image_plane
    self.image_plane_origin = image_plane_origin
    self.pins = pins
    self.thread_width = thread_width
    self.thread_color = thread_color
    self.thread_intensity = thread_intensity
    self.max_threads = max_threads
    self.max_thread_length = max_thread_length or (max_threads * 2 * radius)
    self.ignore_neighbor_ratio = ignore_neighbor_ratio
    self.ignore_neighbor_pins = int(len(pins) * ignore_neighbor_ratio)
    self.cropped_image = self.build_working_image()
    self.working_image = self.cropped_image
    self.current_pin_index = 0
    self.lengths = [utils.line_length(pins[0], pin) for pin in self.pins]

  def build_working_image(self) -> 'np.ndarray':
    return utils.circular_crop(self.image_plane, self.radius,
                               self.image_plane_origin)

  def get_center(self) -> Tuple[int, int]:
    return self.origin[0] + self.radius, self.origin[1] + self.radius

  def run(self) -> None:
    # Find next pin (dst) to connect.
    pin = self.get_next_pin()
    cv2.circle(self.working_image, self.pins[0], 1, (255, 255, 255), -1)
    cv2.line(self.working_image, self.pins[0], pin, (255, 255, 255), 5)
    cv2.imshow("ddd", self.working_image)

  def get_next_pin(self) -> Tuple[int, int]:
    pin_cnt = len(self.pins)
    image = self.working_image
    best_value = 0
    best_pin_index = -1

    # ignore neighboring pins to the current pin (if any).
    i = self.current_pin_index
    j = i + (self.ignore_neighbor_pins // 2)
    for _ in range(pin_cnt - self.ignore_neighbor_pins - 1):
      j = (j + 1) % pin_cnt
      src, dst = self.pins[i], self.pins[j]
      length = self.lengths[abs(i - j)]
      value = utils.line_similarity(image, src, dst, self.thread_color, length)
      if value > best_value:
        best_value = value
        best_pin_index = j

    if best_pin_index != -1:
      self.current_pin_index = best_pin_index
    return self.pins[self.current_pin_index]




