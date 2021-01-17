"""circular_frame.py implements Frame Object."""

from __future__ import absolute_import, division, print_function

import math
from collections import deque
from functools import lru_cache
from time import time
from typing import List, Optional, Text, Tuple

import cv2
import numpy as np

import image_utils as image_utils
import utils

_DEFAULT_RADIUS = 250
_DEFAULT_MAX_THREADS = 4000
_DISPLAY_SCALE = 10


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

  def run(self, debug_display: bool = True) -> None:
    self._layers[0].run()


class _CircularLayer:

  def __init__(self,
               image: 'numpy.ndarray',
               radius: int = _DEFAULT_RADIUS,
               origin: Tuple[int, int] = (0, 0),
               image_origin: Tuple[int, int] = (0, 0),
               pins: List[Tuple[int, int]] = [],
               ignore_neighbor_ratio: float = 0.1,
               thread_intensity: int = 20,
               max_threads: int = _DEFAULT_MAX_THREADS,
               max_thread_length: Optional[float] = None) -> None:
    self.radius = radius
    self.origin = origin
    self.image = image
    self.image_origin = image_origin
    self.pins = pins

    self.thread_intensity = thread_intensity
    self.max_threads = max_threads
    self.max_thread_length = max_thread_length or (max_threads * 2 * radius)

    self.ignore_neighbor_ratio = ignore_neighbor_ratio
    self.current_pin_index = 0
    self.lengths = [image_utils.line_length(pins[0], pin) for pin in self.pins]
    self.last_visited_pins = utils.RingBuffer(size=20)

    self.build_working_images()

  def build_working_images(self) -> None:
    image = self.image
    if image_utils.channels(self.image) == 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    self.working_image = image_utils.circular_crop(image, self.radius,
                                                   self.image_origin)
    self.display_image = 255 * np.ones(
        (_DISPLAY_SCALE * image.shape[0], _DISPLAY_SCALE * image.shape[1]),
        dtype=np.uint8)

  def get_center(self) -> Tuple[int, int]:
    return self.origin[0] + self.radius, self.origin[1] + self.radius

  @lru_cache(maxsize=None)
  def get_linspace(self, src_index, dst_index):
    length = self.lengths[abs(src_index - dst_index)]
    p0, p1 = self.pins[src_index], self.pins[dst_index]
    lines = []
    for k in range(length):
      factor = float(k) / length
      x = int(p0[0] + factor * (p1[0] - p0[0]))
      y = int(p0[1] + factor * (p1[1] - p0[1]))
      lines.append([x, y])
    return lines

  def run(self, debug_display: bool = True) -> None:
    start = time()
    if debug_display:
      # Draw pins.
      for px, py in self.pins:
        px = (px + self.origin[0]) * _DISPLAY_SCALE
        py = (py + self.origin[1]) * _DISPLAY_SCALE
        cv2.circle(self.display_image, (px, py), 1 * _DISPLAY_SCALE,
                   (0, 255, 0), -1)
    done = False
    cnt = 0
    while not done:
      cnt += 1
      done = self.add_next_line() == 0
      if cnt % 15 == 0:
        print('processed {}/{} threads ...'.format(cnt, self.max_threads + cnt))
        if debug_display:
          display = cv2.resize(self.display_image, (1000, 1000))
          cv2.imshow("Display", display)
          cv2.imshow("Working image", self.working_image)
          cv2.waitKey(1)
    print('run time: ', time() - start)

  def get_next_pin(self) -> int:
    pin_cnt = len(self.pins)
    best_value = -1
    best_pin_index = -1
    image = self.working_image

    # Ignore neighboring pins to the current pin (if any).
    last_pins = set(self.last_visited_pins)
    i = self.current_pin_index
    ignore = int(len(self.pins) * self.ignore_neighbor_ratio) // 2
    for step in range(ignore, pin_cnt - ignore - 1, 1):
      j = (i + step + 1) % pin_cnt
      if j in last_pins:
        continue
      value = sum([255 - image[py, px] for px, py in self.get_linspace(i, j)])
      if value > best_value:
        best_value = value
        best_pin_index = j
    self.last_visited_pins.append(best_pin_index)
    return best_pin_index

  def add_next_line(self) -> int:
    if self.max_threads <= 0 or self.max_thread_length <= 0:
      return 0

    next_pin_index = self.get_next_pin()
    if next_pin_index == -1:
      return 0

    src = self.pins[self.current_pin_index]
    dst = self.pins[next_pin_index]
    length = self.lengths[abs(self.current_pin_index - next_pin_index)]
    for k in range(length):
      factor = float(k) / length
      px = int(src[0] + factor * (dst[0] - src[0]))
      py = int(src[1] + factor * (dst[1] - src[1]))
      # TODO(hamidb): compensate for high contrast areas.
      pixel = self.working_image[py, px] + self.thread_intensity
      self.working_image[py, px] = min(255, pixel)

    cv2.line(self.display_image,
             (src[0] * _DISPLAY_SCALE, src[1] * _DISPLAY_SCALE),
             (dst[0] * _DISPLAY_SCALE, dst[1] * _DISPLAY_SCALE), (0, 0, 0), 2,
             cv2.LINE_AA, 0)

    self.current_pin_index = next_pin_index
    self.max_threads -= 1
    self.max_thread_length -= length
    return length
