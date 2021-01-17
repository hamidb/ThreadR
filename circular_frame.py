"""circular_frame.py implements Frame Object."""

from __future__ import absolute_import, division, print_function

import math
import os
from collections import deque
from datetime import datetime
from functools import lru_cache
from time import time
from typing import List, Optional, Text, Tuple

import cv2
import numpy as np

import image_utils as image_utils
import utils

_DEFAULT_RADIUS = 250
_DEFAULT_MAX_THREADS = 3000
_DISPLAY_SCALE = 12


def simple_line_cost(image: 'np.ndarray', iterator: List[Tuple[int, int]]):
  return sum([255 - int(image[py, px]) for px, py in iterator])


def line_cost_contrast_corrected(image: 'np.ndarray',
                                 iterator: List[Tuple[int, int]]):
  value = 0
  prev_intensity = image[iterator[0][1], iterator[0][0]]
  for px, py in iterator:
    pixel = int(image[py, px])
    change = abs(pixel - prev_intensity)
    prev_intensity = pixel
    value += 255 - pixel + change
  return value


LINE_COST_FUNCTIONS = {
    'SIMPLE_LINE_COST': simple_line_cost,
    'LINE_COST_CONTRAST_CORRECTED': line_cost_contrast_corrected
}


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

  def clear(self) -> None:
    self._layers.clear()


class _CircularLayer:

  def __init__(
      self,
      image: 'numpy.ndarray',  # image for the layer.
      radius: int = _DEFAULT_RADIUS,  # layer radius.
      origin: Tuple[int, int] = (0, 0),  # layer origin w.r.t the frame.
      image_origin: Tuple[int, int] = (0, 0),  # image origin w.r.t the layer.
      pins: List[Tuple[int, int]] = [],  # location of layer pins.
      ignore_neighbor_ratio: float = 0.1,  # ratio of adjacent pins to ignore.
      thread_intensity: int = 20,  # how much darkness each line of thread adds.
      max_threads: int = _DEFAULT_MAX_THREADS,  # maximum lines of threads.
      max_thread_length: Optional[float] = None,  # maximum length of threads.
      correct_contrast: bool = False  # whether to compensate low contrast area.
  ) -> None:
    self.radius = radius
    self.origin = origin
    self.image = image
    self.image_origin = image_origin
    self.pins = pins

    self.thread_intensity = thread_intensity
    self.max_threads = max_threads
    self.max_thread_length = max_thread_length or (max_threads * 2 * radius)
    self.thread_count = 0
    self.thread_length = 0

    self.ignore_neighbor_ratio = ignore_neighbor_ratio
    self.current_pin_index = 0
    self.lengths = [image_utils.line_length(pins[0], pin) for pin in self.pins]
    self.last_visited_pins = utils.RingBuffer(size=20)

    self.build_working_images()

    self.correct_contrast = correct_contrast
    self.line_cost_func = LINE_COST_FUNCTIONS['SIMPLE_LINE_COST']
    if self.correct_contrast:
      self.line_cost_func = LINE_COST_FUNCTIONS['LINE_COST_CONTRAST_CORRECTED']

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
        print('processed {}/{} threads ...'.format(self.thread_count,
                                                   self.max_threads))
        if debug_display:
          display = cv2.resize(self.display_image, (1000, 1000))
          cv2.imshow("Display", display)
          cv2.imshow("Working image", self.working_image)
          cv2.waitKey(1)
    print('run time: ', time() - start)
    now = datetime.now()

    # Writing output to a file.
    self.write_outputs()

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
      prev_intensity = image[self.pins[i][1], self.pins[i][0]]
      value = self.line_cost_func(image, self.get_linspace(i, j))
      if value > best_value:
        best_value = value
        best_pin_index = j
    self.last_visited_pins.append(best_pin_index)
    return best_pin_index

  def add_next_line(self) -> int:
    if (self.thread_count >= self.max_threads or
        self.thread_length >= self.max_thread_length):
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
      pixel = int(self.working_image[py, px]) + self.thread_intensity
      self.working_image[py, px] = min(255, pixel)

    self.draw_line(src, dst)

    self.current_pin_index = next_pin_index
    self.thread_count += 1
    self.thread_length += length
    return length

  def draw_line(self, src: Tuple[int, int], dst: Tuple[int, int]) -> None:
    sx = _DISPLAY_SCALE * (src[0] + self.origin[0])
    sy = _DISPLAY_SCALE * (src[1] + self.origin[1])
    dx = _DISPLAY_SCALE * (dst[0] + self.origin[0])
    dy = _DISPLAY_SCALE * (dst[1] + self.origin[1])
    cv2.line(self.display_image, (sx, sy), (dx, dy), (0, 0, 0), 2, cv2.LINE_AA)

  def write_outputs(self) -> None:
    if not os.path.exists('output'):
      os.mkdir('output')

    file_name_suffix = '{}-{}{}.png'.format(
        datetime.now().strftime('%d%m%YT%H-%M'), self.thread_count,
        '-corrected' if self.correct_contrast else '')

    display_image = cv2.resize(self.display_image, (1000, 1000))
    path = os.path.join('output', 'output-' + file_name_suffix)
    cv2.imwrite(path, display_image)
    print('Saved {}'.format(path))

    path = os.path.join('output', 'err-' + file_name_suffix)
    cv2.imwrite(path, self.working_image)
    print('Saved {}'.format(path))
