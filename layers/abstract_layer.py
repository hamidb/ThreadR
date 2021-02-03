"""abstract_layer.py implements AbstractLayer Object.
   Each Frame has 1 or more Layers.
"""

from __future__ import absolute_import, division, print_function

import math
import os
import uuid
from abc import ABCMeta, abstractmethod
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
Point = NewType('Point', Tuple[int, int])
cvImage = NewType('cvImage', 'np.ndarray')
PImage = NewType('PImage', 'Image')


class AbstractLayer(metaclass=ABCMeta):

  def __init__(
      self,
      image: cvImage,  # image for the layer.
      radius: int = _DEFAULT_RADIUS,  # layer radius.
      origin: Point = (0, 0),  # layer origin w.r.t the frame.
      image_origin: Point = (0, 0),  # image origin w.r.t the layer.
      pins: List[Point] = [],  # location of layer pins.
      ignore_neighbor_ratio: float = 0.1,  # ratio of adjacent pins to ignore.
      ignore_last_n_visited_pins: int = 20,  # don't revisit last n pins.
      thread_intensity: int = 20,  # how much darkness each line of thread adds.
      thread_color: Color = (0, 0, 0, 255),  # thread color for display.
      max_threads: int = _DEFAULT_MAX_THREADS,  # maximum lines of threads.
      max_thread_length: Optional[float] = None,  # maximum length of threads.
      alpha: float = 0.005,  # normalization factor to signify details vs shape.
      correct_contrast: bool = False,  # whether to penalty low contrast area.
      layer_name: Text = str(uuid.uuid4())  # name of layer.
  ) -> None:
    self.radius = radius
    self.origin = origin
    self.image = image
    self.image_origin = image_origin
    self.pins = pins

    self.thread_intensity = thread_intensity
    self.thread_color = thread_color
    self.max_threads = max_threads
    self.max_thread_length = max_thread_length or (max_threads * 2 * radius)
    self.thread_count = 0
    self.thread_length = 0
    self.alpha = alpha
    self.correct_contrast = correct_contrast

    self.ignore_neighbor_ratio = ignore_neighbor_ratio
    self.last_visited_pins = utils.RingBuffer(ignore_last_n_visited_pins)
    self.current_pin_index = 0

    self.frame_drawable = None
    self.layer_name = layer_name

    super().__init__()

  @abstractmethod
  def set_line_cost_func(self):
    pass

  @abstractmethod
  def build_working_images(self) -> None:
    pass

  @abstractmethod
  def run(self, max_iter: Optional[int] = None) -> None:
    pass

  @abstractmethod
  def add_next_line(self) -> int:
    pass

  def set_frame_drawable(self, drawable: PImage) -> None:
    self.frame_drawable = drawable

  def set_display_scale(self, display_scale: float) -> None:
    self.display_scale = display_scale

  def setup_layer(self) -> None:
    assert len(self.pins) >= 2, 'At least 2 pins are required to build a layer!'
    pins = self.pins
    self.lengths = [image_utils.line_length(pins[0], pin) for pin in pins]
    self.set_line_cost_func()
    self.build_working_images()
    # pre-compute linespaces for higher performance.
    self.linspaces = len(pins) * len(pins) * [None]
    for i in range(len(self.pins)):
      for j in range(i, len(self.pins)):
        lines = self.get_linspace(i, j)
        self.linspaces[i + j * len(pins)] = lines
        self.linspaces[j + i * len(pins)] = lines

  def get_center(self) -> Point:
    return self.origin[0] + self.radius, self.origin[1] + self.radius

  @lru_cache(maxsize=None)
  def get_linspace(self, src_index, dst_index):
    length = self.lengths[abs(src_index - dst_index)]
    p0, p1 = self.pins[src_index], self.pins[dst_index]
    lines = [[0, 0] for _ in range(length)]
    for i, k in enumerate(range(length)):
      factor = float(k) / length
      lines[i][0] = int(p0[0] + factor * (p1[0] - p0[0]))
      lines[i][1] = int(p0[1] + factor * (p1[1] - p0[1]))
    return lines

  def get_next_pin(self) -> int:
    pin_cnt = len(self.pins)
    best_value = -1
    best_pin_index = -1

    # Ignore neighboring pins to the current pin (if any).
    last_pins = set(self.last_visited_pins)
    i = self.current_pin_index
    ignore = int(len(self.pins) * self.ignore_neighbor_ratio) // 2
    size = len(self.pins)
    for step in range(ignore, pin_cnt - ignore - 1, 1):
      j = (i + step + 1) % pin_cnt
      if j in last_pins:
        continue
      linespace = self.linspaces[i + size * j]
      value = self.line_cost_func(self.image, self.working_image,
                                  self.thread_color, linespace)
      value = abs(self.alpha * value -
                  (1 - self.alpha) * value / len(linespace))
      if value > best_value:
        best_value = value
        best_pin_index = j
    self.last_visited_pins.append(best_pin_index)
    return best_pin_index

  def draw_line(self, src: Point, dst: Point) -> None:
    sx = int(self.display_scale * (src[0] + self.origin[0]))
    sy = int(self.display_scale * (src[1] + self.origin[1]))
    dx = int(self.display_scale * (dst[0] + self.origin[0]))
    dy = int(self.display_scale * (dst[1] + self.origin[1]))
    cv2.line(self.display_image, (sx, sy), (dx, dy), (0, 0, 0), 1, cv2.LINE_AA)
    if self.frame_drawable is not None:
      image_utils.draw_line_aa(self.frame_drawable, sx, sy, dx, dy,
                               tuple(self.thread_color))

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
