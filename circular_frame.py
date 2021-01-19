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
_DISPLAY_SCALE = 2.0

Color = NewType('Color', Union[Tuple[int, int, int], Tuple[int, int, int, int]])
Point = NewType('Point', Tuple[int, int])
cvImage = NewType('cvImage', 'np.ndarray')


def simple_line_cost(image: cvImage, iterator: List[Point]):
  return sum([255 - int(image[py, px]) for px, py in iterator])


def line_cost_corrected(image: cvImage, iterator: List[Point]):
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
    'LINE_COST_CORRECTED': line_cost_corrected,
}


class CircularFrame:

  def __init__(self,
               radius: int = _DEFAULT_RADIUS,
               max_threads: int = _DEFAULT_MAX_THREADS,
               max_thread_length: Optional[float] = None) -> None:
    self.radius = radius
    self.max_threads = max_threads
    self.max_thread_length = max_thread_length or (max_threads * 2 * radius)

    self.layers = []
    size = int(_DISPLAY_SCALE * 2 * radius)
    self.frame_image = Image.new('RGB', (size, size), (255, 255, 255))
    self.drawable = ImageDraw.Draw(self.frame_image, 'RGBA')

  def add_new_layer(self, *args, **kwargs) -> 'CircularLayer':
    layer = CircularLayer(*args, **kwargs)
    layer.set_frame_drawable(self.drawable)
    self.layers.append(layer)
    return layer

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


class CircularLayer:

  def __init__(
      self,
      image: cvImage,  # image for the layer.
      radius: int = _DEFAULT_RADIUS,  # layer radius.
      origin: Point = (0, 0),  # layer origin w.r.t the frame.
      image_origin: Point = (0, 0),  # image origin w.r.t the layer.
      pins: List[Point] = [],  # location of layer pins.
      ignore_neighbor_ratio: float = 0.1,  # ratio of adjacent pins to ignore.
      thread_intensity: int = 20,  # how much darkness each line of thread adds.
      thread_color: Color = (0, 0, 0, 1),  # thread color for display.
      max_threads: int = _DEFAULT_MAX_THREADS,  # maximum lines of threads.
      max_thread_length: Optional[float] = None,  # maximum length of threads.
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

    self.ignore_neighbor_ratio = ignore_neighbor_ratio
    self.current_pin_index = 0
    self.lengths = [image_utils.line_length(pins[0], pin) for pin in self.pins]
    self.last_visited_pins = utils.RingBuffer(size=20)

    self.layer_name = layer_name

    self.build_working_images()
    self.frame_drawable = None

    self.correct_contrast = correct_contrast
    self.line_cost_func = LINE_COST_FUNCTIONS['SIMPLE_LINE_COST']
    if self.correct_contrast:
      self.line_cost_func = LINE_COST_FUNCTIONS['LINE_COST_CORRECTED']

  def set_frame_drawable(self, drawable: cvImage) -> None:
    self.frame_drawable = drawable

  def build_working_images(self) -> None:
    image = self.image
    if image_utils.channels(self.image) == 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    self.working_image = image_utils.circular_crop(image, self.radius,
                                                   self.image_origin)
    w, h = image.shape[1] * _DISPLAY_SCALE, image.shape[0] * _DISPLAY_SCALE
    self.display_image = 255 * np.ones((int(h), int(w)), dtype=np.uint8)

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

  def run(self, max_iter: Optional[int] = None) -> None:
    """run.

    Args:
        max_iter (Optional[int]): Specifies maximum iterations after which the
        function returns. If None, there will be no constraints on iterations.

    Returns:
        None:
    """
    iteration = 0
    while max_iter is None or iteration < max_iter:
      if self.add_next_line() == 0:
        break
      if self.thread_count % 15 == 0:
        print('processed {}/{} threads ...'.format(self.thread_count,
                                                   self.max_threads))
        cv2.imshow('Working image {}'.format(self.layer_name),
                   self.working_image)
      iteration += 1

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

  def draw_line(self, src: Point, dst: Point) -> None:
    sx = int(_DISPLAY_SCALE * (src[0] + self.origin[0]))
    sy = int(_DISPLAY_SCALE * (src[1] + self.origin[1]))
    dx = int(_DISPLAY_SCALE * (dst[0] + self.origin[0]))
    dy = int(_DISPLAY_SCALE * (dst[1] + self.origin[1]))
    cv2.line(self.display_image, (sx, sy), (dx, dy), (0, 0, 0), 1, cv2.LINE_AA)
    if self.frame_drawable is not None:
      self.frame_drawable.line((sx, sy) + (dx, dy),
                               fill=tuple(self.thread_color))

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
