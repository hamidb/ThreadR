"""simple_layer.py implements SimpleLayer object.
   Each Frame can have 1 or more SimpleLayer(s).
"""

from __future__ import absolute_import, division, print_function

import uuid
from typing import List, Optional, Text

import cv2
import numpy as np

import typings as tp
import utils


def monochrome_line_cost(src: tp.cvImage, dst: tp.cvImage, color: tp.Color,
                         iterator: List[tp.Point]) -> int:
  return sum([int(dst[py, px]) - int(src[py, px]) for px, py in iterator])


def rgb_color_line_cost(src: tp.cvImage, dst: tp.cvImage, color: tp.Color,
                        iterator: List[tp.Point]) -> int:
  value = 0
  c0, c1, c2 = color
  for px, py in iterator:
    value += utils.color_dist_sq(dst.item(py, px, 0), dst.item(py, px, 1),
                                 dst.item(py, px, 2), c0, c1, c2)
    value -= utils.color_dist_sq(src.item(py, px, 0), src.item(py, px, 1),
                                 src.item(py, px, 2), c0, c1, c2)
  return value


LINE_COST_FUNCTIONS = {
    'MONOCHROME_COST': monochrome_line_cost,
    'RGB_COLOR_COST': rgb_color_line_cost,
}


class SimpleLayer:

  def __init__(
      self,
      name: Text = str(uuid.uuid4()),  # name of layer.
      type: int = tp.MONOCHROME,
      image: Optional[tp.cvImage] = None,  # image for the layer.
      radius: int = 250,  # layer radius.
      origin: tp.Point = (0, 0),  # layer origin w.r.t the frame.
      image_origin: tp.Point = (0, 0),  # image origin w.r.t the layer.
      max_threads: int = 3000,  # maximum lines of threads.
      max_thread_length: Optional[float] = None,  # maximum length of threads.
      thread_intensity: int = 20,  # how much darkness each line of thread adds.
      thread_color: tp.Color = (0, 0, 0, 255),  # thread color for display.
      pins: List[tp.Point] = [],  # location of layer pins.
      skip_ratio: float = 0.1,  # ratio of adjacent pins to ignore.
      skip_last_n_pins: int = 20,  # don't revisit last n pins.
      norm_factor: float = 0.005  # norm factor to signify details vs shape.
  ) -> None:
    self._name = name
    self._type = type
    self._image = image
    self._radius = radius
    self._origin = origin
    self._image_origin = image_origin
    self._max_threads = max_threads
    self._max_thread_length = max_thread_length or float('inf')
    self._thread_intensity = thread_intensity
    self._thread_color = thread_color
    self._pins = pins
    self._skip_ratio = skip_ratio
    self._skip_last_n_pins = skip_last_n_pins
    self._norm_factor = norm_factor

    self._thread_count = 0
    self._thread_length = 0
    self._current_pin_index = 0
    self._pin_count = 0
    self._visited_pins = []
    self._line_lengths = []
    self._line_spaces = []
    self._line_cost = None
    self._working_image = None
    self._display_scale = None
    self._drawable = None

    super().__init__()

  # Read-only properties
  @property
  def origin(self) -> tp.Point:
    return self._origin

  @property
  def max_threads(self) -> int:
    return self._max_threads

  @property
  def thread_count(self) -> int:
    return self._thread_count

  @property
  def pins(self) -> List[tp.Point]:
    return self._pins

  @property
  def drawable(self) -> tp.PImage:
    return self._drawable

  @property
  def display_scale(self) -> float:
    return self._display_scale

  # Read-write properties
  @pins.setter
  def pins(self, pins: List[tp.Point]) -> None:
    self._pins = pins

  @drawable.setter
  def drawable(self, drawable: tp.PImage) -> None:
    self._drawable = drawable

  @display_scale.setter
  def display_scale(self, display_scale: float) -> None:
    assert display_scale >= 0, 'display_scale must be a non-negative float!'
    self._display_scale = display_scale

  def get_center(self) -> tp.Point:
    return self.origin[0] + self.radius, self.origin[1] + self.radius

  def setup_layer(self) -> None:
    assert self._image is not None, 'Source image is not provided!'
    assert self._pins is not None, 'Layer pins are not provided!'
    self._pin_count = len(self.pins)

    # Configurations for each specific type.
    if self._type & tp.COLOR_RGB:
      assert (utils.channels(self._image) >= 3 and
              'RGB(A) image is expected for a layer with type COLOR_RGB')
      self._line_cost = LINE_COST_FUNCTIONS['RGB_COLOR_COST']
    else:
      assert (utils.channels(self._image) == 1 and
              'GrayScale image is expected for a layer with type MONOCHROME')
      self._line_cost = LINE_COST_FUNCTIONS['MONOCHROME_COST']

    #TODO: support having origin and offset
    if self._working_image is None:
      self._working_image = 255 * np.ones(self._image.shape[:3], np.uint8)
    if len(self._visited_pins) == 0:
      self._visited_pins = self._pin_count * self._pin_count * [0]
    # Pre-compute line lengths and spaces for higher performance.
    if len(self._line_spaces) == 0 or len(self._lengths) == 0:
      self._line_spaces, self._lengths = utils.compute_lines(self._pins)

  def run(self, max_iter: Optional[int] = None) -> None:
    """run.
    Args:
        max_iter (Optional[int]): Specifies maximum iterations after which the
        function returns. If None, there will be no constraints on iterations.
    Returns:
    """
    iteration = 0
    while max_iter is None or iteration < max_iter:
      if self.add_next_line() == 0:
        break
      if self._thread_count % 15 == 0:
        print('processed {}/{} threads ...'.format(self._thread_count,
                                                   self._max_threads))
        cv2.imshow('Working image {}'.format(self._name), self._working_image)
      iteration += 1

  def add_next_line(self) -> int:
    if (self._thread_count >= self._max_threads or
        self._thread_length >= self._max_thread_length):
      return 0

    next_pin_index = self.get_next_pin()
    if next_pin_index == -1:
      return 0

    # Draw the line on working_image and frame drawable.
    i, j = self._current_pin_index, next_pin_index
    self.update_image(i, j)
    self.draw_line(self._pins[i], self._pins[j])

    length = self._lengths[abs(i - j)]
    self._thread_count += 1
    self._thread_length += length
    self._current_pin_index = next_pin_index
    return length

  def get_next_pin(self) -> int:
    # Ignore neighboring pins to the current pin (if any).
    ignore = int(self._pin_count * self._skip_ratio) // 2
    color = self._thread_color[:3]
    alpha = self._norm_factor
    best_value = -1
    best_pin_index = -1
    i = self._current_pin_index
    for step in range(ignore + 1, self._pin_count - ignore, 1):
      j = (i + step) % self._pin_count
      if self._visited_pins[i * self._pin_count + j] > 0:
        self._visited_pins[j * self._pin_count + i] -= 1
        self._visited_pins[i * self._pin_count + j] -= 1
        continue
      line_space = self._line_spaces[self._pin_count * i + j]
      value = self._line_cost(self._image, self._working_image, color,
                              line_space)
      value = alpha * value - (1 - alpha) * value / len(line_space)
      if value > best_value:
        best_value = value
        best_pin_index = j
    self._visited_pins[best_pin_index * self._pin_count + i] = 20
    self._visited_pins[i * self._pin_count + best_pin_index] = 20
    return best_pin_index

  def update_image(self, src_idx: int, dst_idx: int) -> None:
    for px, py in self._line_spaces[src_idx * self._pin_count + dst_idx]:
      pixel = int(self._working_image.item(py, px) - self._thread_intensity)
      self._working_image.itemset((py, px), max(0, pixel))

  def draw_line(self, src: tp.Point, dst: tp.Point) -> None:
    sx = int(self._display_scale * (src[0] + self._origin[0]))
    sy = int(self._display_scale * (src[1] + self._origin[1]))
    dx = int(self._display_scale * (dst[0] + self._origin[0]))
    dy = int(self._display_scale * (dst[1] + self._origin[1]))
    c0, c1, c2, c3 = self._thread_color
    if self._drawable is not None:
      utils.draw_line_aa(self._drawable, sx, sy, dx, dy, (c2, c1, c0, c3))
