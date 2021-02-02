"""monochrome_layer.py implements Monochrome Layer."""

from __future__ import absolute_import, division, print_function

import math
from typing import List, NewType, Optional, Tuple, Union

import cv2
import numpy as np

import image_utils as image_utils
from layers.abstract_layer import AbstractLayer

cvImage = NewType('cvImage', 'np.ndarray')
Point = NewType('Point', Tuple[int, int])
Color = NewType('Color', Union[Tuple[int, int, int], Tuple[int, int, int, int]])


def simple_line_cost(src: cvImage, dst: cvImage, color: Color,
                     iterator: List[Point]) -> int:
  value = sum([int(dst[py, px]) for px, py in iterator])
  return -value


def line_cost_corrected(src: cvImage, dst: cvImage, color: Color,
                        iterator: List[Point]) -> int:
  value = 0
  prev_intensity = int(src[iterator[0][1], iterator[0][0]])
  for px, py in iterator:
    src_p, dst_p = int(src[py, px]), int(dst[py, px])
    change = abs(prev_intensity - src_p)
    value += src_p - dst_p + change
    prev_intensity = src_p
  return value


LINE_COST_FUNCTIONS = {
    'SIMPLE_LINE_COST': simple_line_cost,
    'LINE_COST_CORRECTED': line_cost_corrected,
}


class MonochromeLayer(AbstractLayer):

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

  def build_working_images(self) -> None:
    assert image_utils.channels(
        self.image) == 1, 'Monochrome layer only takes grayscale image!'
    image = self.image
    self.working_image = ~image
    if self.correct_contrast:
      self.working_image = 255 * np.ones(image.shape, dtype=np.uint8)
    w = int(image.shape[1] * self.display_scale)
    h = int(image.shape[0] * self.display_scale)
    self.display_image = 255 * np.ones((h, w), dtype=np.uint8)

  def set_line_cost_func(self) -> None:
    self.line_cost_func = LINE_COST_FUNCTIONS['SIMPLE_LINE_COST']
    if self.correct_contrast:
      self.line_cost_func = LINE_COST_FUNCTIONS['LINE_COST_CORRECTED']

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
      pixel = int(self.working_image[py, px]) - self.thread_intensity
      self.working_image[py, px] = max(0, pixel)

    self.draw_line(src, dst)

    self.current_pin_index = next_pin_index
    self.thread_count += 1
    self.thread_length += length
    return length
