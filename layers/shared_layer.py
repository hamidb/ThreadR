"""shared_layer.py implements SharedLayer object which is a specific type of a
  SimpleLayer. SharedLayer object shares common resources with the parent Frame.
  This is particularly useful and more efficient for cases where some layers
  have common resources such as working_image, pins, space_lines, etc.
"""

from __future__ import absolute_import, division, print_function

from typing import List

import cv2
import numpy as np

import typings as tp
import utils
from layers.simple_layer import SimpleLayer


class SharedLayer(SimpleLayer):
  """SharedLayer.
  This class is used for colorful layer when each layer shares working image
  with the main 'Frame'.
  """

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

  # Write-only properties.
  def image(self, image: tp.cvImage) -> None:
    self._image = image

  def working_image(self, working_image: tp.cvImage) -> None:
    self._working_image = working_image

  def line_spaces(self, line_spaces: List[List[tp.Point]]) -> None:
    self._line_spaces = line_spaces

  def lengths(self, lengths: List[int]) -> None:
    self._lengths = lengths

  def visited_pins(self, visited_pins: List[int]) -> None:
    self._visited_pins = visited_pins

  image = property(None, image)
  working_image = property(None, working_image)
  line_spaces = property(None, line_spaces)
  lengths = property(None, lengths)
  visited_pins = property(None, visited_pins)

  def update_image(self, src_idx: int, dst_idx: int) -> None:
    line_space = self._line_spaces[src_idx * self._pin_count + dst_idx]
    ratio = self._thread_intensity / 255.0
    color = [ratio * c for c in self._thread_color[:3]]
    for px, py in line_space:
      for c in range(utils.channels(self._working_image)):
        pixel = (1.0 - ratio) * self._working_image.item(py, px, c) + color[c]
        self._working_image.itemset((py, px, c), min(255, int(pixel)))
