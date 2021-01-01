"""circular_frame.py implements Frame Object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Optional, Text, Tuple

import image_utils as utils
import cv2
import numpy as np

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

    cv2.imshow(window_name, frame)


class _CircularLayer:

  def __init__(self,
               radius: int = _DEFAULT_RADIUS,
               origin: Tuple[int, int] = (0, 0),
               image_plane: Optional['numpy.ndarray'] = None,
               image_plane_origin: Tuple[int, int] = (0, 0),
               pins: List[Tuple[int, int]] = [],
               max_threads: int = _DEFAULT_MAX_THREADS,
               max_thread_length: Optional[float] = None,
               ignore_neighbor_ratio: float = 0.0) -> None:
    self.radius = radius
    self.origin = origin
    self.image_plane = image_plane
    self.image_plane_origin = image_plane_origin
    self.pins = pins
    self.max_threads = max_threads
    self.max_thread_length = max_thread_length or (max_threads * 2 * radius)
    self.working_image = self.build_working_image()

  def build_working_image(self) -> 'np.ndarray':
    return utils.circular_crop(self.image_plane, self.radius,
                               self.image_plane_origin)

  def get_center(self) -> Tuple[int, int]:
    return self.origin[0] + self.radius, self.origin[1] + self.radius
