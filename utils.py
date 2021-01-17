"""utils.py contains utility methods."""

from __future__ import absolute_import, division, print_function

import math
from typing import List, Tuple


class RingBuffer:

  def __init__(self, size):
    self.size = size
    self.buffer = size * [None]
    self.idx = 0

  def __iter__(self):
    for i in self.buffer:
      yield i

  def __contains__(self, key):
    return key in self.buffer

  def append(self, value):
    self.buffer[self.idx % self.size] = value
    self.idx += 1


def compute_circular_pins(
    num_pins: int, radius: int,
    offset: Tuple[int, int] = (0, 0)) -> List[Tuple[int, int]]:
  pins = num_pins * [[0, 0]]
  radius -= 0.5
  for i in range(num_pins):
    rad = 2.0 * math.pi * i / num_pins
    pins[i] = (int(radius * math.cos(rad)) + offset[0],
               int(radius * math.sin(rad)) + offset[1])
  return pins
