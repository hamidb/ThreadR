"""utils.py contains utility methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Tuple

import math


def compute_circular_pins(
    num_pins: int, radius: int,
    offset: Tuple[int, int] = (0, 0)) -> List[Tuple[int, int]]:
  pins = num_pins * [[0, 0]]
  for i in range(num_pins):
    rad = 2.0 * math.pi * i / num_pins
    pins[i] = (int(radius * math.cos(rad)) + offset[0],
               int(radius * math.sin(rad)) + offset[1])
  return pins
