"""image_utils.py implements image utility methods."""

from __future__ import absolute_import, division, print_function

import math
from typing import List, Optional, Text, Tuple

import cv2
import numpy as np


def channels(image: 'np.ndarray') -> int:
  return image.shape[2] if len(image.shape) >= 3 else 1


def line_length(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]
  return int(math.sqrt(dx * dx + dy * dy))


def color_similarity_mono(c1: int, c2: int) -> int:
  return math.sqrt((c1 - c2) * (c1 - c2))


def color_similarity(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> int:
  diff0 = c1[0] - c2[0]
  diff1 = c1[1] - c2[1]
  diff2 = c1[2] - c2[2]
  return math.sqrt(diff0 * diff0 + diff1 * diff1 + diff2 * diff2)


def line_similarity(image: 'np.ndarray',
                    src: Tuple[int, int],
                    dst: Tuple[int, int],
                    color: Tuple[int, int, int],
                    length: int = -1) -> float:
  # compute length if not given.
  length = length if length >= 0 else line_length(src, dst)
  value = 0
  for k in range(length):
    factor = float(k) / length
    px = int(src[0] + factor * (dst[0] - src[0]))
    py = int(src[1] + factor * (dst[1] - src[1]))
    # TODO(hamidb): compensate for high contrast areas.
    if channels(image) == 3:
      value += 255 - color_similarity(image[py, px, :], color)
    else:
      avg = sum(color) // 3
      print(sum(color), color)
      value += 255 - color_similarity_mono(image[py, px], avg)
  return value / length if length else value


def crop(image: 'np.ndarray', box: Tuple[int, int, int, int]) -> 'np.ndarray':
  x1, y1, x2, y2 = box
  if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
    image, x1, x2, y1, y2 = pad_img_to_fit_bbox(image, x1, x2, y1, y2)
  return image[y1:y2, x1:x2]


def pad_img_to_fit_bbox(image: 'np.ndarray', x1: int, x2: int, y1: int,
                        y2: int) -> Tuple['np.ndarray', int, int, int, int]:
  h, w = image.shape[:2]
  image = cv2.copyMakeBorder(image, -min(0, y1), max(y2 - h, 0), -min(0, x1),
                             max(x2 - w, 0), cv2.BORDER_CONSTANT)
  y2 += -min(0, y1)
  y1 += -min(0, y1)
  x2 += -min(0, x1)
  x1 += -min(0, x1)
  return image, x1, x2, y1, y2


def circular_crop(
    image: 'np.ndarray', radius: float,
    center: Tuple[int, int] = (0, 0)) -> 'np.ndarray':
  x1, y1 = center[0] - radius, center[1] - radius
  x2, y2 = center[0] + radius, center[1] + radius
  # crop 2rx2r box from the center.
  cropped = crop(image, [x1, y1, x2, y2])
  # create a mask with a circular ones(255).
  mask = np.zeros((2 * radius, 2 * radius), dtype=np.uint8)
  cv2.circle(mask, (radius, radius), radius, (255,), -1, None, 0)
  return cv2.bitwise_and(cropped, mask)


def copy_to_roi(src: 'np.ndarray', dst: 'np.ndarray',
                roi: Tuple[int, int, int, int]) -> 'np.ndarray':
  assert (roi[0] <= roi[2] and roi[1] <= roi[3])
  dst_h, dst_w = dst.shape[:2]
  dst_x1, dst_y1 = max(0, roi[0]), max(0, roi[1])
  dst_x2, dst_y2 = min(dst_w, roi[2]), min(dst_h, roi[3])
  src_x1, src_y1 = dst_x1 - roi[0], dst_y1 - roi[1]
  src_x2, src_y2 = dst_x2 - roi[0], dst_y2 - roi[1]
  dst[dst_y1:dst_y2, dst_x1:dst_x2] = src[src_y1:src_y2, src_x1:src_x2]
  return dst
