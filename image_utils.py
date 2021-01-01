"""image_utils.py implements image utility methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Optional, Text, Tuple

import cv2
import numpy as np


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
