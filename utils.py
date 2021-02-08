"""utils.py contains utility methods."""

from __future__ import absolute_import, division, print_function

import math
from typing import List, NewType, Optional, Text, Tuple, Union
import cv2
import numpy as np
from PIL import Image

Color = NewType('Color', Union[Tuple[int, int, int], Tuple[int, int, int, int]])
Point = NewType('Point', Tuple[int, int])
Roi = NewType('Roi', Tuple[int, int, int, int])
cvImage = NewType('cvImage', 'np.ndarray')
PImage = NewType('PImage', 'Image')


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


def line_length(p1: Point, p2: Point) -> int:
  dx, dy = p2[0] - p1[0], p2[1] - p1[1]
  return int(math.sqrt(dx * dx + dy * dy))


def compute_lines(pins: List[Point]) -> Tuple[List[Point], List[int]]:
  """compute_lines.
  For performance reasons, this pre-computes line spaces and lengths between
  each 2 points. For example, to find line_space between two points in pins:
  line_spaces, lengths = build_linespaces(pins)
  line_space_ij = line_space_ji = line_spaces[i+j*len(pins)]

  Args:
      pins (List[Point]): List of pins' 2D location.
  Returns:
    Tuple[List[Point], List[int]]: line_spaces, lengths
  """
  size = len(pins)
  lengths = [line_length(pins[0], p) for p in pins]
  linspaces = size * size * [None]
  for i in range(size):
    for j in range(i, size):
      p0, p1 = pins[i], pins[j]
      dx, dy = (p1[0] - p0[0]), (p1[1] - p0[1])
      lines = []
      length = lengths[abs(i - j)]
      for k in range(length):
        factor = float(k) / length
        lines.append([int(p0[0] + factor * dx), int(p0[1] + factor * dy)])
      linspaces[i + j * size] = lines
      linspaces[j + i * size] = lines
  return linspaces, lengths


def channels(image: cvImage) -> int:
  return image.shape[2] if len(image.shape) >= 3 else 1


def pil_to_opencv(src: PImage) -> cvImage:
  dst = np.array(src)
  dst = dst[:, :, ::-1].copy()
  return dst


def crop(image: cvImage, box: Roi) -> cvImage:
  x1, y1, x2, y2 = box
  if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
    image, x1, x2, y1, y2 = pad_img_to_fit_bbox(image, x1, x2, y1, y2)
  return image[y1:y2, x1:x2]


def pad_img_to_fit_bbox(image: cvImage, x1: int, x2: int, y1: int,
                        y2: int) -> Tuple[int, int, int, int]:
  h, w = image.shape[:2]
  image = cv2.copyMakeBorder(image, -min(0, y1), max(y2 - h, 0), -min(0, x1),
                             max(x2 - w, 0), cv2.BORDER_CONSTANT)
  y2 += -min(0, y1)
  y1 += -min(0, y1)
  x2 += -min(0, x1)
  x1 += -min(0, x1)
  return image, x1, x2, y1, y2


def circular_crop(image: cvImage, radius: float,
                  center: Point = (0, 0)) -> cvImage:
  x1, y1 = center[0] - radius, center[1] - radius
  x2, y2 = center[0] + radius, center[1] + radius
  # crop 2rx2r box from the center.
  cropped = crop(image, [x1, y1, x2, y2])
  # create a mask with a circular ones(255).
  shape = (2*radius, 2*radius)
  fill = (255,)
  if channels(image) != 1:
    shape = (2*radius, 2*radius, 3)
    fill = (255, 255, 255,)
  mask = np.zeros(shape, dtype=np.uint8)
  cv2.circle(mask, (radius, radius), radius, fill, -1, None, 0)
  return cv2.bitwise_and(cropped, mask)


def copy_to_roi(src: cvImage, dst: cvImage, roi: Roi) -> cvImage:
  assert roi[0] <= roi[2] and roi[1] <= roi[3]
  dst_h, dst_w = dst.shape[:2]
  dst_x1, dst_y1 = max(0, roi[0]), max(0, roi[1])
  dst_x2, dst_y2 = min(dst_w, roi[2]), min(dst_h, roi[3])
  src_x1, src_y1 = dst_x1 - roi[0], dst_y1 - roi[1]
  src_x2, src_y2 = dst_x2 - roi[0], dst_y2 - roi[1]
  dst[dst_y1:dst_y2, dst_x1:dst_x2, :] = src[src_y1:src_y2, src_x1:src_x2, :]
  return dst


def color_dist_sq(c10: int, c11: int, c12: int, c20: int, c21: int,
                  c22: int) -> int:
  d0 = c10 - c20
  d1 = c11 - c21
  d2 = c12 - c22
  return int(d0 * d0 + d1 * d1 + d2 * d2)


def color_dist(c10: int, c11: int, c12: int, c20: int, c21: int,
               c22: int) -> int:
  d0 = c10 - c20
  d1 = c11 - c21
  d2 = c12 - c22
  return int(math.sqrt(d0 * d0 + d1 * d1 + d2 * d2))


def color_level(value: float) -> float:
  return (float)(255 * 1.0 / (1 + math.exp(9 - 0.05 * value)))


def extract_color_plane(image: cvImage, color: Color) -> cvImage:
  assert channels(image) >= 3, '3 or 4-channel input is expected!'
  w, h = image.shape[:2]
  output = 255 * np.ones((h, w), dtype=np.uint8)
  for y in range(h):
    for x in range(w):
      p0, p1, p2 = image.item(y, x, 0), image.item(y, x, 1), image.item(y, x, 2)
      dist = color_dist(p0, p1, p2, color[2], color[1], color[0])
      output[y, x] = int(color_level(dist))
  return output


def draw_line_aa(draw, x1, y1, x2, y2, col):
  """draw_line_aa.
     Draw an antialised line in the PIL ImageDraw.
     https://en.wikipedia.org/wiki/Xiaolin_Wu%27s_line_algorithm
     http://stackoverflow.com/questions/3122049/drawing-an-anti-aliased-line-with-thepython-imaging-library
  Args:
      draw:
      x1:
      y1:
      x2:
      y2:
      col:
  """

  def _pil_draw(draw, img, x, y, c, col, steep):
    if steep:
      x, y = y, x
    if x < img.size[0] and y < img.size[1] and x >= 0 and y >= 0:
      c = c * (float(col[3]) / 255.0)
      p = img.getpixel((x, y))
      draw.point((int(x), int(y)),
                 fill=(int((p[0] * (1 - c)) + col[0] * c),
                       int((p[1] * (1 - c)) + col[1] * c),
                       int((p[2] * (1 - c)) + col[2] * c), 255))

  def ipart(x):
    return math.floor(x)

  def fpart(x):
    return x - math.floor(x)

  def rfpart(x):
    return 1 - fpart(x)

  dx, dy = x2 - x1, y2 - y1
  if not dx:
    draw.line((x1, y1, x2, y2), fill=col, width=1)
    return
  steep = abs(dx) < abs(dy)
  if steep:
    x1, y1 = y1, x1
    x2, y2 = y2, x2
    dx, dy = dy, dx
  if x2 < x1:
    x1, x2 = x2, x1
    y1, y2 = y2, y1
  gradient = float(dy) / float(dx)

  # handle first endpoint
  xend = round(x1)
  yend = y1 + gradient * (xend - x1)
  xgap = rfpart(x1 + 0.5)
  xpxl1 = xend  # this will be used in the main loop
  ypxl1 = ipart(yend)
  img = draw.im
  _pil_draw(draw, img, xpxl1, ypxl1, rfpart(yend) * xgap, col, steep)
  _pil_draw(draw, img, xpxl1, ypxl1 + 1, fpart(yend) * xgap, col, steep)
  intery = yend + gradient  # first y-intersection for the main loop
  # handle second endpoint
  xend = round(x2)
  yend = y2 + gradient * (xend - x2)
  xgap = fpart(x2 + 0.5)
  xpxl2, ypxl2 = xend, ipart(yend)
  _pil_draw(draw, img, xpxl2, ypxl2, rfpart(yend) * xgap, col, steep)
  _pil_draw(draw, img, xpxl2, ypxl2 + 1, fpart(yend) * xgap, col, steep)

  # main loop
  for x in range(int(xpxl1 + 1), int(xpxl2)):
    _pil_draw(draw, img, x, ipart(intery), rfpart(intery), col, steep)
    _pil_draw(draw, img, x, ipart(intery) + 1, fpart(intery), col, steep)
    intery += gradient
