"""image_utils.py implements image utility methods."""

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


def channels(image: cvImage) -> int:
  return image.shape[2] if len(image.shape) >= 3 else 1


def pil_to_opencv(src: PImage) -> cvImage:
  dst = np.array(src)
  dst = dst[:, :, ::-1].copy()
  return dst


def line_length(p1: Point, p2: Point) -> int:
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]
  return int(math.sqrt(dx * dx + dy * dy))


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
  mask = np.zeros((2 * radius, 2 * radius), dtype=np.uint8)
  cv2.circle(mask, (radius, radius), radius, (255,), -1, None, 0)
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


def color_dist(c1: Color, c2: Color) -> int:
  d1 = c1[0] - c2[0]
  d2 = c1[1] - c2[1]
  d3 = c1[2] - c2[2]
  return math.sqrt(d1 * d1 + d2 * d2 + d3 * d3)


def color_level(value: float) -> float:
  return (float)(255 * 1.0 / (1 + math.exp(9 - 0.05 * value)))


def extract_color_plane(image: cvImage, color: Color) -> cvImage:
  assert channels(image) >= 3, '3 or 4-channel input is expected!'
  w, h = image.shape[:2]
  color[0], color[2] = color[2], color[0]  # opencv works with BGR format.
  output = 255 * np.ones((h, w), dtype=np.uint8)
  for y in range(h):
    for x in range(w):
      dist = color_dist(image[y, x, :], color)
      output[y, x] = int(color_level(dist))
  return output


def draw_line_aa(draw, x1, y1, x2, y2, col, dash_interval=None):
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
      dash_interval:
  """

  def _pil_draw(draw, img, x, y, c, col, steep, dash_interval):
    if steep:
      x, y = y, x
    if x < img.size[0] and y < img.size[1] and x >= 0 and y >= 0:
      c = c * (float(col[3]) / 255.0)
      p = img.getpixel((x, y))
      x = int(x)
      y = int(y)
      if dash_interval:
        d = dash_interval - 1
        if (x / dash_interval) % d == 0 and (y / dash_interval) % d == 0:
          return
      draw.point((x, y),
                 fill=(int((p[0] * (1 - c)) + col[0] * c),
                       int((p[1] * (1 - c)) + col[1] * c),
                       int((p[2] * (1 - c)) + col[2] * c), 255))

  def ipart(x):
    return math.floor(x)

  def fpart(x):
    return x - math.floor(x)

  def rfpart(x):
    return 1 - fpart(x)

  dx = x2 - x1
  if not dx:
    draw.line((x1, y1, x2, y2), fill=col, width=1)
    return

  dy = y2 - y1
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
  _pil_draw(draw, img, xpxl1, ypxl1,
            rfpart(yend) * xgap, col, steep, dash_interval)
  _pil_draw(draw, img, xpxl1, ypxl1 + 1,
            fpart(yend) * xgap, col, steep, dash_interval)
  intery = yend + gradient  # first y-intersection for the main loop

  # handle second endpoint
  xend = round(x2)
  yend = y2 + gradient * (xend - x2)
  xgap = fpart(x2 + 0.5)
  xpxl2 = xend  # this will be used in the main loop
  ypxl2 = ipart(yend)
  _pil_draw(draw, img, xpxl2, ypxl2,
            rfpart(yend) * xgap, col, steep, dash_interval)
  _pil_draw(draw, img, xpxl2, ypxl2 + 1,
            fpart(yend) * xgap, col, steep, dash_interval)

  # main loop
  for x in range(int(xpxl1 + 1), int(xpxl2)):
    _pil_draw(draw, img, x, ipart(intery), rfpart(intery), col, steep,
              dash_interval)
    _pil_draw(draw, img, x,
              ipart(intery) + 1, fpart(intery), col, steep, dash_interval)
    intery = intery + gradient
