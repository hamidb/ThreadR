"""circular_frame.py implements Frame Object."""

from __future__ import absolute_import, division, print_function

from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

import typings as tp
import utils
from layers.shared_layer import SharedLayer
from layers.simple_layer import SimpleLayer


class CircularFrame:

  def __init__(
      self,
      radius: int = 250,
      max_threads: int = 3000,
      max_thread_length: Optional[float] = None,
      background_color: tp.Color = (255, 255, 255),  # background for display.
      image: Optional[tp.cvImage] = None,  # shared image across layers.
      pins: List[tp.Point] = [],  # shared pins for layers if not empty.
      ignore_last_n_visited_pins: int = 20,  # skip last n visited pins.
      display_scale: float = 3.0,
  ) -> None:
    self.radius = radius
    self.max_threads = max_threads
    self.max_thread_length = max_thread_length or (max_threads * 2 * radius)
    size = int(display_scale * 2 * radius)
    self.frame_image = Image.new('RGB', (size, size), background_color)

    # shared attributes across layers (if provided).
    self.layers = []
    self.image = image
    self.working_image = self.build_working_image()
    self.pins = pins
    self.line_spaces, self.lengths = utils.compute_lines(pins)
    self.visited_pins = len(pins) * len(pins) * [0]
    self.display_scale = display_scale
    self.drawable = ImageDraw.Draw(self.frame_image, 'RGBA')

  def add_new_layer(self, layer: tp.Layer) -> None:
    layer.drawable = self.drawable
    layer.display_scale = self.display_scale

    if isinstance(layer, SharedLayer):
      #  layer.last_pins = self.last_pins
      layer.pins = self.pins
      layer.lengths = self.lengths
      layer.line_spaces = self.line_spaces
      layer.visited_pins = self.visited_pins
      layer.image = self.image
      layer.working_image = self.working_image

    layer.setup_layer()
    self.layers.append(layer)

  def build_working_image(self) -> Optional[tp.cvImage]:
    if self.image is None:
      return None
    return 255 * np.ones(self.image.shape, dtype=np.uint8)

  def run(self) -> None:
    # Draw pins.
    for layer in self.layers:
      for px, py in layer.pins:
        px = int((px + layer.origin[0]) * self.display_scale)
        py = int((py + layer.origin[1]) * self.display_scale)
        r = int(1 * self.display_scale)
        self.drawable.ellipse((px - r, py - r, px + r, py + r),
                              fill=(0, 128, 0, 255))

    processed_layers = set()
    total_threads = sum([l.max_threads for l in self.layers])
    while len(processed_layers) < len(self.layers):
      for layer in self.layers:
        if layer in processed_layers:
          continue
        if layer.thread_count >= layer.max_threads:
          processed_layers.add(layer)
          continue
        # Draw each layers' threads evenly.
        layer.run(max(1, int(20 * layer.max_threads / total_threads)))
        display = utils.pil_to_opencv(self.frame_image)
        display = cv2.resize(display, (1000, 1000))
        cv2.imshow('Frame Image', display)
        cv2.waitKey(1)

  def clear(self) -> None:
    self.layers.clear()
