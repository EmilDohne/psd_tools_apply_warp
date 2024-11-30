from typing import Optional

import numba
import numpy as np
import cv2

import smart_object_warp.point_2d as point_2d

class ImageBuffer:
    def __init__(self, width: int, height: int, buffer: Optional[np.ndarray] = None):
        self.width = width
        self.height = height
        if buffer is not None:
            self.buffer = buffer
        else:
            self.buffer = np.zeros((height, width, 3), dtype=np.uint8)

    @staticmethod
    def load(image_path: str):
        img = cv2.imread(image_path)
        return ImageBuffer(img.shape[1], img.shape[0], img)

    def save(self, output_path: str):
        cv2.imwrite(output_path, self.buffer)

    @staticmethod
    @numba.njit()
    def sample_bilinear_uv(buffer:np.ndarray, width: int, height: int, uv: point_2d.Point2d) -> np.ndarray:
        # Bilinear interpolation based on UV coordinates. UV values should be in the range [0, 1].
        x, y = uv.x * (width - 1), uv.y * (height - 1)
        x1, y1 = int(x), int(y)
        x2, y2 = min(x1 + 1, width - 1), min(y1 + 1, height - 1)

        # Perform bilinear interpolation between four surrounding pixels
        f11 = buffer[y1, x1]
        f12 = buffer[y2, x1]
        f21 = buffer[y1, x2]
        f22 = buffer[y2, x2]

        dx = x - x1
        dy = y - y1
        return (f11 * (1 - dx) * (1 - dy) +
                f21 * dx * (1 - dy) +
                f12 * (1 - dx) * dy +
                f22 * dx * dy)