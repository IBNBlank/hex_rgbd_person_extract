#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2024 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2024-08-13
################################################################

import cv2
import numpy as np


class MergeUtils:

    def __init__(self):
        self.__merge = np.array([])

    def __depth_to_color(self, depth: np.ndarray) -> np.ndarray:
        depth_mask = depth > 0
        depth_color = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_color = cv2.applyColorMap(np.uint8(depth_color),
                                        cv2.COLORMAP_JET)
        return depth_color, depth_mask

    def __call__(self, image: np.ndarray, depth: np.ndarray,
                 result: list) -> np.ndarray:
        depth_color, depth_mask = self.__depth_to_color(depth)
        box_mask = np.zeros_like(depth, dtype=np.bool8)
        for box in result:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            box_mask[y1:y2, x1:x2] = True
        valid_mask = depth_mask & box_mask
        self.__merge = image
        self.__merge[valid_mask] = depth_color[valid_mask] * 0.7 + image[
            valid_mask] * 0.3
        return self.__merge
