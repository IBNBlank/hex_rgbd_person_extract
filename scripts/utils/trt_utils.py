#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2024 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2024-08-10
################################################################

import cv2
import time
import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

WIDTH_PATCH = 0
HEIGHT_PATCH = 12


class TrtUtils:

    def __init__(self, model_path: str):
        # TensorRT initialization
        self.__runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        with open(model_path, "rb") as f:
            self.__engine = self.__runtime.deserialize_cuda_engine(f.read())
        self.__context = self.__engine.create_execution_context()

        # Memory allocation
        # input_shape = (1, 3, 384, 640)
        # output_shape = (1, 84, 5040)
        self.__cpu_input: np.ndarray = np.random.rand(1, 3, 384,
                                                      640).astype(np.float32)
        self.__cpu_output: np.ndarray = np.random.rand(1, 84,
                                                       5040).astype(np.float32)
        self.__gpu_input = cuda.mem_alloc(self.__cpu_input.nbytes)
        self.__gpu_output = cuda.mem_alloc(self.__cpu_output.nbytes)

        # Class names
        self.__names = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            9: 'traffic light',
        }

        # Variables
        self.__result = []

    def __infer(self):
        cuda.memcpy_htod(self.__gpu_input, self.__cpu_input)
        self.__context.execute_v2(
            bindings=[int(self.__gpu_input),
                      int(self.__gpu_output)])
        cuda.memcpy_dtoh(self.__cpu_output, self.__gpu_output)

    def __preprocess(self):
        self.__cpu_input = cv2.copyMakeBorder(self.__cpu_input,
                                              HEIGHT_PATCH,
                                              HEIGHT_PATCH,
                                              WIDTH_PATCH,
                                              WIDTH_PATCH,
                                              cv2.BORDER_CONSTANT,
                                              value=(114, 114, 114))
        self.__cpu_input = self.__cpu_input[..., ::-1]
        self.__cpu_input = self.__cpu_input.transpose((2, 0, 1))
        self.__cpu_input = np.ascontiguousarray(self.__cpu_input)
        self.__cpu_input = self.__cpu_input.astype(np.float32)
        self.__cpu_input /= 255.0

    def __postprocess(self):
        boxes = self.__get_boxes(conf_thres=0.7, max_det=300)
        self.__non_max_suppression(boxes, iou_thres=0.7)

    def __get_boxes(
            self,
            conf_thres=0.25,
            max_det=300,
            classes=np.array([0]),
    ) -> np.ndarray:
        assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        # filter out low confidence boxes
        pred = self.__cpu_output[0]
        pred = pred.transpose(1, 0)
        valid = np.amax(pred[:, 4:], axis=1) > conf_thres
        pred = pred[valid, :]

        # convert xywh to xyxy
        xyxy = self.__xywh2xyxy(pred[..., :4])

        # box: [x1, y1, x2, y2, conf, class]
        conf = np.max(pred[:, 4:], axis=1, keepdims=True)
        cls = np.argmax(pred[:, 4:], axis=1, keepdims=True)
        boxes = np.concatenate((xyxy, conf, cls), 1)
        boxes = boxes[(boxes[:, 5] == classes)]
        if not boxes.shape[0]:
            return boxes

        # Sort by confidence
        sorted_indices = boxes[:, 4].argsort()[::-1]
        boxes = boxes[sorted_indices]
        if boxes.shape[0] > max_det:
            boxes = boxes[:max_det]

        return boxes

    def __xywh2xyxy(self, xywh: np.ndarray) -> np.ndarray:
        assert xywh.shape[
            -1] == 4, f"input shape last dimension expected 4 but input shape is {xywh.shape}"
        xyxy = np.empty_like(xywh)
        xy = xywh[..., :2]
        half_wh = xywh[..., 2:] / 2
        xyxy[..., :2] = xy - half_wh
        xyxy[..., 2:] = xy + half_wh
        return xyxy

    def __non_max_suppression(self, boxes: np.ndarray, iou_thres=0.45):
        assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = np.arange(boxes.shape[0])

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Compute IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            idxes = np.where(iou <= iou_thres)[0]
            order = order[idxes + 1]

        best_boxes = boxes[keep]
        for box in best_boxes:
            self.__result.append(self.__scale_box(box))

    def __scale_box(self, x1y1x2y2_scaled: np.ndarray) -> np.ndarray:
        x1x2y1y2_raw = np.zeros(4)
        x1x2y1y2_raw[0] = x1y1x2y2_scaled[0] - WIDTH_PATCH
        x1x2y1y2_raw[1] = x1y1x2y2_scaled[1] - HEIGHT_PATCH
        x1x2y1y2_raw[2] = x1y1x2y2_scaled[2] - WIDTH_PATCH
        x1x2y1y2_raw[3] = x1y1x2y2_scaled[3] - HEIGHT_PATCH
        return x1x2y1y2_raw

    def __call__(self, image_raw: np.ndarray) -> list:
        self.__cpu_input = np.copy(image_raw)
        self.__result = []
        self.__preprocess()
        self.__infer()
        self.__postprocess()
        return self.__result

    def debug_image(self, image_raw: np.ndarray):
        for box in self.__result:
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(image_raw,
                          p1,
                          p2, (250, 50, 50),
                          thickness=2,
                          lineType=cv2.LINE_AA)
        return image_raw


def main():
    trt_utils = TrtUtils("../../models/yolov8n.trt")

    start = time.perf_counter()
    loop_num = 2000
    for i in range(loop_num):
        image_raw = cv2.imread(f"../../models/2.png")
        result = trt_utils(image_raw)
        if (i + 1) % 250 == 0:
            print(f"Processing {i + 1}")
    end = time.perf_counter()
    print(f"Time: {(end - start) * 1000 / loop_num} ms")

    debug = trt_utils.debug_image(image_raw)
    cv2.imshow("result", debug)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('result', cv2.WND_PROP_VISIBLE) < 1:
            break
    print("Close window")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
