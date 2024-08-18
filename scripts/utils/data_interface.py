#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2024 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2024-08-10
################################################################

import rospy
import open3d as o3d
import numpy as np
import sensor_msgs.point_cloud2 as pc2

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField


class DepthStamped:

    def __init__(self, depth: Image):
        self.__stamp = depth.header.stamp
        self.__depth = np.frombuffer(depth.data, dtype=np.uint16).reshape(
            depth.height, depth.width)

    def get_stamp(self) -> rospy.Time:
        return self.__stamp

    def get_depth(self) -> np.ndarray:
        return self.__depth


class ImageStamped:

    def __init__(self, image: Image):
        self.__stamp = image.header.stamp
        self.__image = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height, image.width, -1)

    def get_stamp(self) -> rospy.Time:
        return self.__stamp

    def get_image(self) -> np.ndarray:
        return self.__image


class RgbdPair:

    def __init__(self, image: np.ndarray, depth: np.ndarray):
        self.__image = np.copy(image)
        self.__depth = np.copy(depth)

    def get_image(self) -> np.ndarray:
        return self.__image

    def get_depth(self) -> np.ndarray:
        return self.__depth


class DataInterface:

    def __init__(self, name: str = "unknown", rate: int = 30):
        ### ros node
        rospy.init_node(name, anonymous=True)
        self.__rate = rospy.Rate(rate)

        ### parameter
        self.__model_path = rospy.get_param("~model_path", "model")
        self.__search_time = rospy.get_param("~search_time", 0.02)
        self.__cloud_frame = rospy.get_param("~cloud_frame", "camera_link")
        self.__depth_intrinsics = rospy.get_param("~depth_intrinsics", [
            346.086669921875, 346.086669921875, 323.208984375,
            183.76251220703125
        ])

        ### variable
        self.__image_curr = None
        self.__depth_buffer = []

        ### publisher
        self.__cloud_pub = rospy.Publisher('/person_cloud',
                                           PointCloud2,
                                           queue_size=10)
        self.__debug_pub = rospy.Publisher('/merge_debug',
                                           Image,
                                           queue_size=10)

        ### subscriber
        self.__image_sub = rospy.Subscriber('/image', Image,
                                            self.__image_callback)
        self.__depth_sub = rospy.Subscriber('/depth', Image,
                                            self.__depth_callback)

    def ok(self):
        return not rospy.is_shutdown()

    def sleep(self):
        self.__rate.sleep()

    def get_model_path(self):
        return self.__model_path

    def get_depth_intrinsics(self):
        return self.__depth_intrinsics

    def is_pair_available(self):
        if self.__image_curr is None:
            return False

        stamp = self.__image_curr.get_stamp()
        while len(self.__depth_buffer) > 0:
            delta_time = (self.__depth_buffer[0].get_stamp() - stamp).to_sec()
            if np.fabs(delta_time) < self.__search_time:
                return True
            self.__depth_buffer.pop(0)

        return False

    def get_pair(self):
        depth = self.__depth_buffer.pop(0)
        return RgbdPair(self.__image_curr.get_image(), depth.get_depth())

    def pub_cloud(self, cloud: np.ndarray):
        # o3d.geometry.PointCloud -> np.ndarray
        points = np.asarray(cloud.points)

        # create header
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.__cloud_frame

        # create point cloud2 message
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        point_cloud2_msg = pc2.create_cloud(header, fields, points)

        # publish person cloud
        self.__cloud_pub.publish(point_cloud2_msg)

    def pub_debug(self, image: np.ndarray):
        # np.ndarray -> Image
        image_msg = Image()
        image_msg.header.stamp = rospy.Time.now()
        image_msg.header.frame_id = "camera_link"
        image_msg.height = image.shape[0]
        image_msg.width = image.shape[1]
        image_msg.encoding = "rgb8"
        image_msg.step = image.shape[1] * image.shape[2]
        image_msg.data = image.tobytes()

        # publish debug image
        self.__debug_pub.publish(image_msg)

    def __image_callback(self, msg: Image):
        self.__image_curr = ImageStamped(msg)

    def __depth_callback(self, msg: Image):
        self.__depth_buffer.append(DepthStamped(msg))
