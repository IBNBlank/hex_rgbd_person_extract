#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2024 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2024-08-10
################################################################

from utils import DataInterface, TrtUtils, CloudUtils, MergeUtils

DEBUG_FLAG = False


class RgbdPersonExtract:

    def __init__(self):
        # utils
        self.__data_interface = DataInterface("rgbd_person_extract", rate=100)
        self.__trt_utils = TrtUtils(self.__data_interface.get_model_path())
        self.__cloud_utils = None
        self.__merge_utils = None
        if DEBUG_FLAG:
            self.__merge_utils = MergeUtils()
        else:
            self.__cloud_utils = CloudUtils(
                self.__data_interface.get_depth_intrinsics())

    def __debug_loop(self):
        if self.__data_interface.is_pair_available():
            pair = self.__data_interface.get_pair()
            trt_result = self.__trt_utils(pair.get_image())
            merge = self.__merge_utils(pair.get_image(), pair.get_depth(),
                                       trt_result)
            self.__data_interface.pub_debug(merge)

    def __work_loop(self):
        if self.__data_interface.is_pair_available():
            pair = self.__data_interface.get_pair()
            trt_result = self.__trt_utils(pair.get_image())
            debug_image = self.__trt_utils.debug_image(pair.get_image())
            person_cloud = self.__cloud_utils(pair.get_depth(), trt_result)
            self.__data_interface.pub_cloud(person_cloud)
            self.__data_interface.pub_debug(debug_image)

    def work(self):
        while self.__data_interface.ok():
            if DEBUG_FLAG:
                self.__debug_loop()
            else:
                self.__work_loop()
            self.__data_interface.sleep()


def main():
    rgbd_person_extract = RgbdPersonExtract()
    rgbd_person_extract.work()


if __name__ == '__main__':
    main()
