#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2024 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2024-08-13
################################################################

import open3d as o3d
import numpy as np


class CloudUtils:

    def __init__(self, intrinsics, radius=[0.1, 3.0]):
        # Depth map initialization
        fx, fy, cx, cy = intrinsics
        u, v = np.meshgrid(np.arange(640), np.arange(360))
        x = (u - cx) / fx
        y = (v - cy) / fy
        self.__depth_map = np.stack((x, y, np.ones_like(x)), axis=2)
        self.__radius_squre = [radius[0] * radius[0], radius[1] * radius[1]]

        # Variable initialization
        self.__cloud = o3d.geometry.PointCloud()

    def __person_cloud(self, points: np.ndarray) -> o3d.geometry.PointCloud:
        # down sample
        downsampled_cloud = o3d.geometry.PointCloud()
        downsampled_cloud.points = o3d.utility.Vector3dVector(points)
        downsampled_cloud = downsampled_cloud.voxel_down_sample(
            voxel_size=0.05)
        down_points = np.asarray(downsampled_cloud.points)

        # radius filter
        distances = (down_points[:, 1] * down_points[:, 1]) + (
            down_points[:, 2] * down_points[:, 2])
        valid_mask = (distances > self.__radius_squre[0]) & (
            distances < self.__radius_squre[1])
        radiused_points = down_points[valid_mask]

        # dbscan cluster
        cluster_cloud = o3d.geometry.PointCloud()
        cluster_cloud.points = o3d.utility.Vector3dVector(radiused_points)
        cluster_labels = np.asarray(
            cluster_cloud.cluster_dbscan(eps=0.1, min_points=15))
        nearest_cluster = self.__find_nearest_cluster(cluster_cloud,
                                                      cluster_labels)

        person_cloud = o3d.geometry.PointCloud()
        if nearest_cluster.size != 0:
            person_cloud.points = o3d.utility.Vector3dVector(nearest_cluster)
        return person_cloud

    def __find_nearest_cluster(self, cloud: o3d.geometry.PointCloud,
                               labels: np.ndarray) -> np.ndarray:
        unique_labels = np.unique(labels)

        min_distance = np.inf
        nearest_cluster = np.array([])
        for label in unique_labels:
            if label == -1:
                continue
            curr_cluster = np.asarray(cloud.points)[labels == label]
            centroid = np.mean(curr_cluster, axis=0)
            curr_distance = centroid[1] * centroid[1] + centroid[2] * centroid[
                2]
            if curr_distance < min_distance:
                min_distance = curr_distance
                nearest_cluster = np.copy(curr_cluster)

        return nearest_cluster

    def __call__(self, depth: np.ndarray,
                 result: list) -> o3d.geometry.PointCloud:
        depth_mask = depth > 0

        self.__cloud = o3d.geometry.PointCloud()
        for box in result:
            box_mask = np.zeros_like(depth, dtype=np.bool8)
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            box_mask[y1:y2, x1:x2] = True
            valid_mask = depth_mask & box_mask
            person_points = self.__depth_map[valid_mask] * depth[
                valid_mask][:, np.newaxis] * 1e-3
            person_cloud = self.__person_cloud(person_points)
            self.__cloud += person_cloud
        self.__cloud = self.__cloud.remove_duplicated_points()

        return self.__cloud
