# hex_rgbd_person_extract

**hex_rgbd_person_extract** is a ROS package designed to extract person point clouds from raw RGBD point cloud data.

## Maintainer

[Dong Zhaorui](mailto:847235539@qq.com)

## Public APIs

### Publish

| Topic           | Msg Type                  | Description                                  |
| --------------- | ------------------------- | -------------------------------------------- |
| `/person_cloud` | `sensor_msgs/PointCloud2` | Extracted point cloud representing a person. |

### Subscribe

| Topic          | Msg Type                  | Description                       |
| -------------- | ------------------------- | --------------------------------- |
| `/point_cloud` | `sensor_msgs/PointCloud2` | Input RGBD point cloud.           |
| `/image`       | `sensor_msgs/Image`       | Input RGB image from RGBD sensor. |

### Parameters

| Name          | Data Type | Description                                                                               |
| ------------- | --------- | ----------------------------------------------------------------------------------------- |
| `model_path`  | `string`  | Path to the TensorRT model.                                                               |
| `search_time` | `float`   | Maximum allowed time difference between the image and point cloud in a pair (in seconds). |

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

1. Create a workspace `catkin_ws` and navigate to the `src` directory:

   ```shell
   mkdir -p catkin_ws/src
   cd catkin_ws/src
   ```

2. Clone this repository:

   ```shell
   git clone https://gitlab.hexmove.cn/IBN_Blank/hex_rgbd_person_extract.git
   ```

3. Navigate back to the `catkin_ws` directory and build the workspace:

   ```shell
   cd ../
   catkin_make
   ```

4. Source the `setup.bash` and run the test:

   ```shell
   source devel/setup.bash --extend
   ```

### Platforms

* [ ] **Jetson Orin NX**
* [ ] **Jetson Orin Nano**
* [ ] **Jetson AGX Orin**
* [ ] **RK3588**

### Prerequisites

What additional things you need to use the software

* **ROS**

   Refer to [ROS Installation](http://wiki.ros.org/ROS/Installation)

* **Orbbec Camera**

   ```shell
   cd catkin_ws/src
   git clone https://github.com/orbbec/OrbbecSDK_ROS1.git
   catkin_make
   source ../devel/setup.bash --extend
   ```

* **Open3D**

   ```shell
   pip3 install --upgrade pip
   pip3 install open3d==0.18.0 --ignore-installed PyYAML
   ```

### Usage

1. Launch the kinematic node:

   ```shell
   roslaunch hex_rgbd_person_extract rgbd_person_extract.launch
   ```

2. Publish the image and point cloud to the `/image` and `/point_cloud` topics.
3. For more details, refer to the test launch file: `rgbd_person_extract_test.launch`.

## Running the Tests

To run the tests, use the following command:

```shell
roslaunch hex_rgbd_person_extract rgbd_person_extract_test.launch
```

## Reminder

1. Ensure that the image and point cloud topics are published at the same rate for optimal performance.
