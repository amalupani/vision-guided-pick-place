#!/usr/bin/env python3
import math
import random

import numpy as np
import pybullet as p
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster

W, H, FOV, NEAR, FAR = 640, 480, 60.0, 0.05, 2.0
EYE, TARGET, UP = [0.5, 0.0, 0.8], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]
CAM_QUAT = (0.7071068, -0.7071068, 0.0, 0.0)   # optical frame in panda_link0 (xyzw)
BASE = "panda_link0"
CAM_FRAME = "camera_optical_frame"
SPAWN_X = (0.37, 0.53)      # provisional zone with a 2 cm margin
SPAWN_Y = (-0.18, 0.18)
CUBE_HALF = 0.02


class SimNode(Node):
    def __init__(self):
        super().__init__("sim_node")
        self.declare_parameter("seed", -1)
        seed = self.get_parameter("seed").value
        if seed >= 0:
            random.seed(seed)

        # --- PyBullet scene (headless) ---
        p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        half = [0.35, 0.5, 0.02]
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=half),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=half, rgbaColor=[0.6, 0.6, 0.6, 1]),
            basePosition=[0.55, 0.0, -0.02])
        self.cube = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[CUBE_HALF] * 3),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[CUBE_HALF] * 3, rgbaColor=[1, 0.1, 0.1, 1]),
            basePosition=[0.5, 0.0, CUBE_HALF])
        self.view = p.computeViewMatrix(EYE, TARGET, UP)
        self.proj = p.computeProjectionMatrixFOV(FOV, W / H, NEAR, FAR)

        # --- intrinsics from the vertical FOV ---
        fy = (H / 2) / math.tan(math.radians(FOV) / 2)
        fx, cx, cy = fy, W / 2, H / 2
        self.info = CameraInfo()
        self.info.header.frame_id = CAM_FRAME
        self.info.width, self.info.height = W, H
        self.info.distortion_model = "plumb_bob"
        self.info.d = [0.0] * 5
        self.info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        self.info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

        # --- ROS interfaces ---
        self.rgb_pub = self.create_publisher(Image, "/camera/color/image_raw", 10)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", 10)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/color/camera_info", 10)
        self.gt_pub = self.create_publisher(PoseStamped, "/ground_truth_pose", 10)
        self.create_service(Trigger, "/reset_scene", self.reset_cb)

        self.tf_static = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = BASE
        t.child_frame_id = CAM_FRAME
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = EYE
        (t.transform.rotation.x, t.transform.rotation.y,
         t.transform.rotation.z, t.transform.rotation.w) = CAM_QUAT
        self.tf_static.sendTransform(t)

        self.gt = None
        self.spawn()
        self.create_timer(0.1, self.tick)          # 10 Hz
        self.get_logger().info("sim_node ready: publishing camera at 10 Hz")

    def spawn(self):
        x = random.uniform(*SPAWN_X)
        y = random.uniform(*SPAWN_Y)
        yaw = random.uniform(0.0, math.pi / 2)
        p.resetBasePositionAndOrientation(
            self.cube, [x, y, CUBE_HALF], p.getQuaternionFromEuler([0, 0, yaw]))
        p.resetBaseVelocity(self.cube, [0, 0, 0], [0, 0, 0])
        for _ in range(20):
            p.stepSimulation()
        self.gt = p.getBasePositionAndOrientation(self.cube)

    def reset_cb(self, request, response):
        self.spawn()
        pos = self.gt[0]
        response.success = True
        response.message = f"cube at x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f}"
        self.get_logger().info(response.message)
        return response

    def _image(self, stamp, encoding, step, data):
        m = Image()
        m.header.stamp = stamp
        m.header.frame_id = CAM_FRAME
        m.height, m.width = H, W
        m.encoding, m.is_bigendian, m.step = encoding, 0, step
        m.data = data
        return m

    def tick(self):
        stamp = self.get_clock().now().to_msg()
        _, _, rgba, zbuf, _ = p.getCameraImage(
            W, H, self.view, self.proj, renderer=p.ER_TINY_RENDERER)
        rgb = np.reshape(rgba, (H, W, 4))[:, :, :3].astype(np.uint8)
        zb = np.reshape(zbuf, (H, W))
        depth = (FAR * NEAR / (FAR - (FAR - NEAR) * zb)).astype(np.float32)

        self.rgb_pub.publish(self._image(stamp, "rgb8", W * 3, rgb.tobytes()))
        self.depth_pub.publish(self._image(stamp, "32FC1", W * 4, depth.tobytes()))
        self.info.header.stamp = stamp
        self.info_pub.publish(self.info)

        pos, orn = self.gt
        g = PoseStamped()
        g.header.stamp = stamp
        g.header.frame_id = BASE
        g.pose.position.x, g.pose.position.y, g.pose.position.z = pos
        (g.pose.orientation.x, g.pose.orientation.y,
         g.pose.orientation.z, g.pose.orientation.w) = orn
        self.gt_pub.publish(g)


def main():
    rclpy.init()
    node = SimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()