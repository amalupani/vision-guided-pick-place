#!/usr/bin/env python3
import csv
import os
import random
import time
from threading import Thread

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from pymoveit2 import MoveIt2, MoveIt2Gripper
from pymoveit2.robots import panda

ARM_GROUP = "panda_arm"
HAND_GROUP = "hand"
BASE_LINK = "panda_link0"
EE_LINK = "panda_link8"
TCP_OFFSET = 0.1034                      # link8 -> fingertip centre (m)
DOWN = [1.0, 0.0, 0.0, 0.0]              # gripper pointing down (xyzw)
HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]   # Panda "ready" pose


class PickPlace(Node):
    def __init__(self):
        super().__init__("motion_node")
        self.declare_parameter("mode", "loop")
        cb = ReentrantCallbackGroup()
        self.arm = MoveIt2(
            node=self, joint_names=panda.joint_names(),
            base_link_name=BASE_LINK, end_effector_name=EE_LINK,
            group_name=ARM_GROUP, callback_group=cb)
        self.arm.max_velocity = 0.3
        self.arm.max_acceleration = 0.3
        self.gripper = MoveIt2Gripper(
            node=self, gripper_joint_names=panda.gripper_joint_names(),
            open_gripper_joint_positions=panda.OPEN_GRIPPER_JOINT_POSITIONS,
            closed_gripper_joint_positions=panda.CLOSED_GRIPPER_JOINT_POSITIONS,
            gripper_group_name=HAND_GROUP, callback_group=cb)

    def add_table(self):
        # top surface at z = 0 in panda_link0; spans x 0.2-0.9, y -0.5..0.5
        try:
            self.arm.add_collision_box(
                id="table", size=(0.7, 1.0, 0.04),
                position=(0.55, 0.0, -0.02),
                quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id=BASE_LINK)
        except TypeError as e:
            self.get_logger().error(
                f"add_collision_box args differ in your pymoveit2: {e}")

    def _ok(self):
        # reports the result of the most recent motion
        for name in ("motion_suceeded", "motion_succeeded"):
            if hasattr(self.arm, name):
                return bool(getattr(self.arm, name))
        return None

    def _go(self, xyz, cartesian=False, label=""):
        self.arm.move_to_pose(position=xyz, quat_xyzw=DOWN, cartesian=cartesian)
        self.arm.wait_until_executed()
        ok = self._ok()
        self.get_logger().info(f"{label}: success={ok}")
        return ok is not False

    def _home(self):
        self.arm.move_to_configuration(HOME)
        self.arm.wait_until_executed()

    def _grip(self, close):
        (self.gripper.close if close else self.gripper.open)()
        self.gripper.wait_until_executed()
        time.sleep(0.5)

    def pick_and_place(self, obj_xyz, place_xyz, hover=0.12):
        x, y, z = obj_xyz
        gz = z + TCP_OFFSET
        px, py, pz = place_xyz
        steps = [
            ("pre-grasp", lambda: self._go([x, y, gz + hover], label="pre-grasp")),
            ("descend", lambda: self._go([x, y, gz], True, "descend")),
            ("lift", lambda: self._go([x, y, gz + hover], True, "lift")),
            ("over-place", lambda: self._go([px, py, pz + TCP_OFFSET + hover], label="over-place")),
            ("lower", lambda: self._go([px, py, pz + TCP_OFFSET], True, "lower")),
            ("retreat", lambda: self._go([px, py, pz + TCP_OFFSET + hover], True, "retreat")),
        ]
        self._home()
        self._grip(False)
        for name, fn in steps:
            if not fn():
                self.get_logger().error(f"FAILED at {name}")
                return False, name
            if name == "descend":
                self._grip(True)
            if name == "lower":
                self._grip(False)
        return True, ""

    def reachability(self, n=30, z=0.02, hover=0.12):
        gz = z + TCP_OFFSET
        rows = []
        for i in range(n):
            x = random.uniform(0.35, 0.55)
            y = random.uniform(-0.20, 0.20)
            self._home()
            ok_pre = self._go([x, y, gz + hover], label=f"reach {i} pre")
            ok_down = ok_pre and self._go([x, y, gz], True, f"reach {i} down")
            rows.append((i, round(x, 3), round(y, 3), int(ok_pre), int(ok_down)))
        path = os.path.expanduser("~/ws/reach.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["i", "x", "y", "pre_ok", "descend_ok"])
            w.writerows(rows)
        self.get_logger().info(f"reachability results written to {path}")


def main():
    rclpy.init()
    node = PickPlace()
    ex = MultiThreadedExecutor(2)
    ex.add_node(node)
    Thread(target=ex.spin, daemon=True).start()
    time.sleep(2.0)
    node.add_table()
    time.sleep(1.0)

    if node.get_parameter("mode").value == "reach":
        node.reachability(30)
    else:
        passed = 0
        for i in range(10):
            node.get_logger().info(f"=== Run {i + 1}/10 ===")
            ok, _ = node.pick_and_place([0.45, 0.10, 0.02], [0.40, -0.25, 0.02])
            passed += ok
        node.get_logger().info(f"RESULT: {passed}/10 runs completed")
    rclpy.shutdown()


if __name__ == "__main__":
    main()