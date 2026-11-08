import numpy as np
import pybullet as p
import cv2

W, H, FOV, NEAR, FAR = 640, 480, 60.0, 0.05, 2.0
EYE, TARGET, UP = [0.5, 0.0, 0.8], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]
CUBE_POS = [0.5, 0.10, 0.02]

p.connect(p.DIRECT)
p.setGravity(0, 0, -9.81)

half = [0.35, 0.5, 0.02]
p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=half),
    baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                             rgbaColor=[0.6, 0.6, 0.6, 1]),
    basePosition=[0.55, 0.0, -0.02])

cube = p.createMultiBody(
    baseMass=0.05,
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02] * 3),
    baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02] * 3,
                                             rgbaColor=[1, 0.1, 0.1, 1]),
    basePosition=CUBE_POS)

for _ in range(100):
    p.stepSimulation()                     # let the cube settle

view = p.computeViewMatrix(EYE, TARGET, UP)
proj = p.computeProjectionMatrixFOV(FOV, W / H, NEAR, FAR)
_, _, rgba, zbuf, seg = p.getCameraImage(W, H, view, proj,
                                         renderer=p.ER_TINY_RENDERER)

rgb = np.reshape(rgba, (H, W, 4))[:, :, :3].astype(np.uint8)
zbuf = np.reshape(zbuf, (H, W))
seg = np.reshape(seg, (H, W))
depth = FAR * NEAR / (FAR - (FAR - NEAR) * zbuf)      # metres, along camera z

cv2.imwrite("rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite("depth.png", np.clip(depth / FAR * 255, 0, 255).astype(np.uint8))

# intrinsics from the vertical FOV
fy = (H / 2) / np.tan(np.deg2rad(FOV) / 2)
fx, cx, cy = fy, W / 2, H / 2

# deproject the cube centroid to a 3D point in the camera optical frame
vs, us = np.where(seg == cube)
u, v = us.mean(), vs.mean()
Z = np.median(depth[seg == cube])
X, Y = (u - cx) * Z / fx, (v - cy) * Z / fy

# optical frame (x right, y down, z forward) -> world, for this camera pose
R = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=float)
world = R @ np.array([X, Y, Z]) + np.array(EYE)

print("pixel centroid:", round(u, 1), round(v, 1))
print("depth at cube:", round(Z, 3), "(expected about 0.76)")
print("estimated world:", world.round(3))
print("ground truth top-centre:", [CUBE_POS[0], CUBE_POS[1], CUBE_POS[2] + 0.02])