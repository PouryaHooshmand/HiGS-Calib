import torch
import numpy as np


def project_points_to_depth(camera_pose, intrinsics, points_3d, image_shape):
    """
    Project 3D point cloud to camera coordinate frame to generate depth map.

    Args:
    - camera_pose : 4x4 camera extrinsic matrix [[R, t], [0, 1]], transforms points from world coordinates to camera coordinates
    - intrinsics  : 3x3 camera intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    - points_3d   : Nx3 3D point cloud in world coordinate frame
    - image_shape : Output depth map dimensions (height, width)

    Returns:
    - depth_map : Depth map, unprojected regions filled with 0 (or 1e-7)
    - mask      : Boolean mask marking valid projection regions
    """
    H, W = image_shape

    # 1. Coordinate transformation to camera frame
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    points_hom = np.hstack((points_3d, np.ones((len(points_3d), 1))))  # Homogeneous coordinates
    points_cam = (camera_pose @ points_hom.T).T[:, :3]  # Apply extrinsic transformation

    # 2. Filter points behind the camera (Z <= 0)
    valid = points_cam[:, 2] > 0
    points_cam = points_cam[valid]
    if points_cam.size == 0:
        return np.zeros((H, W)), np.zeros((H, W), dtype=bool)

    # 3. Project to image plane
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Normalized coordinates (avoid division by zero)
    z = points_cam[:, 2]
    x = points_cam[:, 0] / z
    y = points_cam[:, 1] / z

    # Pixel coordinates (round to nearest integer)
    u = np.round(fx * x + cx).astype(int)
    v = np.round(fy * y + cy).astype(int)

    # 4. Filter out-of-bounds points
    valid_pixels = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[valid_pixels], v[valid_pixels], z[valid_pixels]
    if z.size == 0:
        return np.zeros((H, W)), np.zeros((H, W), dtype=bool)

    # 5. Generate depth map and mask
    depth_map = np.full((H, W), np.inf)  # Initialize with infinity
    mask = np.zeros((H, W), dtype=np.float32)

    # Update with minimum depth (handle multiple points projecting to the same pixel)
    np.minimum.at(depth_map, (v, u), z)

    # Generate valid mask
    mask[v, u] = 1.
    depth_map[mask != 1.] = 1e-7  # Set invalid regions to epsilon

    return depth_map, mask