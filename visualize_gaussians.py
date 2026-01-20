#!/usr/bin/env python3
"""
Visualize 3D Gaussians from final_model.pth using Open3D
Shows the Gaussian positions, scales, and camera views
"""

import torch
import numpy as np
import open3d as o3d


def load_gaussian_model(model_path):
    """Load the Gaussian model from a .pth file"""
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    return model


def create_gaussian_ellipsoids(xyz, scaling, rotation, intensity, max_gaussians=5000):
    """Create ellipsoid meshes representing the Gaussians"""
    geometries = []
    
    # Subsample if too many Gaussians
    n_gaussians = xyz.shape[0]
    if n_gaussians > max_gaussians:
        indices = np.random.choice(n_gaussians, max_gaussians, replace=False)
        xyz = xyz[indices]
        scaling = scaling[indices]
        rotation = rotation[indices]
        intensity = intensity[indices]
    
    # Normalize intensity for coloring
    intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
    
    for i in range(xyz.shape[0]):
        # Create a unit sphere
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=8)
        
        # Scale by the Gaussian scaling factors
        scale = scaling[i].numpy()
        scale = np.clip(scale, 0.001, 10.0)  # Clamp scales
        
        # Create scaling matrix
        sphere.scale(np.mean(scale) * 2, center=sphere.get_center())
        
        # Translate to position
        pos = xyz[i].numpy()
        sphere.translate(pos)
        
        # Color based on intensity (green to yellow colormap)
        int_val = intensity_norm[i].item()
        color = [int_val, 1.0 - 0.5 * int_val, 0.2]  # Green-yellow gradient
        sphere.paint_uniform_color(color)
        
        geometries.append(sphere)
    
    return geometries


def create_gaussian_point_cloud(xyz, intensity, scaling=None):
    """Create a point cloud from Gaussian centers with colors based on intensity"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.numpy())
    
    # Normalize intensity for coloring
    intensity_np = intensity.numpy().flatten()
    intensity_norm = (intensity_np - intensity_np.min()) / (intensity_np.max() - intensity_np.min() + 1e-8)
    
    # Create colormap (viridis-like: purple to yellow)
    colors = np.zeros((len(intensity_norm), 3))
    colors[:, 0] = intensity_norm  # Red channel
    colors[:, 1] = intensity_norm * 0.8  # Green channel
    colors[:, 2] = 1.0 - intensity_norm  # Blue channel
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


def create_camera_frustum(position, look_at, up, fov=60, aspect=1.0, scale=10.0):
    """Create a camera frustum visualization"""
    # Calculate camera coordinate system
    forward = np.array(look_at) - np.array(position)
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    up_vec = np.cross(right, forward)
    
    # Frustum parameters
    near = scale * 0.1
    far = scale
    
    half_height_near = near * np.tan(np.radians(fov / 2))
    half_width_near = half_height_near * aspect
    
    half_height_far = far * np.tan(np.radians(fov / 2))
    half_width_far = half_height_far * aspect
    
    # Frustum corners
    position = np.array(position)
    
    # Near plane corners
    near_center = position + forward * near
    near_tl = near_center + up_vec * half_height_near - right * half_width_near
    near_tr = near_center + up_vec * half_height_near + right * half_width_near
    near_bl = near_center - up_vec * half_height_near - right * half_width_near
    near_br = near_center - up_vec * half_height_near + right * half_width_near
    
    # Far plane corners
    far_center = position + forward * far
    far_tl = far_center + up_vec * half_height_far - right * half_width_far
    far_tr = far_center + up_vec * half_height_far + right * half_width_far
    far_bl = far_center - up_vec * half_height_far - right * half_width_far
    far_br = far_center - up_vec * half_height_far + right * half_width_far
    
    # Create line set for frustum
    points = [
        position,  # 0: camera position
        near_tl, near_tr, near_br, near_bl,  # 1-4: near plane
        far_tl, far_tr, far_br, far_bl  # 5-8: far plane
    ]
    
    lines = [
        # From camera to near corners
        [0, 1], [0, 2], [0, 3], [0, 4],
        # Near plane
        [1, 2], [2, 3], [3, 4], [4, 1],
        # Far plane
        [5, 6], [6, 7], [7, 8], [8, 5],
        # Connect near to far
        [1, 5], [2, 6], [3, 7], [4, 8]
    ]
    
    colors = [[1, 0, 0] for _ in lines]  # Red color for frustum
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set


def create_coordinate_frame(origin, size=50.0):
    """Create a coordinate frame at the origin"""
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
    return coord_frame


def create_bounding_box(xyz):
    """Create a bounding box around the Gaussians"""
    min_bound = xyz.min(axis=0).values.numpy()
    max_bound = xyz.max(axis=0).values.numpy()
    
    # Add some padding
    padding = (max_bound - min_bound) * 0.1
    min_bound -= padding
    max_bound += padding
    
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    bbox.color = (0.5, 0.5, 0.5)
    
    return bbox, min_bound, max_bound


def main():
    # Load the model
    model_path = '/home/armin/Documents/Papers/Publications/paper_3/3Dmicro/final_model.pth'
    print(f"Loading model from {model_path}")
    model = load_gaussian_model(model_path)
    
    # Extract data
    xyz = model['xyz']
    intensity = model['intensity']
    scaling = model['scaling']
    rotation = model['rotation']
    config = model['config']
    
    print(f"Number of Gaussians: {model['num_gaussians']}")
    print(f"XYZ range: [{xyz.min().item():.2f}, {xyz.max().item():.2f}]")
    print(f"Intensity range: [{intensity.min().item():.4f}, {intensity.max().item():.4f}]")
    print(f"Image size from config: {config.get('img_size', 'N/A')}")
    
    # Create point cloud from Gaussian centers
    print("Creating point cloud visualization...")
    pcd = create_gaussian_point_cloud(xyz, intensity, scaling)
    
    # Create bounding box
    bbox, min_bound, max_bound = create_bounding_box(xyz)
    center = (min_bound + max_bound) / 2
    extent = max_bound - min_bound
    
    # Create coordinate frame at origin
    coord_frame = create_coordinate_frame([0, 0, 0], size=max(extent) * 0.1)
    
    # Create multiple camera views around the object
    cameras = []
    num_cameras = 6
    radius = max(extent) * 1.5
    
    for i in range(num_cameras):
        angle = 2 * np.pi * i / num_cameras
        cam_pos = center + np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            extent[2] * 0.5
        ])
        
        cam_frustum = create_camera_frustum(
            position=cam_pos,
            look_at=center,
            up=[0, 0, 1],
            fov=45,
            scale=radius * 0.3
        )
        cameras.append(cam_frustum)
    
    # Add a top-down camera
    top_cam_pos = center + np.array([0, 0, radius])
    top_cam = create_camera_frustum(
        position=top_cam_pos,
        look_at=center,
        up=[0, 1, 0],
        fov=45,
        scale=radius * 0.3
    )
    cameras.append(top_cam)
    
    # Prepare geometries for visualization
    geometries = [pcd, bbox, coord_frame] + cameras
    
    print("\nVisualization Controls:")
    print("  - Left click + drag: Rotate view")
    print("  - Scroll: Zoom in/out")
    print("  - Middle click + drag: Pan")
    print("  - 'R': Reset view")
    print("  - 'Q' or Esc: Quit")
    print("\nLaunching Open3D viewer...")
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name="3D Gaussian Splatting Visualization",
        width=1280,
        height=720,
        point_show_normal=False
    )


if __name__ == "__main__":
    main()
