#!/usr/bin/env python3
"""
Custom Orthographic Gaussian Rasterizer
Supports parallel ray casting for volume rendering of 3D Gaussians
"""

import torch
import torch.nn.functional as F
import numpy as np
import math


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix
    q: (N, 4) quaternions [w, x, y, z]
    Returns: (N, 3, 3) rotation matrices
    """
    q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    R = torch.zeros(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
    
    R[:, 0, 0] = 1 - 2*y*y - 2*z*z
    R[:, 0, 1] = 2*x*y - 2*w*z
    R[:, 0, 2] = 2*x*z + 2*w*y
    R[:, 1, 0] = 2*x*y + 2*w*z
    R[:, 1, 1] = 1 - 2*x*x - 2*z*z
    R[:, 1, 2] = 2*y*z - 2*w*x
    R[:, 2, 0] = 2*x*z - 2*w*y
    R[:, 2, 1] = 2*y*z + 2*w*x
    R[:, 2, 2] = 1 - 2*x*x - 2*y*y
    
    return R


def build_covariance_3d(scaling, rotation):
    """
    Build 3D covariance matrices from scaling and rotation
    scaling: (N, 3) - scale factors
    rotation: (N, 4) - quaternions
    Returns: (N, 3, 3) covariance matrices
    """
    R = quaternion_to_rotation_matrix(rotation)  # (N, 3, 3)
    S = torch.diag_embed(scaling)  # (N, 3, 3) diagonal scale matrices
    
    # Covariance = R @ S @ S^T @ R^T
    RS = torch.bmm(R, S)
    cov = torch.bmm(RS, RS.transpose(1, 2))
    
    return cov


def project_covariance_to_2d(cov3d, view_matrix):
    """
    Project 3D covariance to 2D for orthographic projection
    cov3d: (N, 3, 3) - 3D covariance matrices
    view_matrix: (3, 3) - rotation part of view matrix
    Returns: (N, 2, 2) - 2D covariance matrices
    """
    # Transform covariance to camera space
    # cov_cam = R @ cov @ R^T
    R = view_matrix[:3, :3]
    cov_cam = torch.einsum('ij,njk,lk->nil', R, cov3d, R)
    
    # For orthographic projection, just take the XY part
    cov2d = cov_cam[:, :2, :2]
    
    return cov2d


def gaussian_2d(x, y, mean, cov2d):
    """
    Evaluate 2D Gaussian at given coordinates
    x, y: (H, W) coordinate grids
    mean: (N, 2) Gaussian centers
    cov2d: (N, 2, 2) 2D covariance matrices
    Returns: (N, H, W) Gaussian values
    """
    N = mean.shape[0]
    H, W = x.shape
    
    # Flatten coordinates
    coords = torch.stack([x.flatten(), y.flatten()], dim=1)  # (H*W, 2)
    
    # Compute inverse covariance
    cov2d_inv = torch.inverse(cov2d + 1e-6 * torch.eye(2, device=cov2d.device).unsqueeze(0))  # (N, 2, 2)
    
    # Compute determinant for normalization
    det = torch.det(cov2d)  # (N,)
    
    # Compute Gaussian values
    # diff = coords - mean  (broadcast to N, H*W, 2)
    diff = coords.unsqueeze(0) - mean.unsqueeze(1)  # (N, H*W, 2)
    
    # Mahalanobis distance: diff @ cov_inv @ diff^T
    mahal = torch.einsum('npi,nij,npj->np', diff, cov2d_inv, diff)  # (N, H*W)
    
    # Gaussian value
    gauss = torch.exp(-0.5 * mahal)  # (N, H*W)
    
    # Reshape to (N, H, W)
    gauss = gauss.view(N, H, W)
    
    return gauss


class OrthoGaussianRasterizer:
    """
    Orthographic Gaussian Rasterizer for parallel ray volume rendering
    """
    
    def __init__(self, width, height, device='cuda'):
        self.width = width
        self.height = height
        self.device = device
        
    def create_camera(self, position, look_at, up, ortho_scale):
        """
        Create orthographic camera parameters
        
        Args:
            position: Camera position (3,)
            look_at: Point camera looks at (3,)
            up: Up vector (3,)
            ortho_scale: Half-size of view volume
            
        Returns:
            dict with camera parameters
        """
        position = np.array(position, dtype=np.float32)
        look_at = np.array(look_at, dtype=np.float32)
        up = np.array(up, dtype=np.float32)
        
        # Camera coordinate system
        forward = look_at - position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up_vec = np.cross(right, forward)
        up_vec = up_vec / np.linalg.norm(up_vec)
        
        # View matrix (world to camera rotation)
        R = np.stack([right, up_vec, -forward], axis=0)  # 3x3
        t = -R @ position
        
        # Full 4x4 view matrix
        view_matrix = np.eye(4, dtype=np.float32)
        view_matrix[:3, :3] = R
        view_matrix[:3, 3] = t
        
        # Orthographic bounds
        aspect = self.width / self.height
        left = -ortho_scale * aspect
        right_bound = ortho_scale * aspect
        bottom = -ortho_scale
        top = ortho_scale
        
        return {
            'position': torch.tensor(position, device=self.device),
            'forward': torch.tensor(forward, device=self.device),
            'right': torch.tensor(right, device=self.device),
            'up': torch.tensor(up_vec, device=self.device),
            'view_matrix': torch.tensor(view_matrix, device=self.device),
            'ortho_scale': ortho_scale,
            'left': left,
            'right': right_bound,
            'bottom': bottom,
            'top': top,
        }
    
    def render(self, camera, xyz, colors, opacity, scaling, rotation, bg_color=None, 
               tile_size=16, max_gaussians_per_tile=256):
        """
        Render Gaussians with orthographic projection
        
        Args:
            camera: Camera parameters from create_camera()
            xyz: (N, 3) Gaussian positions
            colors: (N, 3) Gaussian colors
            opacity: (N, 1) Gaussian opacity
            scaling: (N, 3) Gaussian scales
            rotation: (N, 4) Gaussian rotations (quaternions)
            bg_color: (3,) Background color
            
        Returns:
            rendered_image: (3, H, W) tensor
            depth_image: (1, H, W) tensor
        """
        if bg_color is None:
            bg_color = torch.zeros(3, device=self.device)
        else:
            bg_color = bg_color.to(self.device)
            
        N = xyz.shape[0]
        H, W = self.height, self.width
        
        # Move to device
        xyz = xyz.to(self.device)
        colors = colors.to(self.device)
        opacity = opacity.to(self.device)
        scaling = scaling.to(self.device)
        rotation = rotation.to(self.device)
        
        # Transform points to camera space
        xyz_h = torch.cat([xyz, torch.ones(N, 1, device=self.device)], dim=1)
        xyz_cam = (xyz_h @ camera['view_matrix'].T)[:, :3]  # (N, 3)
        
        # Orthographic projection - just use X, Y from camera space
        # Map to pixel coordinates
        aspect = W / H
        scale_x = W / (2 * camera['ortho_scale'] * aspect)
        scale_y = H / (2 * camera['ortho_scale'])
        
        xy_screen = xyz_cam[:, :2].clone()
        xy_screen[:, 0] = xy_screen[:, 0] * scale_x + W / 2
        xy_screen[:, 1] = -xy_screen[:, 1] * scale_y + H / 2  # Flip Y
        
        # Depth is Z in camera space (positive = in front of camera)
        depth = -xyz_cam[:, 2]  # Negate because camera looks down -Z
        
        # Build 3D covariance matrices
        cov3d = build_covariance_3d(scaling, rotation)
        
        # Project covariance to 2D
        R_cam = camera['view_matrix'][:3, :3]
        cov2d = project_covariance_to_2d(cov3d, R_cam)
        
        # Scale covariance to screen space
        cov2d_screen = cov2d.clone()
        cov2d_screen[:, 0, 0] *= scale_x * scale_x
        cov2d_screen[:, 0, 1] *= scale_x * scale_y
        cov2d_screen[:, 1, 0] *= scale_x * scale_y
        cov2d_screen[:, 1, 1] *= scale_y * scale_y
        
        # Sort by depth (front to back)
        depth_order = torch.argsort(depth, descending=True)  # Closest first
        
        # Create output image
        rendered = bg_color.view(3, 1, 1).expand(3, H, W).clone()
        accumulated_alpha = torch.zeros(1, H, W, device=self.device)
        depth_image = torch.zeros(1, H, W, device=self.device)
        
        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Process Gaussians in batches for memory efficiency
        batch_size = 512
        num_rendered = 0
        
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            batch_indices = depth_order[batch_start:batch_end]
            
            batch_xy = xy_screen[batch_indices]  # (B, 2)
            batch_cov = cov2d_screen[batch_indices]  # (B, 2, 2)
            batch_colors = colors[batch_indices]  # (B, 3)
            batch_opacity = opacity[batch_indices]  # (B, 1)
            batch_depth = depth[batch_indices]  # (B,)
            
            # Filter out Gaussians outside the view
            in_view = (batch_xy[:, 0] > -100) & (batch_xy[:, 0] < W + 100) & \
                      (batch_xy[:, 1] > -100) & (batch_xy[:, 1] < H + 100) & \
                      (batch_depth > 0)
            
            if not in_view.any():
                continue
                
            batch_xy = batch_xy[in_view]
            batch_cov = batch_cov[in_view]
            batch_colors = batch_colors[in_view]
            batch_opacity = batch_opacity[in_view]
            batch_depth = batch_depth[in_view]
            
            B = batch_xy.shape[0]
            num_rendered += B
            
            # Compute Gaussian values for this batch
            # For each Gaussian, compute its contribution to each pixel
            for i in range(B):
                # Get bounding box for this Gaussian (3 sigma)
                eigvals = torch.linalg.eigvalsh(batch_cov[i])
                max_sigma = torch.sqrt(eigvals.max()) * 3
                
                cx, cy = batch_xy[i, 0].int().item(), batch_xy[i, 1].int().item()
                r = int(max_sigma.item()) + 1
                
                x_min = max(0, cx - r)
                x_max = min(W, cx + r + 1)
                y_min = max(0, cy - r)
                y_max = min(H, cy + r + 1)
                
                if x_min >= x_max or y_min >= y_max:
                    continue
                
                # Local coordinate grid
                local_x = x_coords[y_min:y_max, x_min:x_max]
                local_y = y_coords[y_min:y_max, x_min:x_max]
                
                # Compute Gaussian value
                diff_x = local_x - batch_xy[i, 0]
                diff_y = local_y - batch_xy[i, 1]
                
                # Inverse covariance
                cov_inv = torch.inverse(batch_cov[i] + 1e-4 * torch.eye(2, device=self.device))
                
                # Mahalanobis distance
                mahal = cov_inv[0, 0] * diff_x * diff_x + \
                        2 * cov_inv[0, 1] * diff_x * diff_y + \
                        cov_inv[1, 1] * diff_y * diff_y
                
                # Gaussian weight
                gauss_weight = torch.exp(-0.5 * mahal)
                
                # Alpha for this Gaussian
                alpha = batch_opacity[i, 0] * gauss_weight
                alpha = alpha.clamp(0, 0.99)
                
                # Alpha compositing (front to back)
                current_alpha = accumulated_alpha[0, y_min:y_max, x_min:x_max]
                contribution = alpha * (1 - current_alpha)
                
                # Update color
                for c in range(3):
                    rendered[c, y_min:y_max, x_min:x_max] += batch_colors[i, c] * contribution
                
                # Update alpha
                accumulated_alpha[0, y_min:y_max, x_min:x_max] += contribution
                
                # Update depth (weighted by contribution)
                depth_image[0, y_min:y_max, x_min:x_max] += batch_depth[i] * contribution
        
        # Normalize depth by accumulated alpha
        depth_image = depth_image / (accumulated_alpha + 1e-8)
        
        print(f"Rendered {num_rendered} Gaussians")
        
        return rendered.clamp(0, 1), depth_image, num_rendered


# Actual volume dimensions (voxels)
VOLUME_SIZE = (820, 650, 100)  # X, Y, Z


def render_with_ortho_rasterizer(model_path, output_path, width=820, height=650):
    """
    Convenience function to render a Gaussian model with orthographic projection
    Uses actual volume aspect ratio: 820 x 650 x 100
    """
    import torch
    from PIL import Image
    
    # Load model
    print(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    
    xyz = model['xyz'].clone()
    intensity = model['intensity']
    scaling = model['scaling'].clone()
    rotation = model['rotation']
    
    print(f"Number of Gaussians: {model['num_gaussians']}")
    print(f"Volume dimensions: {VOLUME_SIZE[0]} x {VOLUME_SIZE[1]} x {VOLUME_SIZE[2]}")
    
    # Scale coordinates by actual volume dimensions
    # Normalized coords are in [0,1], scale to actual voxel dimensions
    volume_scale = torch.tensor([VOLUME_SIZE[0], VOLUME_SIZE[1], VOLUME_SIZE[2]], dtype=torch.float32)
    xyz = xyz * volume_scale
    
    # Also scale the Gaussian sizes
    scaling = scaling + torch.log(volume_scale).unsqueeze(0)
    
    print(f"Scaled XYZ range: [{xyz.min(dim=0).values.numpy()}] to [{xyz.max(dim=0).values.numpy()}]")
    
    # Process parameters
    # Convert intensity to colors (grayscale to RGB)
    intensity_norm = torch.sigmoid(intensity)
    colors = intensity_norm.expand(-1, 3)  # Grayscale
    
    # Opacity from intensity
    opacity = torch.sigmoid(intensity)
    
    # Activate scaling
    scaling_activated = torch.exp(scaling).clamp(0.1, 100)
    
    # Normalize rotation
    rotation_norm = rotation / (rotation.norm(dim=1, keepdim=True) + 1e-8)
    
    # Scene bounds in scaled coordinates
    xyz_np = xyz.numpy()
    center = xyz_np.mean(axis=0)
    xyz_min = xyz_np.min(axis=0)
    xyz_max = xyz_np.max(axis=0)
    extent = xyz_max - xyz_min
    
    print(f"Scene center (scaled): {center}")
    print(f"Scene extent (scaled): {extent}")
    
    # Camera distance based on largest dimension
    max_dim = max(extent)
    radius = max_dim * 1.5
    
    # Create rasterizer
    rasterizer = OrthoGaussianRasterizer(width, height, device='cuda')
    
    # Camera setup - looking along X axis (front view showing Y-Z plane)
    # For top view (looking down Z), we see X-Y plane (820 x 650)
    cam_pos = center + np.array([0, 0, radius])  # Looking down Z axis
    # ortho_scale is half the view height in world units
    ortho_scale = extent[1] * 0.55  # Y extent for height
    camera = rasterizer.create_camera(
        position=cam_pos,
        look_at=center,
        up=[0, 1, 0],
        ortho_scale=ortho_scale
    )
    
    print(f"Camera position: {cam_pos}")
    print(f"Looking at: {center}")
    
    # Render
    bg_color = torch.tensor([0.0, 0.0, 0.0])
    rendered, depth, num_visible = rasterizer.render(
        camera=camera,
        xyz=xyz,
        colors=colors,
        opacity=opacity,
        scaling=scaling_activated,
        rotation=rotation_norm,
        bg_color=bg_color
    )
    
    # Save
    img_np = (rendered.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    img.save(output_path)
    print(f"Saved to {output_path}")
    
    return rendered, depth


def render_multiple_views(model_path, output_dir):
    """
    Render a Gaussian model from multiple viewpoints with orthographic projection.
    Volume dimensions: 820 x 650 x 100
    """
    import torch
    from PIL import Image
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    
    xyz = model['xyz'].clone()
    intensity = model['intensity']
    scaling = model['scaling'].clone()
    rotation = model['rotation']
    
    print(f"Number of Gaussians: {model['num_gaussians']}")
    print(f"Volume dimensions: {VOLUME_SIZE[0]} x {VOLUME_SIZE[1]} x {VOLUME_SIZE[2]}")
    
    # Scale coordinates by actual volume dimensions
    volume_scale = torch.tensor([VOLUME_SIZE[0], VOLUME_SIZE[1], VOLUME_SIZE[2]], dtype=torch.float32)
    xyz = xyz * volume_scale
    scaling = scaling + torch.log(volume_scale).unsqueeze(0)
    
    # Process parameters
    intensity_norm = torch.sigmoid(intensity)
    colors = intensity_norm.expand(-1, 3)  # Grayscale
    opacity = torch.sigmoid(intensity)
    scaling_activated = torch.exp(scaling).clamp(0.1, 100)
    rotation_norm = rotation / (rotation.norm(dim=1, keepdim=True) + 1e-8)
    
    # Scene bounds in scaled coordinates
    xyz_np = xyz.numpy()
    center = xyz_np.mean(axis=0)
    xyz_min = xyz_np.min(axis=0)
    xyz_max = xyz_np.max(axis=0)
    extent = xyz_max - xyz_min
    
    print(f"Scene center (scaled): {center}")
    print(f"Scene extent (scaled): {extent}")
    
    # Camera distance
    max_dim = max(extent)
    radius = max_dim * 1.5
    
    bg_color = torch.tensor([0.0, 0.0, 0.0])
    
    # Define views with appropriate image sizes based on actual aspect ratios
    # Front/Back: looking along X, seeing Y-Z plane (650 x 100 aspect)
    # Left/Right: looking along Y, seeing X-Z plane (820 x 100 aspect)  
    # Top/Bottom: looking along Z, seeing X-Y plane (820 x 650 aspect)
    views = [
        {'name': 'front', 'offset': [radius, 0, 0], 'up': [0, 0, 1], 'width': 650, 'height': 100, 'scale_idx': 2},
        {'name': 'back', 'offset': [-radius, 0, 0], 'up': [0, 0, 1], 'width': 650, 'height': 100, 'scale_idx': 2},
        {'name': 'left', 'offset': [0, -radius, 0], 'up': [0, 0, 1], 'width': 820, 'height': 100, 'scale_idx': 2},
        {'name': 'right', 'offset': [0, radius, 0], 'up': [0, 0, 1], 'width': 820, 'height': 100, 'scale_idx': 2},
        {'name': 'top', 'offset': [0, 0, radius], 'up': [0, 1, 0], 'width': 820, 'height': 650, 'scale_idx': 1},
        {'name': 'bottom', 'offset': [0, 0, -radius], 'up': [0, 1, 0], 'width': 820, 'height': 650, 'scale_idx': 1},
    ]
    
    rendered_images = []
    
    for view in views:
        width = view['width']
        height = view['height']
        rasterizer = OrthoGaussianRasterizer(width, height, device='cuda')
        
        cam_pos = center + np.array(view['offset'])
        # Scale based on the view's height dimension extent
        ortho_scale = extent[view['scale_idx']] * 0.55
        camera = rasterizer.create_camera(
            position=cam_pos,
            look_at=center,
            up=view['up'],
            ortho_scale=ortho_scale
        )
        
        print(f"\nRendering view: {view['name']}")
        
        rendered, depth, num_visible = rasterizer.render(
            camera=camera,
            xyz=xyz,
            colors=colors,
            opacity=opacity,
            scaling=scaling_activated,
            rotation=rotation_norm,
            bg_color=bg_color
        )
        
        # Save individual view
        img_np = (rendered.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        output_path = os.path.join(output_dir, f'view_{view["name"]}.png')
        img.save(output_path)
        print(f"Saved to {output_path}")
        
        rendered_images.append((view['name'], img, width, height))
    
    # Save individual images only (no combined grid due to different sizes)
    print(f"\nRendered {len(rendered_images)} views")
    
    return rendered_images


def visualize_with_open3d(model_path, add_cameras=True):
    """
    Visualize Gaussians and camera positions in Open3D
    """
    import open3d as o3d
    
    # Load model
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    xyz = model['xyz'].numpy()
    intensity = model['intensity'].numpy()
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # Color by intensity
    intensity_norm = 1.0 / (1.0 + np.exp(-intensity))  # sigmoid
    colors = np.tile(intensity_norm, (1, 3))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    geometries = [pcd]
    
    # Scene bounds
    center = xyz.mean(axis=0)
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    max_extent = np.linalg.norm(extent)
    radius = max_extent * 1.2
    
    # Add coordinate frame at center
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=center)
    geometries.append(coord_frame)
    
    if add_cameras:
        # Add camera frustums
        views = [
            {'name': 'front', 'offset': [radius, 0, 0], 'up': [0, 0, 1], 'color': [1, 0, 0]},
            {'name': 'left', 'offset': [0, -radius, 0], 'up': [0, 0, 1], 'color': [0, 1, 0]},
            {'name': 'top', 'offset': [0, 0, radius], 'up': [0, 1, 0], 'color': [0, 0, 1]},
            {'name': 'iso', 'offset': [radius*0.7, radius*0.7, radius*0.7], 'up': [0, 0, 1], 'color': [1, 1, 0]},
        ]
        
        ortho_scale = max_extent * 0.5  # Half extent to fit full volume
        
        for view in views:
            cam_pos = center + np.array(view['offset'])
            
            # Camera direction
            forward = center - cam_pos
            forward = forward / np.linalg.norm(forward)
            
            right = np.cross(forward, np.array(view['up']))
            right = right / np.linalg.norm(right)
            
            up_vec = np.cross(right, forward)
            up_vec = up_vec / np.linalg.norm(up_vec)
            
            # Create orthographic frustum (rectangular prism)
            half_w = ortho_scale * 800 / 600  # aspect ratio
            half_h = ortho_scale
            depth = radius * 2
            
            # Frustum corners at camera
            corners_local = np.array([
                [-half_w, -half_h, 0],
                [half_w, -half_h, 0],
                [half_w, half_h, 0],
                [-half_w, half_h, 0],
                [-half_w, -half_h, depth],
                [half_w, -half_h, depth],
                [half_w, half_h, depth],
                [-half_w, half_h, depth],
            ])
            
            # Transform to world coordinates
            R = np.stack([right, up_vec, forward], axis=1)
            corners_world = (R @ corners_local.T).T + cam_pos
            
            # Create lines for frustum
            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Near plane
                [4, 5], [5, 6], [6, 7], [7, 4],  # Far plane
                [0, 4], [1, 5], [2, 6], [3, 7],  # Connecting edges
            ]
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(corners_world)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([view['color']] * len(lines))
            geometries.append(line_set)
            
            # Camera position sphere
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sphere.translate(cam_pos)
            sphere.paint_uniform_color(view['color'])
            geometries.append(sphere)
    
    # Visualize
    print("Opening Open3D viewer...")
    o3d.visualization.draw_geometries(geometries, window_name="Gaussian Model with Orthographic Cameras")


if __name__ == "__main__":
    model_path = '/home/armin/Documents/Papers/Publications/paper_3/3Dmicro/final_model.pth'
    
    # Single view render
    output_path = '/home/armin/Documents/Papers/Publications/paper_3/3Dmicro/render_ortho.png'
    render_with_ortho_rasterizer(model_path, output_path)
    
    # Multiple views
    output_dir = '/home/armin/Documents/Papers/Publications/paper_3/3Dmicro/ortho_renders'
    render_multiple_views(model_path, output_dir)
    
    # Optionally visualize with Open3D
    # visualize_with_open3d(model_path)
