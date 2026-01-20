"""
Render final_model.pth using r2_gaussian's voxelization.
"""
import sys
import torch
import numpy as np
import tifffile as tiff

# Add r2_gaussian to path
sys.path.insert(0, './r2_gaussian')

from xray_gaussian_rasterization_voxelization import (
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)


def load_model(model_path):
    """Load our model format and return Gaussian parameters."""
    checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
    
    # Our format: xyz is normalized [0,1], intensity is logit, scaling is log
    xyz = checkpoint['xyz'].cuda()  # (N, 3) normalized [0,1]
    intensity = checkpoint['intensity'].cuda()  # (N, 1) logit
    scaling = checkpoint['scaling'].cuda()  # (N, 3) log scale
    rotation = checkpoint['rotation'].cuda()  # (N, 4) quaternion
    
    # Convert intensity from logit to density
    density = torch.sigmoid(intensity)  # (N, 1)
    
    # Convert scaling from log to actual scale
    scales = torch.exp(scaling)  # (N, 3)
    
    # Normalize rotation quaternions
    rotation = torch.nn.functional.normalize(rotation, dim=-1)
    
    print(f"Loaded {xyz.shape[0]} Gaussians")
    print(f"  XYZ range: [{xyz.min().item():.3f}, {xyz.max().item():.3f}]")
    print(f"  Density range: [{density.min().item():.3f}, {density.max().item():.3f}]")
    print(f"  Scale range: [{scales.min().item():.4f}, {scales.max().item():.4f}]")
    
    return xyz, density, scales, rotation, checkpoint.get('config', {})


def splat_gaussians_pytorch(xyz, density, scales, rotation, volume_shape):
    """
    Pure PyTorch Gaussian splatting into a volume using trilinear interpolation.
    
    Args:
        xyz: (N, 3) Gaussian centers in normalized [0,1] coordinates (z, y, x order)
        density: (N, 1) Gaussian densities [0,1]
        scales: (N, 3) Gaussian scales in normalized space
        rotation: (N, 4) Gaussian rotations (not used for isotropic splatting)
        volume_shape: (D, H, W) output volume shape
    """
    D, H, W = volume_shape
    device = xyz.device
    
    # Create output volume
    volume = torch.zeros((1, 1, D, H, W), device=device)
    
    # Convert normalized xyz to voxel coordinates
    # xyz is (z_norm, y_norm, x_norm)
    z_vox = xyz[:, 0] * (D - 1)  # [0, D-1]
    y_vox = xyz[:, 1] * (H - 1)  # [0, H-1]
    x_vox = xyz[:, 2] * (W - 1)  # [0, W-1]
    
    # Convert scales to voxel space (isotropic for simplicity)
    # Take mean scale and multiply by average dimension
    avg_dim = (D + H + W) / 3.0
    sigma = scales.mean(dim=1) * avg_dim  # (N,)
    
    # Clamp sigma to reasonable range
    sigma = torch.clamp(sigma, min=0.5, max=20.0)
    
    print(f"  Sigma range: [{sigma.min().item():.2f}, {sigma.max().item():.2f}]")
    
    # For each Gaussian, splat into the volume
    # Use a kernel size based on sigma (3 sigma covers 99.7%)
    N = xyz.shape[0]
    
    for i in range(N):
        if i % 1000 == 0:
            print(f"  Processing Gaussian {i}/{N}...")
        
        cx, cy, cz = x_vox[i].item(), y_vox[i].item(), z_vox[i].item()
        s = sigma[i].item()
        d = density[i, 0].item()
        
        # Kernel radius (3 sigma)
        r = int(np.ceil(3 * s))
        
        # Bounds
        z0, z1 = max(0, int(cz) - r), min(D, int(cz) + r + 1)
        y0, y1 = max(0, int(cy) - r), min(H, int(cy) + r + 1)
        x0, x1 = max(0, int(cx) - r), min(W, int(cx) + r + 1)
        
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            continue
        
        # Create local grid
        zz = torch.arange(z0, z1, device=device, dtype=torch.float32)
        yy = torch.arange(y0, y1, device=device, dtype=torch.float32)
        xx = torch.arange(x0, x1, device=device, dtype=torch.float32)
        
        grid_z, grid_y, grid_x = torch.meshgrid(zz, yy, xx, indexing='ij')
        
        # Gaussian values
        dist_sq = (grid_x - cx)**2 + (grid_y - cy)**2 + (grid_z - cz)**2
        gauss = d * torch.exp(-0.5 * dist_sq / (s**2 + 1e-8))
        
        # Add to volume
        volume[0, 0, z0:z1, y0:y1, x0:x1] += gauss
    
    return volume.squeeze()


def query_volume_r2(xyz, density, scales, rotation, volume_shape, voxel_size=None):
    """
    Voxelize Gaussians into a volume using r2_gaussian's CUDA voxelizer.
    
    This uses the r2_gaussian coordinate convention:
    - Volume is centered at (center_x, center_y, center_z)
    - Each voxel has size (sVoxel_x, sVoxel_y, sVoxel_z)
    - Voxel (i,j,k) covers the region centered at:
      (center_x + (i - nVoxel_x/2) * sVoxel_x, ...)
    """
    D, H, W = volume_shape
    
    # r2_gaussian convention: volume spans from -extent/2 to +extent/2
    # With sVoxel=1 and center=0, voxel indices go from -n/2 to n/2
    
    # Our xyz is normalized [0,1] in (z, y, x) order
    # Convert to r2_gaussian's (x, y, z) world coordinates
    # Map [0,1] -> [-W/2, W/2] etc.
    
    xyz_world = torch.zeros_like(xyz)
    xyz_world[:, 0] = (xyz[:, 2] - 0.5) * W  # x: [-W/2, W/2]
    xyz_world[:, 1] = (xyz[:, 1] - 0.5) * H  # y: [-H/2, H/2]  
    xyz_world[:, 2] = (xyz[:, 0] - 0.5) * D  # z: [-D/2, D/2]
    
    # Scales in voxel units (also reorder from z,y,x to x,y,z)
    # Our scales are in normalized space [0, 0.1], convert to voxel space
    scales_world = torch.zeros_like(scales)
    scales_world[:, 0] = scales[:, 2] * W
    scales_world[:, 1] = scales[:, 1] * H
    scales_world[:, 2] = scales[:, 0] * D
    
    # Center at origin, voxel size = 1
    center = [0.0, 0.0, 0.0]
    sVoxel = [1.0, 1.0, 1.0]
    
    print(f"Voxelizing into volume of shape {volume_shape}")
    print(f"  XYZ world range: x=[{xyz_world[:,0].min().item():.2f}, {xyz_world[:,0].max().item():.2f}]")
    print(f"                   y=[{xyz_world[:,1].min().item():.2f}, {xyz_world[:,1].max().item():.2f}]")
    print(f"                   z=[{xyz_world[:,2].min().item():.2f}, {xyz_world[:,2].max().item():.2f}]")
    print(f"  Scale range: [{scales_world.min().item():.4f}, {scales_world.max().item():.4f}]")
    
    voxel_settings = GaussianVoxelizationSettings(
        scale_modifier=1.0,
        nVoxel_x=int(W),
        nVoxel_y=int(H),
        nVoxel_z=int(D),
        sVoxel_x=float(sVoxel[0]),
        sVoxel_y=float(sVoxel[1]),
        sVoxel_z=float(sVoxel[2]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        prefiltered=False,
        debug=False,
    )
    voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)
    
    vol_pred, radii = voxelizer(
        means3D=xyz_world,
        opacities=density,
        scales=scales_world,
        rotations=rotation,
        cov3D_precomp=None,
    )
    
    print(f"  Volume output shape: {vol_pred.shape}")
    print(f"  Volume range: [{vol_pred.min().item():.6f}, {vol_pred.max().item():.6f}]")
    
    return vol_pred


def render_2d_projection(xyz, density, scales, rotation, volume_shape, 
                         angle=0, projection='orthographic', image_size=None):
    """
    Render 3D Gaussians to a 2D image using Gaussian splatting.
    
    Args:
        xyz: (N, 3) Gaussian centers in normalized [0,1] coordinates (z, y, x order)
        density: (N, 1) Gaussian densities [0,1]
        scales: (N, 3) Gaussian scales in normalized space
        rotation: (N, 4) Gaussian rotations as quaternions
        volume_shape: (D, H, W) volume shape for coordinate conversion
        angle: Camera rotation angle in degrees (0 = looking along z-axis)
        projection: 'orthographic' or 'perspective'
        image_size: (height, width) output image size, default uses (H, W)
    
    Returns:
        2D image array
    """
    D, H, W = volume_shape
    device = xyz.device
    
    if image_size is None:
        img_h, img_w = H, W
    else:
        img_h, img_w = image_size
    
    # Convert normalized xyz to world coordinates
    # xyz is (z_norm, y_norm, x_norm) in [0, 1]
    z_world = (xyz[:, 0] - 0.5) * D  # depth
    y_world = (xyz[:, 1] - 0.5) * H  # height  
    x_world = (xyz[:, 2] - 0.5) * W  # width
    
    # Apply rotation around y-axis (vertical) based on angle
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    # Rotate x, z coordinates
    x_rot = x_world * cos_a - z_world * sin_a
    z_rot = x_world * sin_a + z_world * cos_a
    y_rot = y_world
    
    # Convert scales to world space (take mean for isotropic projection)
    scale_z = scales[:, 0] * D
    scale_y = scales[:, 1] * H
    scale_x = scales[:, 2] * W
    
    # For orthographic projection, project to xy plane (ignore z)
    # Image coordinates: map from world to pixel
    # World x: [-W/2, W/2] -> pixel [0, img_w]
    # World y: [-H/2, H/2] -> pixel [0, img_h]
    
    if projection == 'orthographic':
        # Direct orthographic projection
        px = (x_rot / W + 0.5) * img_w
        py = (y_rot / H + 0.5) * img_h
        
        # Project scales to 2D (use x and y scales)
        # After rotation, the projected scale depends on angle
        sigma_x = (scale_x * abs(cos_a) + scale_z * abs(sin_a)) * img_w / W
        sigma_y = scale_y * img_h / H
        
    else:  # perspective
        # Simple perspective projection
        focal_length = max(D, H, W)  # focal length in world units
        camera_z = -D * 2  # camera position behind the volume
        
        # Project with perspective
        depth = z_rot - camera_z
        depth = torch.clamp(depth, min=0.1)  # avoid division by zero
        
        px = (x_rot * focal_length / depth / W + 0.5) * img_w
        py = (y_rot * focal_length / depth / H + 0.5) * img_h
        
        # Scale also affected by depth
        scale_factor = focal_length / depth
        sigma_x = scale_x * scale_factor * img_w / W
        sigma_y = scale_y * scale_factor * img_h / H
    
    # Clamp sigmas
    sigma_x = torch.clamp(sigma_x, min=0.5, max=50.0)
    sigma_y = torch.clamp(sigma_y, min=0.5, max=50.0)
    
    print(f"Rendering 2D projection (angle={angle}°, {projection})")
    print(f"  Image size: {img_h} x {img_w}")
    print(f"  Projected x range: [{px.min().item():.1f}, {px.max().item():.1f}]")
    print(f"  Projected y range: [{py.min().item():.1f}, {py.max().item():.1f}]")
    print(f"  Sigma x range: [{sigma_x.min().item():.2f}, {sigma_x.max().item():.2f}]")
    print(f"  Sigma y range: [{sigma_y.min().item():.2f}, {sigma_y.max().item():.2f}]")
    
    # Create output image
    image = torch.zeros((img_h, img_w), device=device)
    
    # Sort by depth for proper alpha compositing (back to front)
    depth_order = torch.argsort(z_rot, descending=True)  # far to near
    
    N = xyz.shape[0]
    
    # Splat each Gaussian
    for idx in range(N):
        i = depth_order[idx].item()
        
        if idx % 2000 == 0:
            print(f"  Splatting Gaussian {idx}/{N}...")
        
        cx, cy = px[i].item(), py[i].item()
        sx, sy = sigma_x[i].item(), sigma_y[i].item()
        d = density[i, 0].item()
        
        # Skip if outside image bounds
        if cx < -sx*3 or cx > img_w + sx*3 or cy < -sy*3 or cy > img_h + sy*3:
            continue
        
        # Kernel radius (3 sigma)
        rx, ry = int(np.ceil(3 * sx)), int(np.ceil(3 * sy))
        
        # Bounds
        y0, y1 = max(0, int(cy) - ry), min(img_h, int(cy) + ry + 1)
        x0, x1 = max(0, int(cx) - rx), min(img_w, int(cx) + rx + 1)
        
        if y1 <= y0 or x1 <= x0:
            continue
        
        # Create local grid
        yy = torch.arange(y0, y1, device=device, dtype=torch.float32)
        xx = torch.arange(x0, x1, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
        
        # Anisotropic Gaussian
        gauss = d * torch.exp(-0.5 * ((grid_x - cx)**2 / (sx**2 + 1e-8) + 
                                       (grid_y - cy)**2 / (sy**2 + 1e-8)))
        
        # Alpha compositing (additive for now)
        image[y0:y1, x0:x1] += gauss
    
    return image


def render_multiple_views(xyz, density, scales, rotation, volume_shape, 
                          angles=[0, 45, 90], projection='orthographic'):
    """
    Render multiple 2D views from different angles.
    """
    views = []
    for angle in angles:
        print(f"\n--- Rendering view at {angle}° ---")
        view = render_2d_projection(xyz, density, scales, rotation, volume_shape,
                                    angle=angle, projection=projection)
        views.append(view.cpu().numpy())
    return views, angles


def create_rotation_gif(xyz, density, scales, rotation, volume_shape,
                        num_frames=30, projection='orthographic', output_path='rotation.gif',
                        fps=10):
    """
    Create a GIF animation rotating around the volume.
    
    Args:
        num_frames: Number of frames (angles) in the animation
        projection: 'orthographic' or 'perspective'
        output_path: Output GIF path
        fps: Frames per second
    """
    import imageio
    
    angles = np.linspace(0, 360, num_frames, endpoint=False)
    frames = []
    
    for i, angle in enumerate(angles):
        print(f"Rendering frame {i+1}/{num_frames} (angle={angle:.1f}°)...")
        
        view = render_2d_projection(xyz, density, scales, rotation, volume_shape,
                                    angle=angle, projection=projection)
        view_np = view.cpu().numpy()
        
        # Normalize to [0, 255] for GIF
        v_min, v_max = view_np.min(), view_np.max()
        if v_max > v_min:
            view_np = (view_np - v_min) / (v_max - v_min)
        
        # Convert to 8-bit
        frame = (view_np * 255).astype(np.uint8)
        frames.append(frame)
    
    # Save as GIF
    print(f"\nSaving GIF to {output_path}...")
    duration = 1000 // fps  # milliseconds per frame
    imageio.mimsave(output_path, frames, duration=duration, loop=0)
    print(f"Created GIF with {num_frames} frames at {fps} fps")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Render Gaussians using r2_gaussian voxelizer')
    parser.add_argument('--model_path', type=str, default='final_model.pth',
                        help='Path to the model checkpoint')
    parser.add_argument('--output', type=str, default='rendered_volume.tif',
                        help='Output TIFF file')
    parser.add_argument('--shape', type=int, nargs=3, default=[109, 657, 818],
                        help='Output volume shape (D H W)')
    parser.add_argument('--method', type=str, default='pytorch', choices=['r2', 'pytorch', '2d'],
                        help='Rendering method: r2 (CUDA voxelizer), pytorch (pure PyTorch 3D), or 2d (2D projection)')
    parser.add_argument('--angle', type=float, default=0,
                        help='Camera angle in degrees for 2D projection (0 = front view)')
    parser.add_argument('--projection', type=str, default='orthographic', choices=['orthographic', 'perspective'],
                        help='Projection type for 2D rendering')
    parser.add_argument('--multi_view', action='store_true',
                        help='Render multiple views at different angles')
    parser.add_argument('--gif', action='store_true',
                        help='Create a rotating GIF animation')
    parser.add_argument('--num_frames', type=int, default=30,
                        help='Number of frames for GIF animation')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for GIF')
    args = parser.parse_args()
    
    # Load model
    xyz, density, scales, rotation, config = load_model(args.model_path)
    
    # Get volume shape from config if available
    if 'img_size' in config:
        volume_shape = config['img_size']
        print(f"Using volume shape from config: {volume_shape}")
    else:
        volume_shape = tuple(args.shape)
        print(f"Using specified volume shape: {volume_shape}")
    
    # Render based on method
    with torch.no_grad():
        if args.method == '2d':
            # 2D projection rendering
            if args.gif:
                # Create rotating GIF
                gif_output = args.output.replace('.tif', '.gif')
                create_rotation_gif(xyz, density, scales, rotation, volume_shape,
                                    num_frames=args.num_frames, 
                                    projection=args.projection,
                                    output_path=gif_output,
                                    fps=args.fps)
            elif args.multi_view:
                views, angles = render_multiple_views(xyz, density, scales, rotation, volume_shape,
                                                      angles=[0, 45, 90, 135, 180],
                                                      projection=args.projection)
                # Save each view
                for view, angle in zip(views, angles):
                    # Normalize
                    v_min, v_max = view.min(), view.max()
                    if v_max > v_min:
                        view = (view - v_min) / (v_max - v_min)
                    view_16bit = (view * 65535).astype(np.uint16)
                    output_path = args.output.replace('.tif', f'_angle{int(angle)}.tif')
                    tiff.imwrite(output_path, view_16bit)
                    print(f"Saved {output_path}")
            else:
                # Single view
                image = render_2d_projection(xyz, density, scales, rotation, volume_shape,
                                             angle=args.angle, projection=args.projection)
                image_np = image.cpu().numpy()
                
                # Normalize
                img_min, img_max = image_np.min(), image_np.max()
                print(f"Image range: [{img_min:.4f}, {img_max:.4f}]")
                if img_max > img_min:
                    image_np = (image_np - img_min) / (img_max - img_min)
                
                # Save
                image_16bit = (image_np * 65535).astype(np.uint16)
                tiff.imwrite(args.output, image_16bit)
                print(f"Saved to {args.output}")
        
        elif args.method == 'r2':
            volume = query_volume_r2(xyz, density, scales, rotation, volume_shape)
            # r2 output is (W, H, D), transpose to (D, H, W)
            volume_np = volume.squeeze().cpu().numpy()
            volume_np = np.transpose(volume_np, (2, 1, 0))
            
            print(f"Final volume shape: {volume_np.shape}")
            print(f"Volume range: [{volume_np.min():.4f}, {volume_np.max():.4f}]")
            
            # Normalize
            vol_min, vol_max = volume_np.min(), volume_np.max()
            if vol_max > vol_min:
                volume_np = (volume_np - vol_min) / (vol_max - vol_min)
            
            volume_16bit = (volume_np * 65535).astype(np.uint16)
            tiff.imwrite(args.output, volume_16bit)
            print(f"Saved to {args.output}")
        
        else:  # pytorch
            volume = splat_gaussians_pytorch(xyz, density, scales, rotation, volume_shape)
            volume_np = volume.cpu().numpy()
            
            print(f"Final volume shape: {volume_np.shape}")
            print(f"Volume range: [{volume_np.min():.4f}, {volume_np.max():.4f}]")
            
            # Normalize
            vol_min, vol_max = volume_np.min(), volume_np.max()
            if vol_max > vol_min:
                volume_np = (volume_np - vol_min) / (vol_max - vol_min)
            
            volume_16bit = (volume_np * 65535).astype(np.uint16)
            tiff.imwrite(args.output, volume_16bit)
            print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
