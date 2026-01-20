"""
Gaussian Splatting Renderer for Microscopy Volumes.

Supports three rendering modes:
- Alpha blending: Standard 3DGS volumetric compositing (front-to-back)
- MIP: Maximum Intensity Projection (common in fluorescence microscopy)
- Additive: Sum of Gaussian contributions (matches volumetric training)

The key insight: If your Gaussians were trained for volumetric density reconstruction,
use 'additive' mode. Alpha blending causes artifacts because it treats overlapping
Gaussians as occlusion, while additive accumulates all contributions.

Usage:
    python render_gaussians.py --mode additive --gif --output rotation.gif
    python render_gaussians.py --mode mip --angle 45 --output mip_45.tif
    python render_gaussians.py --mode alpha --angle 0 --output alpha_0.tif
"""
import torch
import numpy as np
import tifffile as tiff
import math
import argparse
from typing import Tuple, Optional, Dict, List

# Try to import CUDA rasterizer (optional for alpha mode)
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    HAS_CUDA_RASTERIZER = True
except ImportError:
    HAS_CUDA_RASTERIZER = False
    print("Warning: diff_gaussian_rasterization not found. Alpha mode will use PyTorch fallback.")

# Try to import custom MIP kernel (optional)
try:
    from gaussian_mip import mip_render as cuda_mip_render
    HAS_CUDA_MIP = True
except ImportError:
    HAS_CUDA_MIP = False
    print("Warning: gaussian_mip not found. MIP mode will use PyTorch implementation.")


# =============================================================================
# Model Loading
# =============================================================================

def load_model(model_path: str, 
               filter_outliers: bool = True, 
               max_scale_threshold: float = 0.095) -> Tuple[torch.Tensor, ...]:
    """
    Load trained Gaussian model.
    
    Args:
        model_path: Path to checkpoint (.pth file)
        filter_outliers: Remove Gaussians with runaway scales
        max_scale_threshold: Scale threshold for filtering
    
    Returns:
        xyz, opacity, scales, rotation, config
    """
    checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
    
    xyz = checkpoint['xyz'].cuda()
    opacity = torch.sigmoid(checkpoint['intensity'].cuda())
    scales = torch.exp(checkpoint['scaling'].cuda())
    rotation = torch.nn.functional.normalize(checkpoint['rotation'].cuda(), dim=-1)
    config = checkpoint.get('config', {})
    
    n_total = xyz.shape[0]
    
    if filter_outliers:
        max_scale = scales.max(dim=1).values
        mask = max_scale < max_scale_threshold
        xyz, opacity, scales, rotation = xyz[mask], opacity[mask], scales[mask], rotation[mask]
        n_filtered = n_total - xyz.shape[0]
        if n_filtered > 0:
            print(f"Filtered {n_filtered} outlier Gaussians (scale >= {max_scale_threshold})")
    
    print(f"Loaded {xyz.shape[0]} Gaussians")
    print(f"  Position range: [{xyz.min().item():.3f}, {xyz.max().item():.3f}]")
    print(f"  Scale range: [{scales.min().item():.4f}, {scales.max().item():.4f}]")
    print(f"  Opacity range: [{opacity.min().item():.3f}, {opacity.max().item():.3f}]")
    
    return xyz, opacity, scales, rotation, config


# =============================================================================
# Camera Utilities
# =============================================================================

def create_camera(angle_deg: float, 
                  volume_shape: Tuple[int, int, int], 
                  image_size: Optional[Tuple[int, int]] = None) -> Dict:
    """
    Create camera for viewing volume from given angle.
    
    Args:
        angle_deg: Rotation around Y-axis (degrees)
        volume_shape: (D, H, W) volume dimensions
        image_size: (height, width) or None for auto
    
    Returns:
        Camera parameters dict
    """
    D, H, W = volume_shape
    img_h, img_w = image_size or (H, W)
    
    # Camera setup
    distance = max(D, H, W) * 2.0
    fov = math.pi / 2  # 90 degrees
    
    # Camera position (orbit around Y-axis)
    angle_rad = math.radians(angle_deg)
    eye = np.array([
        distance * math.sin(angle_rad),
        0.0,
        distance * math.cos(angle_rad)
    ])
    
    # Look-at matrix (camera looks at origin)
    # Standard 3DGS uses OpenGL convention: camera looks down -Z
    target = np.array([0.0, 0.0, 0.0])
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    
    # View matrix: [right, up, -forward] rows
    R = np.stack([right, up, -forward], axis=0)
    T = -R @ eye
    
    world_view = np.eye(4, dtype=np.float32)
    world_view[:3, :3] = R
    world_view[:3, 3] = T
    
    # Projection matrix
    znear, zfar = 0.01, distance * 10.0
    fovy = fov * img_h / img_w
    tan_fovx, tan_fovy = math.tan(fov / 2), math.tan(fovy / 2)
    
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 1.0 / tan_fovx
    P[1, 1] = 1.0 / tan_fovy
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    
    world_view_t = torch.tensor(world_view).T.cuda()
    proj_t = torch.tensor(P).T.cuda()
    full_proj = world_view_t @ proj_t
    
    return {
        'world_view': world_view_t,
        'full_proj': full_proj,
        'campos': torch.tensor(eye, dtype=torch.float32).cuda(),
        'tanfovx': tan_fovx,
        'tanfovy': tan_fovy,
        'img_h': img_h,
        'img_w': img_w,
        'angle_rad': angle_rad,
    }


def transform_gaussians_to_world(xyz: torch.Tensor, 
                                  scales: torch.Tensor, 
                                  volume_shape: Tuple[int, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert normalized coordinates to world space.
    
    Args:
        xyz: (N, 3) positions in [0, 1] normalized coordinates (z, y, x order)
        scales: (N, 3) scales in normalized space
        volume_shape: (D, H, W)
    
    Returns:
        means3D: (N, 3) world positions (x, y, z order)
        scales_world: (N, 3) world scales
    """
    D, H, W = volume_shape
    
    # Convert: normalized (z,y,x) -> world (x,y,z) centered at origin
    means3D = torch.stack([
        (xyz[:, 2] - 0.5) * W,
        (xyz[:, 1] - 0.5) * H,
        (xyz[:, 0] - 0.5) * D,
    ], dim=1)
    
    scales_world = torch.stack([
        scales[:, 2] * W,
        scales[:, 1] * H,
        scales[:, 0] * D,
    ], dim=1)
    
    return means3D, scales_world


def project_to_2d(means3D: torch.Tensor, 
                  scales_world: torch.Tensor, 
                  angle_rad: float,
                  volume_shape: Tuple[int, int, int],
                  image_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D Gaussians to 2D (orthographic projection with rotation).
    
    Args:
        means3D: (N, 3) world positions (x, y, z)
        scales_world: (N, 3) world scales
        angle_rad: Rotation angle in radians
        volume_shape: (D, H, W)
        image_size: (img_h, img_w)
    
    Returns:
        means2D: (N, 2) pixel positions
        sigmas2D: (N, 2) projected sigmas in pixels
    """
    D, H, W = volume_shape
    img_h, img_w = image_size
    
    # Rotate around Y-axis
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    
    x_rot = means3D[:, 0] * cos_a - means3D[:, 2] * sin_a
    y_world = means3D[:, 1]
    
    # Project scales (combine x and z scales based on rotation)
    sigma_x = torch.sqrt((scales_world[:, 0] * cos_a)**2 + (scales_world[:, 2] * sin_a)**2)
    sigma_y = scales_world[:, 1]
    
    # Map to pixel coordinates (orthographic)
    px = (x_rot / W + 0.5) * img_w
    py = (y_world / H + 0.5) * img_h
    sigma_x_px = torch.clamp(sigma_x * img_w / W, min=0.5, max=100.0)
    sigma_y_px = torch.clamp(sigma_y * img_h / H, min=0.5, max=100.0)
    
    means2D = torch.stack([px, py], dim=1)
    sigmas2D = torch.stack([sigma_x_px, sigma_y_px], dim=1)
    
    return means2D, sigmas2D


# =============================================================================
# Rendering Functions
# =============================================================================

def render_alpha(xyz: torch.Tensor, 
                 opacity: torch.Tensor, 
                 scales: torch.Tensor, 
                 rotation: torch.Tensor, 
                 volume_shape: Tuple[int, int, int], 
                 angle: float = 0, 
                 image_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Render using alpha blending (standard 3DGS).
    
    Uses diff-gaussian-rasterization CUDA module for tile-based parallel rendering
    with proper front-to-back volumetric compositing.
    
    NOTE: This may produce artifacts if Gaussians were trained for volumetric
    density (additive) rather than view synthesis (alpha blending).
    
    Returns:
        (H, W) grayscale image as numpy array
    """
    if not HAS_CUDA_RASTERIZER:
        print("Falling back to additive mode (CUDA rasterizer not available)")
        return render_additive(xyz, opacity, scales, rotation, volume_shape, angle, image_size)
    
    D, H, W = volume_shape
    cam = create_camera(angle, volume_shape, image_size)
    
    means3D, scales_world = transform_gaussians_to_world(xyz, scales, volume_shape)
    
    # Rasterizer setup
    settings = GaussianRasterizationSettings(
        image_height=cam['img_h'],
        image_width=cam['img_w'],
        tanfovx=cam['tanfovx'],
        tanfovy=cam['tanfovy'],
        bg=torch.zeros(3, device='cuda'),
        scale_modifier=10.0,  # Much larger scale
        viewmatrix=cam['world_view'],
        projmatrix=cam['full_proj'],
        sh_degree=0,
        campos=cam['campos'],
        prefiltered=False,
        debug=False,
        antialiasing=False
    )
    
    rasterizer = GaussianRasterizer(settings)
    
    # Boost intensity significantly for visibility
    intensity = opacity.squeeze()
    intensity_boosted = torch.sqrt(intensity) * 5.0  # Strong boost
    alpha = torch.clamp(intensity_boosted * 0.5, 0, 1).unsqueeze(1)
    colors = (intensity_boosted * 0.5).unsqueeze(1).repeat(1, 3)
    
    # Compute screen space positions manually
    means2D = torch.zeros((means3D.shape[0], 3), device='cuda')
    
    with torch.no_grad():
        image, radii, depth = rasterizer(
            means3D=means3D,
            means2D=means2D,
            opacities=alpha,
            shs=None,
            colors_precomp=colors,
            scales=scales_world,
            rotations=rotation,
            cov3D_precomp=None
        )
    
    # NOTE: Alpha mode may not work with volumetric-trained gaussians
    # If you get black output, use --mode additive_fast or --mode mip instead
    img = image[0].cpu().numpy()
    if img.max() > 0:
        img = np.clip(img * 3.0, 0, 1)
    else:
        print("  WARNING: Alpha mode produced empty image. Try --mode additive_fast or --mode mip")
    
    return img


def render_mip(xyz: torch.Tensor, 
               opacity: torch.Tensor, 
               scales: torch.Tensor, 
               rotation: torch.Tensor, 
               volume_shape: Tuple[int, int, int], 
               angle: float = 0, 
               image_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Render using Maximum Intensity Projection (MIP).
    
    For each pixel, takes the maximum Gaussian contribution along the ray.
    Best for fluorescence microscopy visualization.
    
    Returns:
        (H, W) grayscale image as numpy array
    """
    D, H, W = volume_shape
    img_h, img_w = image_size or (H, W)
    
    means3D, scales_world = transform_gaussians_to_world(xyz, scales, volume_shape)
    
    # Apply scale multiplier for smoother gaussians
    scales_world = scales_world * 1.0
    
    angle_rad = math.radians(angle)
    means2D, sigmas2D = project_to_2d(means3D, scales_world, angle_rad, volume_shape, (img_h, img_w))
    
    intensities = opacity.squeeze()
    
    # Use CUDA kernel if available
    if HAS_CUDA_MIP:
        image = cuda_mip_render(
            means2D.contiguous(), 
            sigmas2D.contiguous(), 
            intensities.contiguous(), 
            img_h, img_w
        )
        return image.cpu().numpy()
    
    # PyTorch fallback (slower but works)
    return _render_mip_pytorch(means2D, sigmas2D, intensities, img_h, img_w)


def _render_mip_pytorch(means2D: torch.Tensor, 
                        sigmas2D: torch.Tensor, 
                        intensities: torch.Tensor, 
                        img_h: int, 
                        img_w: int) -> np.ndarray:
    """PyTorch implementation of MIP rendering."""
    image = torch.zeros(img_h, img_w, device='cuda')
    
    # Create pixel grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(img_h, device='cuda', dtype=torch.float32),
        torch.arange(img_w, device='cuda', dtype=torch.float32),
        indexing='ij'
    )
    
    # Process in batches
    batch_size = 5000
    n_gaussians = means2D.shape[0]
    
    for start in range(0, n_gaussians, batch_size):
        end = min(start + batch_size, n_gaussians)
        
        px = means2D[start:end, 0:1, None]  # [B, 1, 1]
        py = means2D[start:end, 1:2, None]
        sx = sigmas2D[start:end, 0:1, None]
        sy = sigmas2D[start:end, 1:2, None]
        intensity = intensities[start:end, None, None]
        
        # Gaussian evaluation
        dx = x_coords[None] - px
        dy = y_coords[None] - py
        gauss = torch.exp(-0.5 * ((dx / sx)**2 + (dy / sy)**2))
        contribution = intensity * gauss
        
        # MIP: take maximum
        batch_max = contribution.max(dim=0).values
        image = torch.maximum(image, batch_max)
    
    return image.cpu().numpy()


def render_additive(xyz: torch.Tensor, 
                    opacity: torch.Tensor, 
                    scales: torch.Tensor, 
                    rotation: torch.Tensor, 
                    volume_shape: Tuple[int, int, int], 
                    angle: float = 0, 
                    image_size: Optional[Tuple[int, int]] = None,
                    normalize: bool = True,
                    sigma_cutoff: float = 4.0) -> np.ndarray:
    """
    Render using additive blending (sum of Gaussian contributions).
    
    This is the correct mode for Gaussians trained on volumetric density.
    All Gaussian contributions are summed without occlusion.
    
    Formula: I(p) = Σᵢ αᵢ · exp(-0.5 · ||p - μᵢ||²_Σᵢ)
    
    Args:
        xyz: (N, 3) positions
        opacity: (N, 1) intensities  
        scales: (N, 3) scales
        rotation: (N, 4) quaternions (unused in orthographic projection)
        volume_shape: (D, H, W)
        angle: View angle in degrees
        image_size: Output (height, width)
        normalize: Whether to normalize output to [0, 1]
        sigma_cutoff: Ignore contributions beyond this many sigmas
    
    Returns:
        (H, W) grayscale image as numpy array
    """
    D, H, W = volume_shape
    img_h, img_w = image_size or (H, W)
    
    means3D, scales_world = transform_gaussians_to_world(xyz, scales, volume_shape)
    angle_rad = math.radians(angle)
    means2D, sigmas2D = project_to_2d(means3D, scales_world, angle_rad, volume_shape, (img_h, img_w))
    
    intensities = opacity.squeeze()
    
    # Initialize output
    image = torch.zeros(img_h, img_w, device='cuda')
    
    # Create pixel grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(img_h, device='cuda', dtype=torch.float32),
        torch.arange(img_w, device='cuda', dtype=torch.float32),
        indexing='ij'
    )
    
    # Process in batches to manage memory
    batch_size = 5000
    n_gaussians = means2D.shape[0]
    
    for start in range(0, n_gaussians, batch_size):
        end = min(start + batch_size, n_gaussians)
        
        px = means2D[start:end, 0:1, None]  # [B, 1, 1]
        py = means2D[start:end, 1:2, None]
        sx = sigmas2D[start:end, 0:1, None]
        sy = sigmas2D[start:end, 1:2, None]
        intensity = intensities[start:end, None, None]
        
        # Gaussian evaluation
        dx = x_coords[None] - px  # [B, H, W]
        dy = y_coords[None] - py
        
        d2 = (dx / sx)**2 + (dy / sy)**2
        
        # Apply cutoff for efficiency
        mask = d2 < (sigma_cutoff ** 2)
        gauss = torch.where(mask, torch.exp(-0.5 * d2), torch.zeros_like(d2))
        
        contribution = intensity * gauss
        
        # Additive accumulation
        image += contribution.sum(dim=0)
    
    # Normalize
    if normalize and image.max() > 0:
        image = image / image.max()
    
    return image.cpu().numpy()


def render_additive_fast(xyz: torch.Tensor, 
                         opacity: torch.Tensor, 
                         scales: torch.Tensor, 
                         rotation: torch.Tensor, 
                         volume_shape: Tuple[int, int, int], 
                         angle: float = 0, 
                         image_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Fast additive rendering using sparse operations.
    
    Only evaluates Gaussians within their bounding boxes.
    Much faster for sparse volumes (like neurons).
    """
    D, H, W = volume_shape
    img_h, img_w = image_size or (H, W)
    
    means3D, scales_world = transform_gaussians_to_world(xyz, scales, volume_shape)
    angle_rad = math.radians(angle)
    means2D, sigmas2D = project_to_2d(means3D, scales_world, angle_rad, volume_shape, (img_h, img_w))
    
    intensities = opacity.squeeze()*0.1  # Boost brightness by 3x
    
    # Initialize output
    image = torch.zeros(img_h, img_w, device='cuda')
    
    # Render each Gaussian to its bounding box only
    sigma_mult = 3.0  # 3-sigma bounding box
    
    px = means2D[:, 0]
    py = means2D[:, 1]
    sx = sigmas2D[:, 0]
    sy = sigmas2D[:, 1]
    
    # Compute bounding boxes
    x_min = torch.clamp((px - sigma_mult * sx).long(), 0, img_w - 1)
    x_max = torch.clamp((px + sigma_mult * sx).long() + 1, 0, img_w)
    y_min = torch.clamp((py - sigma_mult * sy).long(), 0, img_h - 1)
    y_max = torch.clamp((py + sigma_mult * sy).long() + 1, 0, img_h)
    
    # Sort by bounding box size (render large ones first for better cache)
    areas = (x_max - x_min) * (y_max - y_min)
    sorted_idx = torch.argsort(areas, descending=True)
    
    for idx in sorted_idx:
        i = idx.item()
        xmin, xmax = x_min[i].item(), x_max[i].item()
        ymin, ymax = y_min[i].item(), y_max[i].item()
        
        if xmax <= xmin or ymax <= ymin:
            continue
        
        # Local coordinates
        x_local = torch.arange(xmin, xmax, device='cuda', dtype=torch.float32)
        y_local = torch.arange(ymin, ymax, device='cuda', dtype=torch.float32)
        yy, xx = torch.meshgrid(y_local, x_local, indexing='ij')
        
        # Gaussian evaluation
        dx = (xx - px[i]) / sx[i]
        dy = (yy - py[i]) / sy[i]
        gauss = torch.exp(-0.5 * (dx**2 + dy**2)) * intensities[i]
        
        # Accumulate
        image[ymin:ymax, xmin:xmax] += gauss
    
    # Clamp to [0, 1] instead of normalizing
    image = torch.clamp(image, 0, 1)
    
    return image.cpu().numpy()


# =============================================================================
# High-Level API
# =============================================================================

def render(xyz: torch.Tensor, 
           opacity: torch.Tensor, 
           scales: torch.Tensor, 
           rotation: torch.Tensor, 
           volume_shape: Tuple[int, int, int], 
           angle: float = 0, 
           image_size: Optional[Tuple[int, int]] = None, 
           mode: str = 'additive') -> np.ndarray:
    """
    Render Gaussians to 2D image.
    
    Args:
        xyz: (N, 3) positions in normalized [0,1] coordinates
        opacity: (N, 1) intensities
        scales: (N, 3) Gaussian scales
        rotation: (N, 4) quaternions
        volume_shape: (D, H, W)
        angle: View angle in degrees
        image_size: (height, width) or None
        mode: Rendering mode:
            - 'alpha': Standard 3DGS alpha blending (front-to-back)
            - 'mip': Maximum Intensity Projection
            - 'additive': Sum of contributions (best for volumetric training)
            - 'additive_fast': Sparse additive (faster for neuron-like data)
    
    Returns:
        (H, W) numpy array
    """
    if mode == 'mip':
        return render_mip(xyz, opacity, scales, rotation, volume_shape, angle, image_size)
    elif mode == 'additive':
        return render_additive(xyz, opacity, scales, rotation, volume_shape, angle, image_size)
    elif mode == 'additive_fast':
        return render_additive_fast(xyz, opacity, scales, rotation, volume_shape, angle, image_size)
    elif mode == 'alpha':
        return render_alpha(xyz, opacity, scales, rotation, volume_shape, angle, image_size)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'alpha', 'mip', 'additive', or 'additive_fast'")


def create_rotation_gif(xyz: torch.Tensor, 
                        opacity: torch.Tensor, 
                        scales: torch.Tensor, 
                        rotation: torch.Tensor, 
                        volume_shape: Tuple[int, int, int],
                        num_frames: int = 30, 
                        output_path: str = 'rotation.gif', 
                        fps: int = 10,
                        image_size: Optional[Tuple[int, int]] = None, 
                        mode: str = 'additive') -> List[np.ndarray]:
    """
    Create animated GIF rotating around the volume.
    
    Args:
        num_frames: Number of frames in animation
        output_path: Output file path
        fps: Frames per second
        image_size: (height, width) or None
        mode: Rendering mode
    
    Returns:
        List of frames as numpy arrays
    """
    try:
        import imageio
    except ImportError:
        raise ImportError("Please install imageio: pip install imageio")
    
    frames = []
    angles = np.linspace(0, 360, num_frames, endpoint=False)
    
    print(f"Creating {num_frames}-frame animation using '{mode}' mode...")
    
    for i, angle in enumerate(angles):
        print(f"  Frame {i+1}/{num_frames} ({angle:.0f}°)", end='\r')
        img = render(xyz, opacity, scales, rotation, volume_shape, angle, image_size, mode)
        img = np.clip(img, 0, 1)
        frames.append((img * 255).astype(np.uint8))
    
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"\nSaved {output_path}")
    return frames


def compare_modes(xyz: torch.Tensor, 
                  opacity: torch.Tensor, 
                  scales: torch.Tensor, 
                  rotation: torch.Tensor, 
                  volume_shape: Tuple[int, int, int],
                  angle: float = 0,
                  image_size: Optional[Tuple[int, int]] = None,
                  output_path: str = 'comparison.png') -> Dict[str, np.ndarray]:
    """
    Render with all modes and create side-by-side comparison.
    
    Useful for debugging to see which mode matches your training.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Please install matplotlib: pip install matplotlib")
    
    modes = ['alpha', 'mip', 'additive']
    images = {}
    
    print(f"Rendering comparison at angle {angle}°...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, mode in zip(axes, modes):
        print(f"  Rendering {mode}...")
        img = render(xyz, opacity, scales, rotation, volume_shape, angle, image_size, mode)
        images[mode] = img
        
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'{mode.upper()}\nmin={img.min():.3f}, max={img.max():.3f}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison to {output_path}")
    return images


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Render 3D Gaussian Splatting for Microscopy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Rendering Modes:
  alpha     - Standard 3DGS with front-to-back alpha blending
              Best for: View synthesis trained Gaussians
              
  mip       - Maximum Intensity Projection  
              Best for: Fluorescence microscopy visualization
              
  additive  - Sum of all Gaussian contributions (no occlusion)
              Best for: Volumetric density trained Gaussians
              
  additive_fast - Sparse additive (bounding box optimization)
              Best for: Sparse structures like neurons

Examples:
  # Compare all modes to find which matches your training
  python render_gaussians.py --compare --output comparison.png
  
  # Create rotation animation with additive mode
  python render_gaussians.py --mode additive --gif --output rotation.gif
  
  # Single frame render
  python render_gaussians.py --mode additive --angle 45 --output render.tif
        """
    )
    parser.add_argument('--model', default='final_model.pth', help='Model path')
    parser.add_argument('--output', '-o', default='render.tif', help='Output file')
    parser.add_argument('--mode', choices=['alpha', 'mip', 'additive', 'additive_fast'], 
                        default='additive',
                        help='Rendering mode (default: additive)')
    parser.add_argument('--angle', type=float, default=0, help='View angle (degrees)')
    parser.add_argument('--gif', action='store_true', help='Create rotation GIF')
    parser.add_argument('--frames', type=int, default=30, help='GIF frames')
    parser.add_argument('--fps', type=int, default=10, help='GIF fps')
    parser.add_argument('--size', type=int, nargs=2, metavar=('H', 'W'), help='Image size')
    parser.add_argument('--compare', action='store_true', help='Compare all rendering modes')
    args = parser.parse_args()
    
    # Load model
    xyz, opacity, scales, rotation, config = load_model(args.model)
    volume_shape = tuple(config.get('img_size', (100, 650, 820)))
    image_size = tuple(args.size) if args.size else None
    
    print(f"Volume shape: {volume_shape}")
    print(f"Rendering mode: {args.mode}")
    
    if args.compare:
        compare_modes(xyz, opacity, scales, rotation, volume_shape,
                      args.angle, image_size, args.output.replace('.tif', '.png'))
    elif args.gif:
        create_rotation_gif(xyz, opacity, scales, rotation, volume_shape,
                            args.frames, args.output, args.fps, image_size, args.mode)
    else:
        img = render(xyz, opacity, scales, rotation, volume_shape,
                     args.angle, image_size, args.mode)
        
        if args.output.endswith('.tif') or args.output.endswith('.tiff'):
            tiff.imwrite(args.output, (img * 65535).astype(np.uint16))
        else:
            import imageio
            imageio.imwrite(args.output, (img * 255).astype(np.uint8))
        
        print(f"Saved {args.output}")
        print(f"  Intensity range: [{img.min():.4f}, {img.max():.4f}]")


if __name__ == '__main__':
    main()