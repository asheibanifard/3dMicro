"""
Export Gaussian Splatting Model for WebGL Viewer

Converts trained .pth model files to formats compatible with the interactive 
WebGL viewer (volume.raw + dims.json).

Usage:
    python export_for_viewer.py model.pth --output viewer/
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path


def load_model(model_path, filter_outliers=True, max_scale_threshold=0.095):
    """Load trained Gaussian model from checkpoint.
    
    Args:
        model_path: Path to .pth checkpoint file
        filter_outliers: Remove Gaussians with large scales
        max_scale_threshold: Scale threshold for filtering
        
    Returns:
        xyz, opacity, scales, rotation, config
    """
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    xyz = checkpoint['xyz']
    intensity = checkpoint['intensity']
    scaling = checkpoint['scaling']
    rotation = checkpoint['rotation']
    
    # Convert from optimization space
    opacity = torch.sigmoid(intensity)
    scales = torch.exp(scaling)
    rotation = torch.nn.functional.normalize(rotation, dim=-1)
    
    print(f"Loaded {xyz.shape[0]} Gaussians")
    print(f"  XYZ range: [{xyz.min().item():.3f}, {xyz.max().item():.3f}]")
    print(f"  Opacity range: [{opacity.min().item():.3f}, {opacity.max().item():.3f}]")
    print(f"  Scale range: [{scales.min().item():.4f}, {scales.max().item():.4f}]")
    
    # Filter outliers
    if filter_outliers:
        max_scale_per_gaussian = scales.max(dim=1).values
        valid_mask = max_scale_per_gaussian < max_scale_threshold
        num_filtered = (~valid_mask).sum().item()
        
        if num_filtered > 0:
            xyz = xyz[valid_mask]
            opacity = opacity[valid_mask]
            scales = scales[valid_mask]
            rotation = rotation[valid_mask]
            print(f"  Filtered {num_filtered} outlier Gaussians")
    
    config = checkpoint.get('config', {})
    if 'volume_shape' in checkpoint:
        config['volume_shape'] = checkpoint['volume_shape']
    
    return xyz, opacity, scales, rotation, config


def gaussians_to_volume(xyz, opacity, scales, volume_shape=(64, 256, 320)):
    """Rasterize Gaussians to a 3D volume grid.
    
    Args:
        xyz: (N, 3) Gaussian centers in [0, 1] normalized coords (z, y, x order)
        opacity: (N, 1) Gaussian opacities
        scales: (N, 3) Gaussian scales
        volume_shape: Output volume dimensions (z, y, x)
        
    Returns:
        volume: 3D numpy array with rendered Gaussians
    """
    D, H, W = volume_shape
    volume = np.zeros((D, H, W), dtype=np.float32)
    
    xyz_np = xyz.numpy() if torch.is_tensor(xyz) else xyz
    opacity_np = opacity.numpy().squeeze() if torch.is_tensor(opacity) else opacity.squeeze()
    scales_np = scales.numpy() if torch.is_tensor(scales) else scales
    
    # Scale positions to volume coordinates
    positions = xyz_np * np.array([D-1, H-1, W-1])
    
    # Scale factors in voxels
    scale_voxels = scales_np * np.array([D, H, W])
    
    print(f"Rasterizing {len(xyz_np)} Gaussians to {volume_shape} volume...")
    
    # Efficient rasterization using bounding boxes
    for i in range(len(xyz_np)):
        if i % 1000 == 0:
            print(f"  Processing Gaussian {i}/{len(xyz_np)}")
        
        pos = positions[i]
        op = opacity_np[i]
        sc = scale_voxels[i]
        
        # Bounding box (3 sigma)
        radius = sc * 3
        z_min = max(0, int(pos[0] - radius[0]))
        z_max = min(D, int(pos[0] + radius[0]) + 1)
        y_min = max(0, int(pos[1] - radius[1]))
        y_max = min(H, int(pos[1] + radius[1]) + 1)
        x_min = max(0, int(pos[2] - radius[2]))
        x_max = min(W, int(pos[2] + radius[2]) + 1)
        
        if z_max <= z_min or y_max <= y_min or x_max <= x_min:
            continue
        
        # Create coordinate grids for this bounding box
        zs = np.arange(z_min, z_max)
        ys = np.arange(y_min, y_max)
        xs = np.arange(x_min, x_max)
        
        zz, yy, xx = np.meshgrid(zs, ys, xs, indexing='ij')
        
        # Compute Gaussian values
        dz = (zz - pos[0]) / max(sc[0], 0.01)
        dy = (yy - pos[1]) / max(sc[1], 0.01)
        dx = (xx - pos[2]) / max(sc[2], 0.01)
        
        gaussian = np.exp(-0.5 * (dz**2 + dy**2 + dx**2))
        
        # Accumulate using MIP (max)
        volume[z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(
            volume[z_min:z_max, y_min:y_max, x_min:x_max],
            gaussian * op
        )
    
    return volume


def export_for_webgl(volume, output_dir, dims_name='dims.json', volume_name='volume.raw'):
    """Export volume data for WebGL viewer.
    
    Args:
        volume: 3D numpy array (z, y, x)
        output_dir: Output directory path
        dims_name: Name of dimensions JSON file
        volume_name: Name of raw volume file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    D, H, W = volume.shape
    
    # Normalize to [0, 255]
    vol_min, vol_max = volume.min(), volume.max()
    if vol_max > vol_min:
        volume_norm = (volume - vol_min) / (vol_max - vol_min) * 255
    else:
        volume_norm = np.zeros_like(volume)
    
    volume_uint8 = volume_norm.astype(np.uint8)
    
    # WebGL expects x, y, z order for 3D texture
    # Transpose from (z, y, x) to (x, y, z)
    volume_transposed = np.transpose(volume_uint8, (2, 1, 0))
    
    # Save dimensions (in x, y, z order for WebGL)
    dims = {'x': W, 'y': H, 'z': D}
    with open(output_dir / dims_name, 'w') as f:
        json.dump(dims, f)
    print(f"Saved dimensions: {dims}")
    
    # Save volume as raw bytes
    volume_transposed.flatten().tofile(output_dir / volume_name)
    print(f"Saved volume: {output_dir / volume_name} ({volume_transposed.nbytes / 1024 / 1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description='Export Gaussian model for WebGL viewer')
    parser.add_argument('model', type=str, help='Path to .pth model file')
    parser.add_argument('--output', '-o', type=str, default='viewer/',
                        help='Output directory (default: viewer/)')
    parser.add_argument('--resolution', '-r', type=int, nargs=3, default=None,
                        metavar=('Z', 'Y', 'X'),
                        help='Volume resolution (z y x). Auto-detect from model if not specified.')
    parser.add_argument('--no-filter', action='store_true',
                        help='Disable outlier filtering')
    
    args = parser.parse_args()
    
    # Load model
    xyz, opacity, scales, rotation, config = load_model(
        args.model, 
        filter_outliers=not args.no_filter
    )
    
    # Determine volume shape
    if args.resolution:
        volume_shape = tuple(args.resolution)
    elif 'volume_shape' in config:
        volume_shape = tuple(config['volume_shape'])
    else:
        # Default resolution based on model
        volume_shape = (64, 256, 320)
        print(f"Using default resolution: {volume_shape}")
    
    print(f"Output volume shape: {volume_shape}")
    
    # Rasterize to volume
    volume = gaussians_to_volume(xyz, opacity, scales, volume_shape)
    
    # Export for WebGL
    export_for_webgl(volume, args.output)
    
    print(f"\nExport complete! To view:")
    print(f"  cd {args.output}")
    print(f"  python -m http.server 8000")
    print(f"  Open http://localhost:8000 in browser")


if __name__ == '__main__':
    main()
