"""
Export Gaussian model to Blender-compatible format.
Saves Gaussian data as .npz file that Blender can load without PyTorch.

Usage:
    python export_for_blender.py
    python export_for_blender.py --model skeleton_cylinder_gaussians.pth --output gaussians.npz
"""

import torch
import numpy as np
import argparse


def export_gaussians(model_path, output_path):
    """
    Export Gaussian model to numpy format for Blender.
    
    Args:
        model_path: Path to .pth checkpoint
        output_path: Path to save .npz file
    """
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    xyz = checkpoint['xyz'].cpu().numpy()
    opacity = torch.sigmoid(checkpoint['intensity']).cpu().numpy()
    scales = torch.exp(checkpoint['scaling']).cpu().numpy()
    rotation = checkpoint['rotation'].cpu().numpy()
    config = checkpoint.get('config', {})
    
    # Filter outliers
    max_scale = scales.max(axis=1)
    mask = max_scale < 0.095
    xyz = xyz[mask]
    opacity = opacity[mask]
    scales = scales[mask]
    rotation = rotation[mask]
    
    print(f"Filtered to {len(xyz)} Gaussians")
    print(f"  Position range: [{xyz.min():.3f}, {xyz.max():.3f}]")
    print(f"  Opacity range: [{opacity.min():.3f}, {opacity.max():.3f}]")
    
    # Save to numpy format
    np.savez_compressed(
        output_path,
        xyz=xyz,
        opacity=opacity,
        scales=scales,
        rotation=rotation,
        volume_shape=config.get('img_size', (100, 650, 820))
    )
    
    print(f"Saved to {output_path}")
    print(f"File size: {np.stat(output_path).st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Gaussians for Blender")
    parser.add_argument('--model', default='final_model.pth', help='Model path')
    parser.add_argument('--output', default='gaussians_blender.npz', help='Output .npz path')
    args = parser.parse_args()
    
    export_gaussians(args.model, args.output)
