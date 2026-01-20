"""
Blender Gaussian Viewer
Load and visualize 3D Gaussians in Blender.

SETUP:
1. First run in terminal: python export_for_blender.py
   This creates gaussians_blender.npz (no torch needed in Blender)
2. Open Blender's Scripting workspace
3. Load this script
4. Run the script (Alt+P)

Features:
- Loads Gaussian positions as point cloud or sphere instances
- Color-codes by opacity/intensity
- No PyTorch required in Blender!
"""

import bpy
import sys
import os
from pathlib import Path
import numpy as np

# =============================================================================
# Configuration
# =============================================================================

DATA_PATH = "gaussians_blender.npz"  # Created by export_for_blender.py
VISUALIZATION_MODE = "spheres"  # "points" or "spheres"
MAX_GAUSSIANS = 10000  # Limit for performance (None = all)
SPHERE_SIZE_MULTIPLIER = 1.0  # Scale of spheres relative to Gaussian scale


# =============================================================================
# Model Loading (No PyTorch!)
# =============================================================================

def load_gaussian_data(data_path):
    """Load Gaussian data from .npz file."""
    data = np.load(data_path)
    
    xyz = data['xyz']
    opacity = data['opacity']
    scales = data['scales']
    rotation = data['rotation']
    volume_shape = tuple(data['volume_shape'])
    
    print(f"Loaded {len(xyz)} Gaussians")
    print(f"  Position range: [{xyz.min():.3f}, {xyz.max():.3f}]")
    print(f"  Opacity range: [{opacity.min():.3f}, {opacity.max():.3f}]")
    print(f"  Volume shape: {volume_shape}")
    
    return xyz, opacity, scales, rotation, volume_shape


# =============================================================================
# Blender Visualization
# =============================================================================

def clear_scene():
    """Remove all mesh objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def create_material_with_color(name, color, opacity):
    """Create a material with given color and opacity."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # Clear default nodes
    nodes.clear()
    
    # Create new nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    
    # Set color and strength
    emission.inputs['Color'].default_value = (*color, 1.0)
    emission.inputs['Strength'].default_value = opacity * 2.0
    
    # Link nodes
    mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    return mat


def visualize_as_spheres(xyz, opacity, scales, volume_shape=(100, 650, 820)):
    """Visualize Gaussians as sphere instances."""
    D, H, W = volume_shape
    
    # Convert normalized coordinates to world space
    xyz_world = xyz.copy()
    xyz_world[:, 0] = (xyz[:, 0] - 0.5) * D  # z
    xyz_world[:, 1] = (xyz[:, 1] - 0.5) * H  # y
    xyz_world[:, 2] = (xyz[:, 2] - 0.5) * W  # x
    
    # Reorder to Blender convention (x, y, z)
    positions = xyz_world[:, [2, 1, 0]]
    
    # Create base sphere
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 0))
    base_sphere = bpy.context.active_object
    base_sphere.name = "GaussianSphere_Base"
    
    # Create collection for instances
    collection = bpy.data.collections.new("Gaussians")
    bpy.context.scene.collection.children.link(collection)
    
    # Limit number for performance
    n_display = min(len(positions), MAX_GAUSSIANS) if MAX_GAUSSIANS else len(positions)
    
    print(f"Creating {n_display} sphere instances...")
    
    for i in range(n_display):
        # Create instance
        instance = base_sphere.copy()
        instance.data = base_sphere.data
        instance.name = f"Gaussian_{i}"
        
        # Set position
        instance.location = positions[i]
        
        # Set scale (average of 3D Gaussian scales)
        avg_scale = scales[i].mean() * SPHERE_SIZE_MULTIPLIER * 100
        instance.scale = (avg_scale, avg_scale, avg_scale)
        
        # Create material based on opacity
        intensity = opacity[i, 0]
        color = (intensity, intensity, intensity)
        mat = create_material_with_color(f"Mat_{i}", color, intensity)
        instance.data.materials.append(mat)
        
        # Add to collection
        collection.objects.link(instance)
    
    # Hide base sphere
    base_sphere.hide_set(True)
    base_sphere.hide_render = True
    
    print("Visualization complete!")


def visualize_as_points(xyz, opacity, volume_shape=(100, 650, 820)):
    """Visualize Gaussians as point cloud."""
    D, H, W = volume_shape
    
    # Convert normalized coordinates to world space
    xyz_world = xyz.copy()
    xyz_world[:, 0] = (xyz[:, 0] - 0.5) * D  # z
    xyz_world[:, 1] = (xyz[:, 1] - 0.5) * H  # y
    xyz_world[:, 2] = (xyz[:, 2] - 0.5) * W  # x
    
    # Reorder to Blender convention (x, y, z)
    positions = xyz_world[:, [2, 1, 0]]
    
    # Limit number for performance
    n_display = min(len(positions), MAX_GAUSSIANS) if MAX_GAUSSIANS else len(positions)
    
    # Create mesh
    mesh = bpy.data.meshes.new("GaussianPoints")
    vertices = positions[:n_display].tolist()
    mesh.from_pydata(vertices, [], [])
    
    # Create object
    obj = bpy.data.objects.new("Gaussians_PointCloud", mesh)
    bpy.context.collection.objects.link(obj)
    
    # Add vertex colors based on opacity
    if not mesh.vertex_colors:
        mesh.vertex_colors.new()
    
    color_layer = mesh.vertex_colors[0]
    for i, poly in enumerate(mesh.polygons):
        for loop_idx in poly.loop_indices:
            vertex_idx = mesh.loops[loop_idx].vertex_index
            if vertex_idx < n_display:
                intensity = opacity[vertex_idx, 0]
                color_layer.data[loop_idx].color = (intensity, intensity, intensity, 1.0)
    
    # Switch to vertex paint to see colors
    bpy.context.view_layer.objects.active = obj
    
    print(f"Created point cloud with {n_display} points")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    # Get script directory
    script_dir = Path(bpy.data.filepath).parent if bpy.data.filepath else Path.cwd()
    data_path = script_dir / DATA_PATH
    
    if not data_path.exists():
        print(f"ERROR: Data file not found at {data_path}")
        print("Please run 'python export_for_blender.py' first!")
        return
    
    print(f"Loading data from {data_path}")
    xyz, opacity, scales, rotation, volume_shape = load_gaussian_data(str(data_path))
    
    # Clear existing scene
    print("Clearing scene...")
    clear_scene()
    
    # Visualize
    if VISUALIZATION_MODE == "spheres":
        visualize_as_spheres(xyz, opacity, scales, volume_shape)
    else:
        visualize_as_points(xyz, opacity, volume_shape)
    
    # Set viewport shading to rendered
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'RENDERED'
    
    print("Done! Adjust camera and lighting as needed.")


if __name__ == "__main__":
    main()
