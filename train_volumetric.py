"""
Volumetric Gaussian Splatting Training (Improved)

Train 3D Gaussians to represent volumetric microscopy data.
Key improvements:
- Anisotropic Gaussian initialization along skeleton branches
- Stratified sampling (foreground-biased)
- Tube-aware loss with skeleton edge sampling
- Better densification with connectivity preservation
"""
import sys
import torch
import torch.nn.functional as F
import numpy as np
import tifffile as tiff
from tqdm import tqdm
import os
import math
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Enable TF32 for faster training on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Import GaussianModel from gaussian-splatting
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gaussian-splatting'))
from scene.gaussian_model import GaussianModel

# Import CUDA MIP kernel (differentiable version)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gaussian_mip_src'))


def load_swc(swc_path):
    """Load SWC file and extract 3D coordinates and connectivity."""
    nodes = []
    with open(swc_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 7:
                n = int(parts[0])
                x, y, z, r = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                p = int(parts[6])
                nodes.append([n, x, y, z, r, p])
    
    nodes = np.array(nodes)
    node_id_to_idx = {int(nodes[i, 0]): i for i in range(len(nodes))}
    
    coords = nodes[:, 1:4]
    radii = nodes[:, 4]
    parent_ids = nodes[:, 5].astype(int)
    parents = np.array([node_id_to_idx.get(p, -1) if p != -1 else -1 for p in parent_ids])
    
    print(f"Loaded {len(coords)} points from SWC")
    print(f"  Coordinate range: x=[{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}]")
    print(f"                    y=[{coords[:, 1].min():.1f}, {coords[:, 1].max():.1f}]")
    print(f"                    z=[{coords[:, 2].min():.1f}, {coords[:, 2].max():.1f}]")
    print(f"  Radius range: [{radii.min():.2f}, {radii.max():.2f}]")
    print(f"  Edges: {np.sum(parents >= 0)}")
    
    return coords, radii, parents


def quaternion_from_direction(direction):
    """Compute quaternion to rotate z-axis to align with given direction.
    
    Args:
        direction: (3,) normalized direction vector
        
    Returns:
        quaternion: (4,) quaternion [w, x, y, z]
    """
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    z_axis = np.array([0, 0, 1], dtype=np.float32)
    
    # Handle parallel/antiparallel cases
    dot = np.dot(z_axis, direction)
    if dot > 0.9999:
        return np.array([1, 0, 0, 0], dtype=np.float32)
    elif dot < -0.9999:
        return np.array([0, 1, 0, 0], dtype=np.float32)
    
    # Cross product gives rotation axis
    axis = np.cross(z_axis, direction)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    
    # Angle from dot product
    angle = np.arccos(np.clip(dot, -1, 1))
    
    # Quaternion from axis-angle
    w = np.cos(angle / 2)
    xyz = axis * np.sin(angle / 2)
    
    return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float32)


def initialize_gaussians_anisotropic(swc_coords, swc_radii, parents, volume_shape):
    """Initialize anisotropic Gaussians elongated along skeleton branches.
    
    Args:
        swc_coords: (N, 3) SWC coordinates (x, y, z) in pixel space
        swc_radii: (N,) SWC radii in pixels
        parents: (N,) parent indices
        volume_shape: (D, H, W) target volume dimensions
        
    Returns:
        xyz, colors, opacities, scales, rotations arrays
        Note: xyz and scales are in canonical (x, y, z) order normalized to [0, 1]
    """
    D, H, W = volume_shape
    N = len(swc_coords)
    
    # Normalize coordinates to [0, 1] in CANONICAL (x, y, z) order
    # This matches standard 3DGS conventions and makes quaternion math correct
    xyz = np.zeros((N, 3), dtype=np.float32)
    xyz[:, 0] = swc_coords[:, 0] / W  # x
    xyz[:, 1] = swc_coords[:, 1] / H  # y
    xyz[:, 2] = swc_coords[:, 2] / D  # z
    
    # Initialize opacities
    opacities = np.ones((N,), dtype=np.float32) * 0.9
    
    # Initialize scales and rotations based on branch direction
    # Scales are in (x, y, z) order matching xyz
    scales = np.zeros((N, 3), dtype=np.float32)
    rotations = np.zeros((N, 4), dtype=np.float32)
    rotations[:, 0] = 1.0  # default identity
    
    for i in range(N):
        parent_idx = parents[i]
        radius = swc_radii[i]
        
        if parent_idx >= 0:
            # Compute direction from parent to child IN PIXEL SPACE
            direction_xyz = swc_coords[i] - swc_coords[parent_idx]  # (x, y, z) in pixels
            edge_length = np.linalg.norm(direction_xyz)
            
            if edge_length > 1e-6:
                direction_xyz = direction_xyz / edge_length
                
                # Use larger isotropic scales for smoother rendering
                # Minimum 3.0 pixels prevents isolated dots/noise
                scale_iso = max(radius * 2.0, 3.0)  # at least 3 pixels for smooth overlap
                
                # Store in (x, y, z) order in normalized coordinates
                scales[i, 0] = scale_iso / W  # x
                scales[i, 1] = scale_iso / H  # y
                scales[i, 2] = scale_iso / D  # z
                
                # Keep rotation as identity for now - let optimization learn anisotropy
                # This avoids the bug where incorrect rotations cause z-stretching
                rotations[i, 0] = 1.0  # identity quaternion
            else:
                # Fallback to isotropic (x, y, z) order
                scales[i] = np.array([radius / W, radius / H, radius / D]) * 2.0
        else:
            # Root node: isotropic based on radius (x, y, z) order
            scales[i] = np.array([radius / W, radius / H, radius / D]) * 2.0
    
    # Clamp scales to reasonable range - increased minimum
    scales = np.clip(scales, 0.005, 0.12)
    
    # Colors (grayscale white)
    colors = np.ones((N, 3), dtype=np.float32) * 0.5
    
    print(f"Initialized {N} anisotropic Gaussians")
    print(f"  XYZ range: [{xyz.min():.3f}, {xyz.max():.3f}]")
    print(f"  Scale range: [{scales.min():.4f}, {scales.max():.4f}]")
    print(f"  Anisotropy ratio (max/min per Gaussian): {(scales.max(axis=1) / (scales.min(axis=1) + 1e-6)).mean():.2f}")
    
    return xyz, colors, opacities, scales, rotations


def sample_along_skeleton_edges(swc_coords, parents, volume_shape, samples_per_edge=20):
    """Sample dense points along skeleton edges for tube-aware loss.
    
    Args:
        swc_coords: (N, 3) coordinates in pixel space (x, y, z)
        parents: (N,) parent indices
        volume_shape: (D, H, W)
        samples_per_edge: points to sample per edge
        
    Returns:
        edge_points: (M, 3) tensor of (z, y, x) voxel coordinates
    """
    D, H, W = volume_shape
    points = []
    
    for i, parent_idx in enumerate(parents):
        if parent_idx >= 0:
            child = swc_coords[i]
            parent = swc_coords[parent_idx]
            
            # Sample along edge
            for t in np.linspace(0, 1, samples_per_edge):
                interp = parent + t * (child - parent)
                x, y, z = interp
                # Convert to (z, y, x) voxel coordinates
                z_idx = np.clip(z, 0, D - 1)
                y_idx = np.clip(y, 0, H - 1)
                x_idx = np.clip(x, 0, W - 1)
                points.append([z_idx, y_idx, x_idx])
    
    # Also add node positions
    for coord in swc_coords:
        x, y, z = coord
        z_idx = np.clip(z, 0, D - 1)
        y_idx = np.clip(y, 0, H - 1)
        x_idx = np.clip(x, 0, W - 1)
        points.append([z_idx, y_idx, x_idx])
    
    points = np.array(points, dtype=np.float32)
    print(f"Sampled {len(points)} points along skeleton edges")
    
    return torch.from_numpy(points)


def precompute_skeleton_tangents(swc_coords, parents, volume_shape):
    """Precompute skeleton points with their local tangent directions.
    
    Args:
        swc_coords: (N, 3) coordinates in pixel space (x, y, z)
        parents: (N,) parent indices
        volume_shape: (D, H, W)
        
    Returns:
        skeleton_info: dict with 'positions' (M, 3) and 'tangents' (M, 3)
        Both are in CANONICAL (x, y, z) order, normalized to [0, 1]
    """
    D, H, W = volume_shape
    positions = []
    tangents = []
    
    for i, parent_idx in enumerate(parents):
        if parent_idx >= 0:
            child = swc_coords[i]  # (x, y, z) in pixels
            parent = swc_coords[parent_idx]
            
            # Tangent direction (child - parent) in (x, y, z)
            direction = child - parent
            length = np.linalg.norm(direction)
            if length > 1e-6:
                tangent = direction / length
            else:
                tangent = np.array([1., 0., 0.])  # default direction
            
            # Store midpoint in CANONICAL (x, y, z) normalized coordinates
            midpoint = (parent + child) / 2
            x, y, z = midpoint
            pos_norm = np.array([x / W, y / H, z / D])  # canonical (x, y, z) order
            
            # Tangent in canonical (x, y, z) normalized space
            tangent_norm = np.array([tangent[0] / W, tangent[1] / H, tangent[2] / D])
            tangent_norm = tangent_norm / (np.linalg.norm(tangent_norm) + 1e-8)
            
            positions.append(pos_norm)
            tangents.append(tangent_norm)
            
            # Also store both endpoints
            for point in [parent, child]:
                x, y, z = point
                pos_norm = np.array([x / W, y / H, z / D])  # canonical (x, y, z) order
                positions.append(pos_norm)
                tangents.append(tangent_norm)  # same tangent for the edge
    
    positions = np.array(positions, dtype=np.float32)
    tangents = np.array(tangents, dtype=np.float32)
    
    print(f"Precomputed {len(positions)} skeleton points with tangents")
    
    return {
        'positions': torch.from_numpy(positions),  # (M, 3) in normalized [0,1] coords
        'tangents': torch.from_numpy(tangents)     # (M, 3) normalized tangent vectors
    }


def compute_skeleton_alignment_loss(gaussians, skeleton_info, volume_shape, device):
    """Compute loss penalizing misalignment between Gaussian principal axes and skeleton tangents.
    
    For each Gaussian, finds nearest skeleton point and penalizes angle between
    the Gaussian's principal axis and the local skeleton tangent.
    
    All computations are in CANONICAL (x, y, z) order - both GaussianModel params
    and skeleton_info should be in this order.
    
    Args:
        gaussians: GaussianModel with xyz in (x, y, z) order
        skeleton_info: dict with 'positions' and 'tangents' in (x, y, z) order
        volume_shape: (D, H, W)
        device: torch device
        
    Returns:
        loss: scalar tensor
    """
    xyz = gaussians.get_xyz  # (N, 3) normalized [0,1] in (x, y, z)
    scales = gaussians.get_scaling  # (N, 3) in (x, y, z)
    rotations = gaussians.get_rotation  # (N, 4) quaternions
    
    N = xyz.shape[0]
    skel_pos = skeleton_info['positions'].to(device)  # (M, 3)
    skel_tan = skeleton_info['tangents'].to(device)   # (M, 3)
    M = skel_pos.shape[0]
    
    # For efficiency, sample a subset of Gaussians if too many
    if N > 1000:
        sample_idx = torch.randperm(N, device=device)[:1000]
        xyz_sample = xyz[sample_idx]
        scales_sample = scales[sample_idx]
        rotations_sample = rotations[sample_idx]
    else:
        xyz_sample = xyz
        scales_sample = scales
        rotations_sample = rotations
    
    N_sample = xyz_sample.shape[0]
    
    # Find nearest skeleton point for each Gaussian
    # (N_sample, 1, 3) - (1, M, 3) -> (N_sample, M)
    dist_sq = ((xyz_sample.unsqueeze(1) - skel_pos.unsqueeze(0)) ** 2).sum(dim=2)
    nearest_idx = dist_sq.argmin(dim=1)  # (N_sample,)
    
    # Get tangent for each Gaussian's nearest skeleton point
    target_tangents = skel_tan[nearest_idx]  # (N_sample, 3)
    
    # Get each Gaussian's principal axis (column of rotation matrix for max scale)
    rot_matrices = quaternion_to_rotation_matrix(rotations_sample)  # (N_sample, 3, 3)
    principal_idx = scales_sample.argmax(dim=1)  # (N_sample,)
    
    # Extract principal axis for each Gaussian
    # rot_matrices[:, :, principal_idx] but with per-sample indexing
    batch_idx = torch.arange(N_sample, device=device)
    principal_axes = rot_matrices[batch_idx, :, principal_idx]  # (N_sample, 3)
    
    # Normalize axes (should already be normalized, but ensure)
    principal_axes = principal_axes / (principal_axes.norm(dim=1, keepdim=True) + 1e-8)
    
    # Compute alignment: |cos(angle)| = |dot product|
    # We want principal axis aligned with tangent (cos = ±1), so penalize deviation from 1
    alignment = torch.abs((principal_axes * target_tangents).sum(dim=1))  # (N_sample,)
    
    # Loss: penalize when alignment is low (axis perpendicular to tangent)
    # alignment ∈ [0, 1], we want it close to 1
    loss = (1.0 - alignment).mean()
    
    return loss


def sample_voxels_stratified(target_volume, num_samples, foreground_ratio=0.3, threshold=0.1):
    """Sample voxels with explicit foreground/background split.
    
    Args:
        target_volume: (D, H, W) tensor
        num_samples: total samples
        foreground_ratio: fraction of samples from foreground (default 0.3 = more background)
        threshold: intensity threshold for foreground
        
    Returns:
        fg_samples: (n_fg, 3) foreground coordinates
        bg_samples: (n_bg, 3) background coordinates
    """
    D, H, W = target_volume.shape
    device = target_volume.device
    
    # Find foreground voxels
    fg_mask = target_volume > threshold
    fg_indices = fg_mask.nonzero()  # (M, 3)
    
    n_fg = int(num_samples * foreground_ratio)
    n_bg = num_samples - n_fg
    
    # Sample foreground
    if len(fg_indices) > 0 and n_fg > 0:
        fg_idx = torch.randint(0, len(fg_indices), (n_fg,), device=device)
        fg_samples = fg_indices[fg_idx].float()
    else:
        fg_samples = torch.empty((0, 3), device=device)
        n_bg += n_fg  # fallback to all background if no foreground
    
    # Sample background (explicitly from low-intensity regions)
    bg_mask = target_volume <= threshold
    bg_indices = bg_mask.nonzero()
    
    if len(bg_indices) > 0 and n_bg > 0:
        bg_idx = torch.randint(0, len(bg_indices), (n_bg,), device=device)
        bg_samples = bg_indices[bg_idx].float()
    else:
        # Fallback: uniform random
        bg_samples = torch.stack([
            torch.randint(0, D, (n_bg,), device=device).float(),
            torch.randint(0, H, (n_bg,), device=device).float(),
            torch.randint(0, W, (n_bg,), device=device).float()
        ], dim=1)
    
    return fg_samples, bg_samples


def query_gaussians_fast(means_world, opacity, inv_cov, voxel_coords):
    """Optimized Gaussian evaluation kernel."""
    # Compute distances: (M, N, 3)
    diff = voxel_coords.unsqueeze(1) - means_world.unsqueeze(0)  # (M, N, 3)
    
    # Mahalanobis distance: diff @ inv_cov @ diff^T
    # Optimized: (M, N, 3) @ (N, 3, 3) -> (M, N, 3)
    diff_transformed = torch.einsum('mnk,nkj->mnj', diff, inv_cov)
    mahal_sq = (diff * diff_transformed).sum(dim=2)  # (M, N)
    
    # Gaussian response
    gaussian_vals = torch.exp(-0.5 * mahal_sq)  # (M, N)
    
    # Weighted sum with opacity (MIP approximation)
    weighted = gaussian_vals * opacity.unsqueeze(0)  # (M, N)
    return weighted.max(dim=1).values  # (M,)


class GaussianSpatialIndex:
    """Spatial index for efficient Gaussian queries using voxel binning.
    
    Pre-bins Gaussians into a coarse 3D grid. For each query point, only
    Gaussians from nearby bins are evaluated, reducing O(M*N) to O(M*k)
    where k << N for sparse volumes.
    """
    
    def __init__(self, query_params, volume_shape, bin_size=8.0, radius_multiplier=3.0):
        """Build spatial index from precomputed Gaussian parameters.
        
        Args:
            query_params: dict with 'means_world', 'opacity', 'precision', 'scales_world', 'norm_factor'
            volume_shape: (D, H, W)
            bin_size: size of each bin in voxels
            radius_multiplier: query radius = radius_multiplier * max_sigma
        """
        self.device = query_params['means_world'].device
        D, H, W = volume_shape
        self.volume_shape = volume_shape
        self.bin_size = bin_size
        self.radius_multiplier = radius_multiplier
        
        # Grid dimensions
        self.grid_shape = (
            int(np.ceil(D / bin_size)),
            int(np.ceil(H / bin_size)),
            int(np.ceil(W / bin_size))
        )
        
        # Use precomputed parameters
        self.means_world = query_params['means_world']  # (N, 3)
        self.scales_world = query_params['scales_world']  # (N, 3)
        self.opacity = query_params['opacity']  # (N,)
        self.precision = query_params['precision']  # (N, 3, 3)
        self.norm_factor = query_params['norm_factor']  # (N,) normalization for proper density
        
        # Compute effective radius for each Gaussian (max scale * multiplier)
        self.radii = self.scales_world.max(dim=1).values * radius_multiplier  # (N,)
        
        # Build bin index: which Gaussians are in each bin
        self._build_index()
    
    def _build_index(self):
        """Build spatial hash map: bin -> list of Gaussian indices."""
        N = len(self.means_world)
        gd, gh, gw = self.grid_shape
        
        # For each Gaussian, find all bins it could affect (based on position + radius)
        bin_to_gaussians = {}
        
        means_np = self.means_world.detach().cpu().numpy()
        radii_np = self.radii.detach().cpu().numpy()
        
        for i in range(N):
            center = means_np[i]
            r = radii_np[i]
            
            # Bin range this Gaussian covers
            bin_min = np.floor((center - r) / self.bin_size).astype(int)
            bin_max = np.ceil((center + r) / self.bin_size).astype(int)
            
            # Clamp to grid bounds
            bin_min = np.maximum(bin_min, 0)
            bin_max = np.minimum(bin_max, [gd, gh, gw])
            
            # Add to all covered bins
            for bz in range(bin_min[0], bin_max[0]):
                for by in range(bin_min[1], bin_max[1]):
                    for bx in range(bin_min[2], bin_max[2]):
                        key = (bz, by, bx)
                        if key not in bin_to_gaussians:
                            bin_to_gaussians[key] = []
                        bin_to_gaussians[key].append(i)
        
        # Convert to tensors for efficient GPU lookup
        self.bin_to_gaussians = {}
        for key, indices in bin_to_gaussians.items():
            self.bin_to_gaussians[key] = torch.tensor(indices, device=self.device, dtype=torch.long)
        
        # Stats
        n_bins = len(bin_to_gaussians)
        if n_bins > 0:
            avg_per_bin = sum(len(v) for v in bin_to_gaussians.values()) / n_bins
        else:
            avg_per_bin = 0
        self.stats = {'n_bins': n_bins, 'avg_per_bin': avg_per_bin}
    
    def query(self, voxel_coords):
        """Query Gaussian field at given voxel coordinates.
        
        Args:
            voxel_coords: (M, 3) voxel coordinates
            
        Returns:
            values: (M,) rendered intensities
        """
        M = len(voxel_coords)
        device = self.device
        
        # Determine which bin each query point falls into
        bin_coords = (voxel_coords / self.bin_size).long()
        bin_coords = bin_coords.cpu().numpy()
        
        # Query each point
        results = torch.zeros(M, device=device)
        
        # Group queries by bin for efficiency
        bin_to_queries = {}
        for qi in range(M):
            key = tuple(bin_coords[qi])
            if key not in bin_to_queries:
                bin_to_queries[key] = []
            bin_to_queries[key].append(qi)
        
        # Process each bin's queries
        for bin_key, query_indices in bin_to_queries.items():
            if bin_key not in self.bin_to_gaussians:
                continue  # No Gaussians in this bin
            
            gauss_indices = self.bin_to_gaussians[bin_key]
            if len(gauss_indices) == 0:
                continue
            
            query_indices_t = torch.tensor(query_indices, device=device, dtype=torch.long)
            coords = voxel_coords[query_indices_t]  # (Q, 3)
            Q = len(coords)
            K = len(gauss_indices)
            
            # Get relevant Gaussian params
            means = self.means_world[gauss_indices]  # (K, 3)
            prec = self.precision[gauss_indices]  # (K, 3, 3)
            opac = self.opacity[gauss_indices]  # (K,)
            norm = self.norm_factor[gauss_indices]  # (K,)
            
            # Compute Mahalanobis distance
            diff = coords.unsqueeze(1) - means.unsqueeze(0)  # (Q, K, 3)
            temp = torch.einsum('qki,kij->qkj', diff, prec)  # (Q, K, 3)
            mahal_sq = (temp * diff).sum(dim=2)  # (Q, K)
            
            # Normalized Gaussian values weighted by opacity
            # norm_factor ensures scale changes don't trivially affect amplitude
            gaussian_vals = torch.exp(-0.5 * mahal_sq.clamp(max=20))
            contributions = gaussian_vals * opac.unsqueeze(0) * norm.unsqueeze(0)  # (Q, K)
            
            # Sum contributions
            results[query_indices_t] = contributions.sum(dim=1)
        
        return results


def precompute_gaussian_query_params(gaussians, volume_shape):
    """Precompute Gaussian parameters for efficient querying.
    
    Call this once per iteration, then pass the result to query functions.
    Avoids recomputing covariance/precision matrices multiple times.
    
    IMPORTANT: GaussianModel stores parameters in canonical (x, y, z) order.
    This function converts to voxel (z, y, x) order for volume queries.
    
    Args:
        gaussians: GaussianModel with xyz in (x, y, z) order
        volume_shape: (D, H, W) - depth, height, width
        normalize: if True, normalize Gaussians by 1/sqrt(det(cov)) to ensure
                   scale changes don't trivially affect amplitude
        
    Returns:
        query_params: dict with precomputed parameters in (z, y, x) voxel order
    """
    D, H, W = volume_shape
    device = gaussians.get_xyz.device
    
    # Dimension tensors for coordinate conversion
    # Canonical (x, y, z) -> voxel (z, y, x) requires reordering
    xyz_dims = torch.tensor([W, H, D], device=device, dtype=torch.float32)  # for (x, y, z) -> pixels
    zyx_dims = torch.tensor([D, H, W], device=device, dtype=torch.float32)  # for (z, y, x) voxel order
    
    # Get raw parameters (in canonical x, y, z order)
    xyz_canonical = gaussians.get_xyz  # (N, 3) normalized [0,1] in (x, y, z)
    opacity = gaussians.get_opacity.squeeze(1)  # (N,)
    scales_canonical = gaussians.get_scaling  # (N, 3) in (x, y, z)
    rotations = gaussians.get_rotation  # (N, 4) quaternions
    
    # Convert xyz from canonical (x, y, z) to voxel (z, y, x) order for volume queries
    # xyz_canonical[:, 0] = x/W, xyz_canonical[:, 1] = y/H, xyz_canonical[:, 2] = z/D
    xyz_zyx = torch.stack([
        xyz_canonical[:, 2],  # z
        xyz_canonical[:, 1],  # y
        xyz_canonical[:, 0],  # x
    ], dim=1)  # (N, 3) in (z, y, x) order, still normalized [0,1]
    
    # Convert to world/voxel coordinates
    means_world = xyz_zyx * zyx_dims  # (N, 3) in voxel space (z, y, x)
    
    # Convert scales from (x, y, z) to (z, y, x) order
    scales_zyx = torch.stack([
        scales_canonical[:, 2],  # z scale
        scales_canonical[:, 1],  # y scale
        scales_canonical[:, 0],  # x scale
    ], dim=1)
    # Minimum 2.0 voxels prevents isolated dots/noise artifacts
    scales_world = torch.clamp(scales_zyx * zyx_dims, min=2.0)  # (N, 3) in voxel space
    
    # Compute rotation matrices from quaternions
    # Note: quaternions work in canonical (x, y, z) space, so we need to
    # permute the rotation matrix to work in (z, y, x) space
    rot_matrices_xyz = quaternion_to_rotation_matrix(rotations)  # (N, 3, 3) in (x, y, z)
    
    # Permutation matrix to convert (x, y, z) -> (z, y, x): P @ v_xyz = v_zyx
    # This swaps x and z axes
    perm = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=device, dtype=torch.float32)
    # R_zyx = P @ R_xyz @ P^T
    rot_matrices = perm @ rot_matrices_xyz @ perm.T  # (N, 3, 3) in (z, y, x)
    
    # Compute covariance matrices: R @ S @ S @ R^T
    S = torch.diag_embed(scales_world)  # (N, 3, 3)
    cov = rot_matrices @ S @ S @ rot_matrices.transpose(-1, -2)  # (N, 3, 3)
    
    # Compute precision (inverse covariance) with regularization
    cov_reg = cov + torch.eye(3, device=device) * 1e-4
    precision = torch.linalg.inv(cov_reg)  # (N, 3, 3)
    
    # NOTE: Gaussian normalization (1/sqrt(det(cov))) is DISABLED for volumetric rendering.
    # With scales of 2-20 voxels, det(cov) becomes huge (e.g. 250000), making norm_factor
    # extremely small (~0.002) and causing vanishing rendered values.
    # Instead, we let opacity handle the amplitude - this is standard in 3DGS.
    # The loss will learn appropriate opacity values to match target intensities.
    norm_factor = torch.ones(opacity.shape[0], device=device)
    
    return {
        'means_world': means_world,
        'scales_world': scales_world,
        'opacity': opacity,
        'precision': precision,
        'norm_factor': norm_factor,
        'n_gaussians': xyz_canonical.shape[0]
    }


# Global cache for spatial index
_spatial_index_cache = {'n_gaussians': -1, 'index': None, 'iteration': -1}


def query_gaussians_chunked(gaussians, voxel_coords, volume_shape, chunk_size=4000, 
                            use_spatial_index=True, query_params=None):
    """Evaluate Gaussians at voxel positions with spatial indexing for efficiency.
    
    Args:
        gaussians: GaussianModel
        voxel_coords: (M, 3) voxel coordinates (z, y, x)
        volume_shape: (D, H, W)
        chunk_size: process this many samples at a time (for fallback)
        use_spatial_index: if True, use spatial binning for O(M*k) instead of O(M*N)
        query_params: precomputed params from precompute_gaussian_query_params() (optional)
        
    Returns:
        rendered_values: (M,) intensities
    """
    global _spatial_index_cache
    
    D, H, W = volume_shape
    device = gaussians.get_xyz.device
    M = len(voxel_coords)
    N = gaussians.get_xyz.shape[0]
    
    # Precompute params if not provided
    if query_params is None:
        query_params = precompute_gaussian_query_params(gaussians, volume_shape)
    
    # NOTE: The Python-based spatial index has too much overhead for training.
    # The brute-force chunked approach is faster for N < ~5000 Gaussians due to:
    # 1. Spatial index rebuilding every iteration (Python loops)
    # 2. Python dict lookups and CPU-GPU sync overhead
    # For production, consider a CUDA-based spatial hash or BVH.
    use_spatial_index = False  # Disabled - brute force is faster for training
    
    if use_spatial_index and N > 5000:
        # Only use spatial index for very large N where brute force is too slow
        spatial_index = GaussianSpatialIndex(query_params, volume_shape)
        return spatial_index.query(voxel_coords)
    
    # Fallback: brute force with chunking (for small N or when index disabled)
    means_world = query_params['means_world']
    opacity = query_params['opacity']
    precision = query_params['precision']
    norm_factor = query_params['norm_factor']
    
    # Process in chunks
    rendered_chunks = []
    
    for i in range(0, M, chunk_size):
        chunk_coords = voxel_coords[i:i+chunk_size]  # (C, 3)
        
        # Compute differences: (C, N, 3)
        diff = chunk_coords.unsqueeze(1) - means_world.unsqueeze(0)
        
        # Mahalanobis distance: diff^T @ precision @ diff
        # (C, N, 3) @ (N, 3, 3) -> (C, N, 3)
        temp = torch.einsum('cni,nij->cnj', diff, precision)
        # (C, N, 3) * (C, N, 3) -> (C, N)
        mahal_sq = (temp * diff).sum(dim=2)
        del diff, temp  # Free memory immediately
        
        # Normalized Gaussian values
        gaussian_vals = torch.exp(-0.5 * mahal_sq.clamp(max=20))  # (C, N)
        del mahal_sq  # Free memory
        
        # Weight by opacity and normalization factor, then sum
        chunk_rendered = (gaussian_vals * opacity.unsqueeze(0) * norm_factor.unsqueeze(0)).sum(dim=1)
        del gaussian_vals  # Free memory
        
        rendered_chunks.append(chunk_rendered)
    
    return torch.cat(rendered_chunks, dim=0)


def quaternion_to_rotation_matrix(q):
    """Convert quaternions to rotation matrices.
    
    Args:
        q: (N, 4) quaternions [w, x, y, z]
        
    Returns:
        R: (N, 3, 3) rotation matrices
    """
    # Normalize quaternions
    q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
    
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Rotation matrix from quaternion
    R = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y], dim=1),
        torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x], dim=1),
        torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y], dim=1)
    ], dim=1)  # (N, 3, 3)
    
    return R


def trilinear_sample(volume, coords):
    """Sample volume at continuous coordinates using trilinear interpolation.
    
    Args:
        volume: (D, H, W) tensor
        coords: (M, 3) coordinates (z, y, x)
        
    Returns:
        values: (M,) sampled values
    """
    D, H, W = volume.shape
    device = volume.device
    
    # Clamp coordinates
    z = coords[:, 0].clamp(0, D - 1.001)
    y = coords[:, 1].clamp(0, H - 1.001)
    x = coords[:, 2].clamp(0, W - 1.001)
    
    # Get integer indices
    z0, y0, x0 = z.long(), y.long(), x.long()
    z1, y1, x1 = (z0 + 1).clamp(max=D-1), (y0 + 1).clamp(max=H-1), (x0 + 1).clamp(max=W-1)
    
    # Get fractional parts
    zd, yd, xd = z - z0.float(), y - y0.float(), x - x0.float()
    
    # Trilinear interpolation
    c000 = volume[z0, y0, x0]
    c001 = volume[z0, y0, x1]
    c010 = volume[z0, y1, x0]
    c011 = volume[z0, y1, x1]
    c100 = volume[z1, y0, x0]
    c101 = volume[z1, y0, x1]
    c110 = volume[z1, y1, x0]
    c111 = volume[z1, y1, x1]
    
    c00 = c000 * (1 - xd) + c001 * xd
    c01 = c010 * (1 - xd) + c011 * xd
    c10 = c100 * (1 - xd) + c101 * xd
    c11 = c110 * (1 - xd) + c111 * xd
    
    c0 = c00 * (1 - yd) + c01 * yd
    c1 = c10 * (1 - yd) + c11 * yd
    
    return c0 * (1 - zd) + c1 * zd

def render_slice_comparison(gaussians, target_volume, volume_shape, slice_positions, output_path, iteration, max_resolution=256):
    """
    Render and save slice comparisons between ground truth and Gaussian reconstruction.
    
    Args:
        gaussians: GaussianModel
        target_volume: (D, H, W) ground truth volume
        volume_shape: (D, H, W)
        slice_positions: list of (axis, position) tuples, e.g., [('z', 50), ('y', 300), ...]
        output_path: directory to save images
        iteration: current iteration number
        max_resolution: maximum resolution for slice rendering (downsample if larger)
    """
    D, H, W = volume_shape
    device = gaussians.get_xyz.device
    
    for axis, pos in slice_positions:
        if axis == 'z':
            # XY slice at z=pos - original shape (H, W)
            orig_shape = (H, W)
            # Compute downsample factor
            downsample = max(1, max(H, W) // max_resolution)
            render_H, render_W = H // downsample, W // downsample
            
            y_coords, x_coords = torch.meshgrid(
                torch.linspace(0, H-1, render_H, device=device, dtype=torch.float32),
                torch.linspace(0, W-1, render_W, device=device, dtype=torch.float32),
                indexing='ij'
            )
            z_coords = torch.full_like(x_coords, pos, dtype=torch.float32)
            voxel_coords = torch.stack([z_coords.flatten(), y_coords.flatten(), x_coords.flatten()], dim=1)
            
            # Ground truth (downsampled)
            gt_slice = target_volume[pos, ::downsample, ::downsample]
            slice_shape = (render_H, render_W)
            
        elif axis == 'y':
            # XZ slice at y=pos - original shape (D, W)
            orig_shape = (D, W)
            downsample = max(1, max(D, W) // max_resolution)
            render_D, render_W = D // downsample, W // downsample
            
            z_coords, x_coords = torch.meshgrid(
                torch.linspace(0, D-1, render_D, device=device, dtype=torch.float32),
                torch.linspace(0, W-1, render_W, device=device, dtype=torch.float32),
                indexing='ij'
            )
            y_coords = torch.full_like(x_coords, pos, dtype=torch.float32)
            voxel_coords = torch.stack([z_coords.flatten(), y_coords.flatten(), x_coords.flatten()], dim=1)
            
            # Ground truth (downsampled)
            gt_slice = target_volume[::downsample, pos, ::downsample]
            slice_shape = (render_D, render_W)
            
        else:  # axis == 'x'
            # YZ slice at x=pos - original shape (D, H)
            orig_shape = (D, H)
            downsample = max(1, max(D, H) // max_resolution)
            render_D, render_H = D // downsample, H // downsample
            
            z_coords, y_coords = torch.meshgrid(
                torch.linspace(0, D-1, render_D, device=device, dtype=torch.float32),
                torch.linspace(0, H-1, render_H, device=device, dtype=torch.float32),
                indexing='ij'
            )
            x_coords = torch.full_like(y_coords, pos, dtype=torch.float32)
            voxel_coords = torch.stack([z_coords.flatten(), y_coords.flatten(), x_coords.flatten()], dim=1)
            
            # Ground truth (downsampled)
            gt_slice = target_volume[::downsample, ::downsample, pos]
            slice_shape = (render_D, render_H)
        
        # Render from Gaussians (at reduced resolution)
        rendered_flat = query_gaussians_chunked(gaussians, voxel_coords, volume_shape, chunk_size=10000)
        rendered_slice = rendered_flat.reshape(slice_shape).cpu().numpy()
        
        # Convert ground truth to numpy if it's a tensor
        if torch.is_tensor(gt_slice):
            gt_slice = gt_slice.cpu().numpy()
        else:
            gt_slice = np.array(gt_slice)
        
        # Ensure gt_slice matches rendered_slice shape (in case of rounding)
        gt_slice = gt_slice[:slice_shape[0], :slice_shape[1]]
        
        # Normalize for visualization
        gt_normalized = (gt_slice - gt_slice.min()) / (gt_slice.max() - gt_slice.min() + 1e-8)
        rendered_normalized = (rendered_slice - rendered_slice.min()) / (rendered_slice.max() - rendered_slice.min() + 1e-8)
        
        # Compute error map
        error_map = np.abs(gt_normalized - rendered_normalized)
        
        # Create comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(gt_normalized, cmap='gray', aspect='auto')
        axes[0].set_title(f'Ground Truth ({axis}={pos})')
        axes[0].axis('off')
        
        axes[1].imshow(rendered_normalized, cmap='gray', aspect='auto')
        axes[1].set_title(f'Rendered ({axis}={pos})')
        axes[1].axis('off')
        
        im = axes[2].imshow(error_map, cmap='hot', aspect='auto', vmin=0, vmax=1)
        axes[2].set_title(f'Error Map ({axis}={pos})')
        axes[2].axis('off')
        
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        # Save
        filename = f'slice_comparison_{axis}{pos}_iter_{iteration}.png'
        plt.savefig(os.path.join(output_path, filename), dpi=150, bbox_inches='tight')
        plt.close()


def densify_and_split_tubes_microscopy(gaussians, volume_shape, voxel_spacing, iteration, 
                                        skeleton_info=None, target_volume=None,
                                        densify_warmup_iter=200):
    """Microscopy-oriented densification using physical units and importance metric.
    
    Key improvements:
    - Uses physical units (microns) aware of anisotropic voxel spacing
    - Importance = grad_ema * opacity_ema (avoids noisy gradients)
    - Anisotropy-aware splitting
    - Clone offset along skeleton tangent for connectivity preservation
    - EMA warmup: no densification until EMA has stabilized
    - GT intensity gating: only densify Gaussians near actual foreground
    
    Note: GaussianModel stores scales in canonical (x, y, z) order.
    """
    D, H, W = volume_shape
    device = gaussians.get_xyz.device
    
    # Check if we have EMA tracking
    if not hasattr(gaussians, 'grad_ema') or not hasattr(gaussians, 'opacity_ema'):
        return 0, 0
    
    # EMA warmup: skip densification until EMA has stabilized
    if not hasattr(gaussians, 'ema_initialized') or not gaussians.ema_initialized:
        return 0, 0
    
    # Compute importance metric
    importance = (gaussians.grad_ema.squeeze(1) * gaussians.opacity_ema.squeeze(1))
    
    # GT intensity gating: only densify Gaussians whose centers are near foreground
    # This prevents spawning Gaussians in background regions
    if target_volume is not None:
        xyz = gaussians.get_xyz  # (N, 3) in canonical (x, y, z) normalized
        # Convert to voxel coordinates for sampling
        voxel_coords = torch.stack([
            (xyz[:, 2] * D).clamp(0, D-1),  # z
            (xyz[:, 1] * H).clamp(0, H-1),  # y
            (xyz[:, 0] * W).clamp(0, W-1),  # x
        ], dim=1)  # (N, 3) in (z, y, x) voxel space
        
        # Sample GT intensity at Gaussian centers
        gt_intensity = trilinear_sample(target_volume, voxel_coords)
        
        # Only allow densification where GT intensity > threshold
        fg_gate = gt_intensity > 0.05  # Gaussians must be near foreground
    else:
        fg_gate = torch.ones(gaussians.get_xyz.shape[0], dtype=torch.bool, device=device)
    
    # Get scales in physical units
    # scales are in canonical (x, y, z) order
    scales = gaussians.get_scaling  # normalized [0,1] in (x, y, z)
    scales_voxels = scales * torch.tensor([W, H, D], device=device)  # (x, y, z) in voxels
    # voxel_spacing is (vz, vy, vx) so reorder for (x, y, z)
    voxel_spacing_xyz = torch.tensor([voxel_spacing[2], voxel_spacing[1], voxel_spacing[0]], device=device)
    scales_physical = scales_voxels * voxel_spacing_xyz  # in microns (x, y, z)
    
    s_max = scales_physical.max(dim=1).values
    s_min = scales_physical.min(dim=1).values
    anisotropy = s_max / (s_min + 1e-6)
    
    # Physical unit thresholds (in microns) - increased for smoother rendering
    mean_xy_spacing = (voxel_spacing[1] + voxel_spacing[2]) / 2
    s_clone = 2.5 * mean_xy_spacing  # ~2-3 voxels worth (was 1.5)
    s_split = 6.0 * mean_xy_spacing  # ~5-6 voxels worth (was 4.0)
    
    # Importance and anisotropy thresholds
    op_thr = 0.02
    g_thr = 5e-4
    I_thr = op_thr * g_thr  # ~1e-5
    r_max = 8.0  # max anisotropy ratio (was 10.0, lower = rounder Gaussians)
    
    # Densification masks - now gated by GT foreground
    important_mask = (importance > I_thr) & fg_gate
    
    # Clone: small important Gaussians
    small_mask = s_max < s_clone
    clone_mask = important_mask & small_mask
    
    # Split: large OR too anisotropic important Gaussians
    large_mask = s_max > s_split
    aniso_mask = anisotropy > r_max
    split_mask = important_mask & (large_mask | aniso_mask)
    
    n_clone = clone_mask.sum().item()
    n_split = split_mask.sum().item()
    
    if n_clone == 0 and n_split == 0:
        return 0, 0
    
    new_xyz = []
    new_opacity = []
    new_scaling = []
    new_rotation = []
    new_features_dc = []
    
    # Clone operation: duplicate Gaussian with offset along skeleton tangent
    if n_clone > 0:
        clone_xyz = gaussians._xyz[clone_mask].clone()
        
        # Compute offset direction: use skeleton tangent if available, else local principal axis
        if skeleton_info is not None:
            # Find nearest skeleton point for each Gaussian to be cloned
            skel_pos = skeleton_info['positions'].to(device)  # (M, 3)
            skel_tan = skeleton_info['tangents'].to(device)   # (M, 3)
            
            # Compute distances to skeleton points (chunked for memory efficiency)
            n_clone_pts = clone_xyz.shape[0]
            offset_dirs = torch.zeros_like(clone_xyz)
            chunk_size = 1000
            
            for i in range(0, n_clone_pts, chunk_size):
                chunk = clone_xyz[i:i+chunk_size]  # (C, 3)
                dists = torch.cdist(chunk, skel_pos)  # (C, M)
                nearest_idx = dists.argmin(dim=1)  # (C,)
                offset_dirs[i:i+chunk_size] = skel_tan[nearest_idx]  # tangent at nearest skeleton point
            
            # Add small random component to break symmetry
            offset = offset_dirs * 0.01 + torch.randn_like(clone_xyz) * 0.002
        else:
            # Fallback: use principal axis from rotation (local tangent)
            rot_matrices = quaternion_to_rotation_matrix(gaussians.get_rotation[clone_mask])
            # Principal axis is first column of rotation matrix (corresponds to largest scale)
            principal_axis = rot_matrices[:, :, 0]  # (N, 3)
            offset = principal_axis * 0.01 + torch.randn_like(clone_xyz) * 0.002
        
        clone_xyz = clone_xyz + offset
        
        new_xyz.append(clone_xyz)
        new_opacity.append(gaussians._opacity[clone_mask].clone())
        new_scaling.append(gaussians._scaling[clone_mask].clone())
        new_rotation.append(gaussians._rotation[clone_mask].clone())
        new_features_dc.append(gaussians._features_dc[clone_mask].clone())
    
    # Split operation: create two smaller Gaussians along principal axis
    if n_split > 0:
        split_xyz = gaussians._xyz[split_mask]
        split_scales = gaussians._scaling[split_mask]
        split_rotations = gaussians._rotation[split_mask]
        
        # Get principal axis (largest scale direction) in world frame
        # scales_raw is normalized [0,1] in canonical (x, y, z), convert to voxel space
        scales_raw = gaussians.get_scaling[split_mask]
        scales_voxels = scales_raw * torch.tensor([W, H, D], device=device)  # (x, y, z) in voxels
        principal_idx = scales_raw.argmax(dim=1)  # (N_split,)
        
        # Compute offset along principal axis
        rot_matrices = quaternion_to_rotation_matrix(gaussians.get_rotation[split_mask])
        
        for local_i in range(len(split_xyz)):
            axis_idx = principal_idx[local_i]
            principal_dir = rot_matrices[local_i, :, axis_idx]  # column of rotation matrix
            
            # Offset distance in voxel space = half of principal scale (in voxels)
            offset_dist_voxels = scales_voxels[local_i, axis_idx] * 0.3
            # Convert offset to normalized coordinates
            # principal_dir is in (x, y, z), normalize by (W, H, D)
            offset_normalized = principal_dir * offset_dist_voxels / torch.tensor([W, H, D], device=device, dtype=principal_dir.dtype)
            
            # Create two new Gaussians
            xyz1 = split_xyz[local_i:local_i+1] + offset_normalized.unsqueeze(0)
            xyz2 = split_xyz[local_i:local_i+1] - offset_normalized.unsqueeze(0)
            
            # Reduce scale along principal axis
            new_scale = split_scales[local_i:local_i+1].clone()
            new_scale[0, axis_idx] *= 0.6
            
            new_xyz.extend([xyz1, xyz2])
            new_opacity.extend([gaussians._opacity[split_mask][local_i:local_i+1].clone() * 0.8] * 2)
            new_scaling.extend([new_scale.clone(), new_scale.clone()])
            new_rotation.extend([split_rotations[local_i:local_i+1].clone()] * 2)
            new_features_dc.extend([gaussians._features_dc[split_mask][local_i:local_i+1].clone()] * 2)
    
    # Concatenate new Gaussians
    if len(new_xyz) > 0:
        new_xyz = torch.cat(new_xyz, dim=0)
        new_opacity = torch.cat(new_opacity, dim=0)
        new_scaling = torch.cat(new_scaling, dim=0)
        new_rotation = torch.cat(new_rotation, dim=0)
        new_features_dc = torch.cat(new_features_dc, dim=0)
        
        # Append to existing
        n_new = new_xyz.shape[0]
        gaussians._xyz = torch.nn.Parameter(torch.cat([gaussians._xyz, new_xyz], dim=0))
        gaussians._opacity = torch.nn.Parameter(torch.cat([gaussians._opacity, new_opacity], dim=0))
        gaussians._scaling = torch.nn.Parameter(torch.cat([gaussians._scaling, new_scaling], dim=0))
        gaussians._rotation = torch.nn.Parameter(torch.cat([gaussians._rotation, new_rotation], dim=0))
        gaussians._features_dc = torch.nn.Parameter(torch.cat([gaussians._features_dc, new_features_dc], dim=0))
        gaussians._features_rest = torch.nn.Parameter(
            torch.cat([gaussians._features_rest, 
                      torch.zeros((len(new_xyz), 0, 3), device=device)], dim=0)
        )
        
        # Reset gradient accumulators
        n_total = gaussians._xyz.shape[0]
        gaussians.xyz_gradient_accum = torch.zeros((n_total, 1), device=device)
        gaussians.denom = torch.zeros((n_total, 1), device=device)
        gaussians.max_radii2D = torch.zeros(n_total, device=device)
        
        # Initialize EMA trackers for new Gaussians
        if hasattr(gaussians, 'grad_ema'):
            gaussians.grad_ema = torch.cat([
                gaussians.grad_ema,
                torch.zeros((n_new, 1), device=device)
            ], dim=0)
            gaussians.opacity_ema = torch.cat([
                gaussians.opacity_ema,
                torch.zeros((n_new, 1), device=device)
            ], dim=0)
            gaussians.low_importance_counter = torch.cat([
                gaussians.low_importance_counter,
                torch.zeros(n_new, dtype=torch.int32, device=device)
            ], dim=0)
            gaussians.large_scale_counter = torch.cat([
                gaussians.large_scale_counter,
                torch.zeros(n_new, dtype=torch.int32, device=device)
            ], dim=0)
    
    return n_clone, n_split


def prune_gaussians_microscopy(gaussians, volume_shape, voxel_spacing,
                               swc_radii_stats=None):
    """Microscopy-oriented pruning with persistence tracking and data-driven thresholds.
    
    Only prunes Gaussians that have been persistently low-importance or oversized.
    Uses SWC radii statistics for data-driven scale thresholds when available.
    
    Args:
        swc_radii_stats: dict with 'mean', 'std', 'max' in pixel units (optional)
    """
    D, H, W = volume_shape
    device = gaussians.get_xyz.device
    
    # EMA warmup check: don't prune until EMA has stabilized
    if not hasattr(gaussians, 'ema_initialized') or not gaussians.ema_initialized:
        return 0
    
    # Physical unit thresholds
    mean_xy_spacing = (voxel_spacing[1] + voxel_spacing[2]) / 2
    
    # Data-driven scale threshold from SWC radii if available
    if swc_radii_stats is not None:
        # max_scale = 3 * max_radius (allow some margin)
        max_radius_physical = swc_radii_stats['max'] * mean_xy_spacing
        s_prune = max(3.0 * max_radius_physical, 15.0 * mean_xy_spacing)
    else:
        s_prune = 15.0 * mean_xy_spacing  # ~10-20 voxels worth
    
    op_prune = 0.01
    g_prune = 1e-4
    K_persistence = 10  # must be bad for K checks
    
    # Get current state
    # scales are in canonical (x, y, z) order
    scales = gaussians.get_scaling
    scales_voxels = scales * torch.tensor([W, H, D], device=device)  # (x, y, z) in voxels
    # voxel_spacing is (vz, vy, vx), reorder for (x, y, z)
    voxel_spacing_xyz = torch.tensor([voxel_spacing[2], voxel_spacing[1], voxel_spacing[0]], device=device)
    scales_physical = scales_voxels * voxel_spacing_xyz
    s_max = scales_physical.max(dim=1).values
    
    # Check conditions
    low_opacity = gaussians.opacity_ema.squeeze(1) < op_prune
    low_gradient = gaussians.grad_ema.squeeze(1) < g_prune
    low_importance = low_opacity & low_gradient
    
    oversized = s_max > s_prune
    
    # Update persistence counters
    gaussians.low_importance_counter = torch.where(
        low_importance,
        gaussians.low_importance_counter + 1,
        torch.zeros_like(gaussians.low_importance_counter)
    )
    
    gaussians.large_scale_counter = torch.where(
        oversized,
        gaussians.large_scale_counter + 1,
        torch.zeros_like(gaussians.large_scale_counter)
    )
    
    # Prune only if persistently bad
    prune_low_importance = gaussians.low_importance_counter >= K_persistence
    prune_oversized = gaussians.large_scale_counter >= K_persistence
    
    keep_mask = ~(prune_low_importance | prune_oversized)
    
    n_pruned = (~keep_mask).sum().item()
    
    if n_pruned > 0:
        gaussians._xyz = torch.nn.Parameter(gaussians._xyz[keep_mask])
        gaussians._opacity = torch.nn.Parameter(gaussians._opacity[keep_mask])
        gaussians._scaling = torch.nn.Parameter(gaussians._scaling[keep_mask])
        gaussians._rotation = torch.nn.Parameter(gaussians._rotation[keep_mask])
        gaussians._features_dc = torch.nn.Parameter(gaussians._features_dc[keep_mask])
        gaussians._features_rest = torch.nn.Parameter(gaussians._features_rest[keep_mask])
        
        # Reset accumulators
        n_total = gaussians._xyz.shape[0]
        gaussians.xyz_gradient_accum = torch.zeros((n_total, 1), device=device)
        gaussians.denom = torch.zeros((n_total, 1), device=device)
        gaussians.max_radii2D = torch.zeros(n_total, device=device)
        
        # Maintain EMA trackers
        if hasattr(gaussians, 'grad_ema'):
            gaussians.grad_ema = gaussians.grad_ema[keep_mask]
            gaussians.opacity_ema = gaussians.opacity_ema[keep_mask]
            gaussians.low_importance_counter = gaussians.low_importance_counter[keep_mask]
            gaussians.large_scale_counter = gaussians.large_scale_counter[keep_mask]
    
    return n_pruned


def train_volumetric_gaussians(
    target_volume,
    swc_path,
    output_path='output',
    num_iterations=5000,
    save_interval=500,
    learning_rate=0.005,
    densify_from_iter=500,
    densify_until_iter=4000,
    densification_interval=500,
    densify_grad_threshold=0.0005,  # Increased from 0.0001 to reduce excessive splitting
    voxel_spacing=(1.0, 0.2, 0.2),  # (z, y, x) in microns - typical microscopy anisotropic spacing
    device='cuda'
):
    """Train volumetric Gaussians with improved tubular structure preservation."""
    os.makedirs(output_path, exist_ok=True)
    
    # Load target volume
    if isinstance(target_volume, np.ndarray):
        target_volume = torch.from_numpy(target_volume).float()
    target_volume = target_volume.to(device)
    
    # Voxel spacing for physical unit awareness
    voxel_spacing = torch.tensor(voxel_spacing, device=device)  # (vz, vy, vx)
    
    # Normalize to [0, 1]
    target_volume = (target_volume - target_volume.min()) / (target_volume.max() - target_volume.min() + 1e-8)
    
    D, H, W = target_volume.shape
    volume_shape = (D, H, W)
    print(f"Target volume shape: {D} x {H} x {W}")
    
    # Load SWC
    swc_coords, swc_radii, swc_parents = load_swc(swc_path)
    
    # Sample skeleton edge points for tube-aware loss
    skeleton_points = sample_along_skeleton_edges(swc_coords, swc_parents, volume_shape, samples_per_edge=30)
    skeleton_points = skeleton_points.to(device)
    
    # Precompute skeleton tangents for rotation alignment
    skeleton_info = precompute_skeleton_tangents(swc_coords, swc_parents, volume_shape)
    
    # Initialize anisotropic Gaussians
    xyz, colors, opacities, scales, rotations = initialize_gaussians_anisotropic(
        swc_coords, swc_radii, swc_parents, volume_shape
    )
    
    # Create GaussianModel
    gaussians = GaussianModel(sh_degree=0)
    
    # Set parameters
    xyz_tensor = torch.tensor(xyz, device=device, dtype=torch.float32)
    colors_tensor = torch.tensor(colors, device=device, dtype=torch.float32)
    opacities_tensor = torch.tensor(opacities, device=device, dtype=torch.float32)
    scales_tensor = torch.tensor(scales, device=device, dtype=torch.float32)
    rotations_tensor = torch.tensor(rotations, device=device, dtype=torch.float32)
    
    gaussians._xyz = torch.nn.Parameter(xyz_tensor.requires_grad_(True))
    gaussians._features_dc = torch.nn.Parameter(colors_tensor.unsqueeze(1).requires_grad_(True))
    gaussians._features_rest = torch.nn.Parameter(torch.zeros((xyz.shape[0], 0, 3), device=device).requires_grad_(True))
    gaussians._opacity = torch.nn.Parameter(gaussians.inverse_opacity_activation(opacities_tensor).unsqueeze(1).requires_grad_(True))
    gaussians._scaling = torch.nn.Parameter(gaussians.scaling_inverse_activation(scales_tensor).requires_grad_(True))
    gaussians._rotation = torch.nn.Parameter(rotations_tensor.requires_grad_(True))
    
    # Initialize exposure
    exposure = torch.eye(3, 4, device=device)[None]
    gaussians._exposure = torch.nn.Parameter(exposure.requires_grad_(True))
    gaussians.exposure_mapping = {}
    gaussians.pretrained_exposures = None
    
    # Tracking variables
    gaussians.max_radii2D = torch.zeros((gaussians._xyz.shape[0]), device=device)
    gaussians.xyz_gradient_accum = torch.zeros((gaussians._xyz.shape[0], 1), device=device)
    gaussians.denom = torch.zeros((gaussians._xyz.shape[0], 1), device=device)
    gaussians.spatial_lr_scale = 1.0
    
    # Initialize EMA trackers for microscopy-oriented densification
    n_gaussians = gaussians._xyz.shape[0]
    # Initialize opacity_ema to current opacity (not zero) for proper warmup
    gaussians.grad_ema = torch.zeros((n_gaussians, 1), device=device)
    gaussians.opacity_ema = gaussians.get_opacity.detach().clone()  # Initialize to current opacity
    gaussians.low_importance_counter = torch.zeros(n_gaussians, dtype=torch.int32, device=device)
    gaussians.large_scale_counter = torch.zeros(n_gaussians, dtype=torch.int32, device=device)
    gaussians.ema_initialized = False  # Will be set True after warmup
    gaussians.ema_warmup_counter = 0
    
    # Compute SWC radii statistics for data-driven thresholds
    swc_radii_stats = {
        'mean': float(np.mean(swc_radii)),
        'std': float(np.std(swc_radii)),
        'max': float(np.max(swc_radii)),
        'min': float(np.min(swc_radii)),
    }
    print(f"SWC radii stats (pixels): mean={swc_radii_stats['mean']:.2f}, "
          f"std={swc_radii_stats['std']:.2f}, max={swc_radii_stats['max']:.2f}")
    
    # Optimizer with improved settings
    param_groups = [
        {'params': [gaussians._xyz], 'lr': learning_rate, 'name': 'xyz'},
        {'params': [gaussians._opacity], 'lr': learning_rate * 2, 'name': 'opacity'},
        {'params': [gaussians._scaling], 'lr': learning_rate * 0.3, 'name': 'scaling'},  # Lower LR for scales
        {'params': [gaussians._rotation], 'lr': learning_rate * 0.3, 'name': 'rotation'},
        {'params': [gaussians._features_dc], 'lr': learning_rate * 0.5, 'name': 'features_dc'},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, weight_decay=1e-5)  # AdamW with weight decay
    
    # LR scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
    
    # Foreground threshold for stratified sampling
    fg_threshold = 0.1
    fg_count = (target_volume > fg_threshold).sum().item()
    print(f"Foreground voxels: {fg_count} ({100*fg_count/(D*H*W):.2f}%)")
    
    # Training loop
    for iteration in tqdm(range(num_iterations), desc="Training"):
        optimizer.zero_grad()
        
        # === Precompute Gaussian params once per iteration ===
        query_params = precompute_gaussian_query_params(gaussians, volume_shape)
        
        # === Loss computation ===
        
        # 1. Tube-aware loss: sample along skeleton edges
        # Note: 10k samples is sufficient - more samples use more memory
        skeleton_samples = skeleton_points[torch.randperm(len(skeleton_points), device=device)[:10000]]
        rendered_skeleton = query_gaussians_chunked(gaussians, skeleton_samples, volume_shape, query_params=query_params)
        target_skeleton = trilinear_sample(target_volume, skeleton_samples)
        loss_skeleton = F.mse_loss(rendered_skeleton, target_skeleton)
        
        # 2. Stratified voxel loss with explicit foreground/background weighting
        # Use 30% foreground, 70% background to better learn to suppress background
        # Note: 15k samples is sufficient - more samples use more memory
        fg_samples, bg_samples = sample_voxels_stratified(target_volume, num_samples=15000, 
                                                          foreground_ratio=0.3, threshold=fg_threshold)
        
        # Foreground loss (reconstruct the neuron)
        if len(fg_samples) > 0:
            rendered_fg = query_gaussians_chunked(gaussians, fg_samples, volume_shape, query_params=query_params)
            target_fg = trilinear_sample(target_volume, fg_samples)
            loss_fg = F.mse_loss(rendered_fg, target_fg)
        else:
            loss_fg = torch.tensor(0.0, device=device)
        
        # Background loss (should render to ~0, weighted higher to suppress noise)
        rendered_bg = query_gaussians_chunked(gaussians, bg_samples, volume_shape, query_params=query_params)
        target_bg = trilinear_sample(target_volume, bg_samples)
        # Weight background loss higher since we want Gaussians to NOT activate there
        loss_bg = F.mse_loss(rendered_bg, target_bg) * 2.0  # 2x weight on background
        
        loss_random = loss_fg + loss_bg
        
        # 3. Anisotropy encouragement: penalize isotropic (spherical) Gaussians
        # With proper normalization, anisotropy directly affects shape without affecting mass
        # For tubular structures, we want elongated Gaussians (anisotropy > 1)
        scales = gaussians.get_scaling
        anisotropy = scales.max(dim=1).values / (scales.min(dim=1).values + 1e-6)
        # Target anisotropy: at least 2:1 ratio for tube-like shapes
        target_anisotropy = 2.0
        # Penalize Gaussians that are too spherical (anisotropy < target)
        loss_anisotropy = F.relu(target_anisotropy - anisotropy).mean()
        
        # 4. Scale regularization: prevent extreme scales AND encourage overlap
        # scales are in canonical (x, y, z) order
        scales_world = scales * torch.tensor([W, H, D], device=device)  # (x, y, z) in voxels
        min_scale = 3.0  # Minimum 3 voxels - prevents isolated dots/noise
        max_scale = 30.0  # Maximum 30 voxels - prevents blobs swallowing detail
        loss_scale_min = F.relu(min_scale - scales_world.min(dim=1).values).mean()
        loss_scale_max = F.relu(scales_world.max(dim=1).values - max_scale).mean()
        
        # 4b. Encourage sufficient scale for smooth rendering
        # Penalize very small scales that cause noise - stronger penalty
        avg_scale = scales_world.mean(dim=1)
        loss_scale_small = F.relu(5.0 - avg_scale).mean() * 2.0  # encourage avg scale >= 5 voxels
        
        # 5. Skeleton-aware rotation alignment
        # Align each Gaussian's principal axis to local skeleton tangent direction
        # This is more appropriate for tubular structures than random diversity
        loss_alignment = compute_skeleton_alignment_loss(gaussians, skeleton_info, volume_shape, device)
        
        # 6. Sparsity penalty on opacity - encourage fewer, stronger Gaussians
        # This helps reduce "fog" from many low-opacity Gaussians
        opacity = gaussians.get_opacity.squeeze(1)  # (N,)
        # L1 penalty encourages sparsity (many zeros, few large values)
        loss_sparsity = opacity.mean() * 0.1  # small weight
        
        # Total loss with curriculum: skeleton-heavy early, balanced later
        skeleton_weight = max(0.5, 1.0 - iteration / 2000)  # 1.0 -> 0.5
        random_weight = 1.0 - skeleton_weight + 0.5  # 0.5 -> 1.0
        
        loss_total = (
            skeleton_weight * loss_skeleton +
            random_weight * loss_random +
            0.02 * loss_anisotropy +  # Encourage elongated (tube-like) Gaussians
            0.05 * loss_scale_min +   # Stronger penalty for too-small scales
            0.02 * loss_scale_max +
            0.05 * loss_scale_small + # Stronger penalty for small avg scales
            0.05 * loss_alignment +   # Align Gaussian orientations to skeleton tangents
            loss_sparsity             # Sparsity penalty on opacity
        )
        
        # Backward
        loss_total.backward()
        
        # Clear GPU cache periodically to reduce memory fragmentation
        if iteration % 50 == 0:
            torch.cuda.empty_cache()
        
        # Track gradients for densification with EMA
        with torch.no_grad():
            if gaussians._xyz.grad is not None:
                grad_norm = gaussians._xyz.grad.norm(dim=1, keepdim=True)
                # Standard accumulation for basic densification
                gaussians.xyz_gradient_accum += grad_norm
                gaussians.denom += 1
                
                # EMA updates (alpha=0.1 for smooth tracking)
                alpha = 0.1
                gaussians.grad_ema = (1 - alpha) * gaussians.grad_ema + alpha * grad_norm
                current_opacity = gaussians.get_opacity
                gaussians.opacity_ema = (1 - alpha) * gaussians.opacity_ema + alpha * current_opacity
                
                # EMA warmup: mark as initialized after sufficient iterations
                ema_warmup_iters = 100  # ~10 effective samples with alpha=0.1
                gaussians.ema_warmup_counter += 1
                if not gaussians.ema_initialized and gaussians.ema_warmup_counter >= ema_warmup_iters:
                    gaussians.ema_initialized = True
                    print(f"\n[Iter {iteration}] EMA warmup complete, densification enabled")
        
        # Densification
        if iteration >= densify_from_iter and iteration < densify_until_iter:
            if iteration % densification_interval == 0:
                n_clone, n_split = densify_and_split_tubes_microscopy(
                    gaussians, volume_shape, voxel_spacing, iteration,
                    skeleton_info=skeleton_info,
                    target_volume=target_volume
                )
                n_pruned = prune_gaussians_microscopy(
                    gaussians, volume_shape, voxel_spacing,
                    swc_radii_stats=swc_radii_stats
                )
                
                if n_clone > 0 or n_split > 0 or n_pruned > 0:
                    print(f"\n[Iter {iteration}] Clone: {n_clone}, Split: {n_split}, Pruned: {n_pruned}, Total: {gaussians._xyz.shape[0]}")
                    
                    # Rebuild optimizer and scheduler after densification
                    param_groups = [
                        {'params': [gaussians._xyz], 'lr': learning_rate, 'name': 'xyz'},
                        {'params': [gaussians._opacity], 'lr': learning_rate * 2, 'name': 'opacity'},
                        {'params': [gaussians._scaling], 'lr': learning_rate * 0.5, 'name': 'scaling'},
                        {'params': [gaussians._rotation], 'lr': learning_rate * 0.5, 'name': 'rotation'},
                        {'params': [gaussians._features_dc], 'lr': learning_rate * 0.5, 'name': 'features_dc'},
                    ]
                    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, weight_decay=1e-5)
                    # Rebuild scheduler with adjusted gamma to account for remaining iterations
                    remaining_iters = num_iterations - iteration
                    adjusted_gamma = 0.9995 ** (iteration / max(remaining_iters, 1))  # Compensate for missed steps
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        
        # Logging (tqdm handles progress automatically)
        if iteration % 100 == 0:
            tqdm.write(f"[{iteration}] loss={loss_total.item():.4f} skel={loss_skeleton.item():.4f} "
                      f"rand={loss_random.item():.4f} align={loss_alignment.item():.4f} n={gaussians._xyz.shape[0]}")
        
        # Save checkpoint
        if (iteration + 1) % save_interval == 0 or iteration == num_iterations - 1:
            checkpoint = {
                'xyz': gaussians._xyz.detach().cpu(),
                'opacity': gaussians._opacity.detach().cpu(),
                'scaling': gaussians._scaling.detach().cpu(),
                'rotation': gaussians._rotation.detach().cpu(),
                'features_dc': gaussians._features_dc.detach().cpu(),
                'iteration': iteration + 1,
                'volume_shape': volume_shape,
            }
            
            checkpoint_path = os.path.join(output_path, f'model_iter_{iteration+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"\nSaved checkpoint: {checkpoint_path}")
            
            # Save MIP visualizations
            with torch.no_grad():
                
                # Save slice comparisons (5 pairs)
                slice_positions = [
                    ('z', D // 4),      # 25% through z
                    ('z', D // 2),      # 50% through z
                    ('y', H // 3),      # 33% through y
                    ('y', 2 * H // 3),  # 66% through y
                    ('x', W // 2),      # 50% through x
                ]
                render_slice_comparison(gaussians, target_volume, volume_shape, 
                                      slice_positions, output_path, iteration + 1)
            
            if iteration == num_iterations - 1:
                torch.save(checkpoint, os.path.join(output_path, 'final_model.pth'))
    
    return gaussians


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train volumetric Gaussian splats (improved)')
    parser.add_argument('--volume', type=str, required=True, help='Path to target volume (.tif)')
    parser.add_argument('--swc', type=str, required=True, help='Path to SWC skeleton file')
    parser.add_argument('--output', type=str, default='output_volume', help='Output directory')
    parser.add_argument('--iterations', type=int, default=5000, help='Number of iterations')
    parser.add_argument('--save_interval', type=int, default=500, help='Save every N iterations')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    
    args = parser.parse_args()
    
    print(f"Loading target volume from {args.volume}")
    target_volume = tiff.imread(args.volume)
    print(f"Volume shape: {target_volume.shape}")
    
    train_volumetric_gaussians(
        target_volume=target_volume,
        swc_path=args.swc,
        output_path=args.output,
        num_iterations=args.iterations,
        save_interval=args.save_interval,
        learning_rate=args.lr,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )


if __name__ == '__main__':
    main()