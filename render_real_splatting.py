"""
Real 3D Gaussian Splatting Renderer

Renders volumetric Gaussian models using the official diff-gaussian-rasterization
CUDA kernel from the original 3D Gaussian Splatting paper.

Key Features:
- Tile-based parallel rasterization for high performance
- Proper front-to-back alpha blending
- Multiple rendering modes: alpha blending and MIP (Maximum Intensity Projection)
- Rotation animations and multi-view rendering

Coordinate Systems:
- Model space: Normalized [0,1] coordinates (z, y, x) order
- World space: Centered at origin, scaled by volume dimensions (x, y, z) order
- Camera space: Standard OpenGL convention (right, up, -forward)

Requirements:
- diff-gaussian-rasterization CUDA module (pip install)
- PyTorch with CUDA support
- Optional: gaussian_mip CUDA module for fast MIP rendering
"""
import sys
import torch
import numpy as np
import tifffile as tiff
import math

# The diff_gaussian_rasterization module must be pip installed from the submodules
# Do NOT add the source folder to sys.path (causes circular import issues)
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def load_model(model_path, filter_outliers=True, max_scale_threshold=0.095):
    """Load trained Gaussian model and convert to renderable format.
    
    This function loads a model checkpoint and performs necessary transformations:
    - Converts logit intensities to opacity values via sigmoid
    - Converts log scales to actual scale values via exp
    - Normalizes rotation quaternions
    - Optionally filters out outlier Gaussians with runaway scales
    
    Args:
        model_path: Path to the .pth model checkpoint file
        filter_outliers: If True, removes Gaussians with maximum scale >= threshold
                        (These are typically artifacts from training)
        max_scale_threshold: Scale threshold for filtering (default 0.095)
                           Set just below 0.1 to catch clamped values
    
    Returns:
        xyz: (N, 3) Gaussian centers in normalized [0,1] space, (z, y, x) order
        opacity: (N, 1) Gaussian opacity values in [0, 1]
        scales: (N, 3) Gaussian scale values (positive real numbers)
        rotation: (N, 4) Normalized quaternion rotations (w, x, y, z)
        config: Dictionary containing training configuration (e.g., volume shape)
    """
    checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
    
    # Extract raw parameters from checkpoint
    # Our training format stores parameters in their optimization-friendly forms:
    # - xyz: normalized [0,1] for stable gradients
    # - intensity: logit space for unbounded optimization
    # - scaling: log space to ensure positivity
    # - rotation: quaternion (w, x, y, z) for continuous SO(3) representation
    xyz = checkpoint['xyz'].cuda()  # (N, 3) normalized [0,1], (z, y, x) order
    intensity = checkpoint['intensity'].cuda()  # (N, 1) logit-space intensity
    scaling = checkpoint['scaling'].cuda()  # (N, 3) log-space scales
    rotation = checkpoint['rotation'].cuda()  # (N, 4) quaternion (w, x, y, z)
    
    # Convert intensity from logit to opacity using sigmoid activation
    # sigmoid(x) maps (-inf, inf) → (0, 1), allowing unconstrained optimization
    # This ensures opacity stays in valid [0, 1] range during training
    opacity = torch.sigmoid(intensity)  # (N, 1)
    
    # Convert scaling from log to actual scale using exponential
    # exp(x) maps (-inf, inf) → (0, inf), ensuring positive scales
    # Log-space optimization prevents negative scales and handles wide scale ranges
    scales = torch.exp(scaling)  # (N, 3)
    
    # Normalize rotation quaternions to unit length
    # Quaternions represent rotations only when ||q|| = 1
    # Normalization ensures numerical stability despite gradient updates
    rotation = torch.nn.functional.normalize(rotation, dim=-1)
    
    print(f"Loaded {xyz.shape[0]} Gaussians")
    print(f"  XYZ range: [{xyz.min().item():.3f}, {xyz.max().item():.3f}]")
    print(f"  Opacity range: [{opacity.min().item():.3f}, {opacity.max().item():.3f}]")
    print(f"  Scale range: [{scales.min().item():.4f}, {scales.max().item():.4f}]")
    
    # Filter outliers: Gaussians with runaway scales (training artifacts)
    # These typically occur when optimization pushes scales to clamping limits
    # Filtering improves visual quality by removing abnormally large Gaussians
    if filter_outliers:
        max_scale_per_gaussian = scales.max(dim=1).values  # Get largest scale dimension per Gaussian
        valid_mask = max_scale_per_gaussian < max_scale_threshold  # Boolean mask for valid Gaussians
        num_filtered = (~valid_mask).sum().item()  # Count how many will be filtered
        
        if num_filtered > 0:
            xyz = xyz[valid_mask]
            opacity = opacity[valid_mask]
            scales = scales[valid_mask]
            rotation = rotation[valid_mask]
            print(f"  Filtered {num_filtered} outlier Gaussians (max_scale >= {max_scale_threshold})")
            print(f"  Remaining: {xyz.shape[0]} Gaussians")
    
    return xyz, opacity, scales, rotation, checkpoint.get('config', {})


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    """Compute world-to-view transformation matrix.
    
    This function constructs a 4x4 homogeneous transformation matrix that maps
    world coordinates to camera/view coordinates. It follows the standard
    computer graphics convention.
    
    Args:
        R: (3, 3) rotation matrix from world to camera
        t: (3,) translation vector (camera position in world coords)
        translate: Additional translation offset (default: origin)
        scale: Uniform scale factor to apply (default: 1.0)
    
    Returns:
        (4, 4) world-to-view transformation matrix as float32
    
    Mathematical Process:
    1. Build extrinsic matrix [R^T | t]
    2. Invert to get camera-to-world C2W
    3. Apply optional translation and scaling to camera center
    4. Invert back to get final world-to-view matrix
    """
    # Build 4x4 extrinsic matrix
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()  # Rotation component
    Rt[:3, 3] = t  # Translation component
    Rt[3, 3] = 1.0  # Homogeneous coordinate

    # Get camera-to-world transformation
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    
    # Apply additional transformation to camera center
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    
    # Invert back to world-to-view
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    """Compute OpenGL-style perspective projection matrix.
    
    This creates a projection matrix that maps camera space coordinates to
    normalized device coordinates (NDC). Uses OpenGL conventions:
    - Right-handed coordinate system
    - Z-axis points into the screen (negative Z is in front)
    - NDC range: [-1, 1] for x, y, z
    
    Args:
        znear: Near clipping plane distance (must be > 0)
        zfar: Far clipping plane distance (must be > znear)
        fovX: Horizontal field of view in radians
        fovY: Vertical field of view in radians
    
    Returns:
        (4, 4) projection matrix as torch.Tensor
    
    The matrix transforms points as: [x', y', z', w'] = P @ [x, y, z, 1]
    After perspective divide: [x'/w', y'/w', z'/w'] gives NDC coordinates
    """
    # Compute half-tangents of field of view
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    # Compute frustum bounds at near plane
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    # Initialize projection matrix
    P = torch.zeros(4, 4)
    z_sign = 1.0  # OpenGL uses negative Z for forward direction

    # Fill projection matrix components
    # X scaling (maps [-left, right] to [-1, 1])
    P[0, 0] = 2.0 * znear / (right - left)
    # Y scaling (maps [bottom, top] to [-1, 1])
    P[1, 1] = 2.0 * znear / (top - bottom)
    # X offset (for asymmetric frustums)
    P[0, 2] = (right + left) / (right - left)
    # Y offset (for asymmetric frustums)
    P[1, 2] = (top + bottom) / (top - bottom)
    # Perspective divide trigger
    P[3, 2] = z_sign
    # Z mapping (non-linear depth compression)
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def create_camera(angle_deg, volume_shape, image_size=None, distance=None):
    """
    Create camera matrices for viewing the volume from a given angle.
    
    Args:
        angle_deg: Rotation angle around Y-axis in degrees
        volume_shape: (D, H, W) volume shape
        image_size: (height, width) output image size
        distance: Camera distance from center (auto-computed if None)
    
    Returns:
        world_view_transform, full_proj_transform, camera_center, tanfovx, tanfovy
    """
    D, H, W = volume_shape
    
    if image_size is None:
        img_h, img_w = H, W
    else:
        img_h, img_w = image_size
    
    # Scene bounds: compute world space extent
    # Volume is centered at origin with dimensions (W, H, D) in world coordinates
    extent = max(D, H, W)
    
    if distance is None:
        # Camera distance: far enough to see the whole volume
        # 2.0 * extent ensures complete visibility with some margin
        distance = extent * 2.0
    
    # Camera parameters
    znear = 0.1  # Near clipping plane (objects closer than this are not rendered)
    zfar = distance * 10.0  # Far clipping plane (must be >> distance to include volume)
    
    # Field of view: wide enough to see the whole volume
    # 90 degrees (π/2 radians) provides good balance between coverage and distortion
    fov = math.pi / 2  # 90 degrees
    fovx = fov  # Horizontal FOV
    fovy = fov * img_h / img_w  # Vertical FOV (adjusted for aspect ratio)
    
    # Camera rotation: orbit around Y-axis at fixed height
    # Angle=0 → camera at (0, 0, distance) looking toward origin
    angle_rad = math.radians(angle_deg)
    
    # Camera position: circle around the volume at constant radius
    # XZ plane rotation (Y is up axis)
    cam_x = distance * math.sin(angle_rad)  # Rightward component
    cam_y = 0.0  # Keep camera at volume's center height
    cam_z = distance * math.cos(angle_rad)  # Forward component
    
    # Build rotation matrix: camera looks at origin using look-at formulation
    # This creates a view matrix where the camera is oriented toward a target point
    eye = np.array([cam_x, cam_y, cam_z])  # Camera position
    target = np.array([0.0, 0.0, 0.0])  # Look-at point (volume center)
    up = np.array([0.0, 1.0, 0.0])  # Up vector (world Y-axis)
    
    # Forward direction: unit vector from camera toward target
    # Note: This is +Z in camera space (OpenGL uses -Z as forward)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    
    # Right direction: perpendicular to both forward and up
    # Uses right-hand rule: right = forward × up
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        # Degenerate case: camera looking straight up or down
        # Use alternative up vector to avoid singularity
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    # Recompute up: ensure orthogonal basis
    # up = right × forward (right-hand rule)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # Rotation matrix: maps world coordinates to camera coordinates
    # Camera space: X=right, Y=up, Z=-forward (OpenGL convention)
    R = np.array([
        [right[0], right[1], right[2]],
        [up[0], up[1], up[2]],
        [-forward[0], -forward[1], -forward[2]]
    ])
    
    # Translation: camera position in world space
    T = -R @ eye
    
    # Build 4x4 world-to-view matrix
    # This transforms points from world coordinates to camera/view coordinates
    world_view = np.eye(4, dtype=np.float32)
    world_view[:3, :3] = R  # Upper-left 3x3: rotation
    world_view[:3, 3] = T  # Upper-right column: translation
    
    # Transpose for PyTorch (it uses row-major convention)
    world_view_transform = torch.tensor(world_view).transpose(0, 1).cuda()
    
    # Projection matrix: transforms camera coordinates to clip space
    projection_matrix = getProjectionMatrix(znear, zfar, fovx, fovy).transpose(0, 1).cuda()
    
    # Full projection transform: combines world-to-view and view-to-clip
    # This single matrix takes world coordinates directly to clip space
    # Mathematically: P * V where P=projection, V=view
    full_proj_transform = world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0)).squeeze(0)
    
    # Camera center in world coordinates (needed for culling calculations)
    camera_center = torch.tensor(eye, dtype=torch.float32).cuda()
    
    # Tangent of half field-of-view (used for screen-space Gaussian projection)
    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)
    
    return world_view_transform, full_proj_transform, camera_center, tanfovx, tanfovy, img_h, img_w


def render_gaussians(xyz, opacity, scales, rotation, volume_shape, 
                     angle=0, image_size=None, bg_color=None):
    """
    Render Gaussians using the official 3D Gaussian Splatting CUDA rasterizer.
    
    This function implements the complete rendering pipeline:
    1. Coordinate transformation: normalized model space → world space
    2. Camera setup: position, view matrix, projection matrix
    3. Rasterization: tile-based parallel GPU rendering with alpha blending
    
    The renderer uses proper volumetric compositing (front-to-back) which is
    different from simple additive rendering. This is the standard 3DGS approach
    for photorealistic novel view synthesis.
    
    Coordinate Transformations:
    - Input xyz: (z_norm, y_norm, x_norm) in [0, 1]
    - World space: (x, y, z) centered at origin, scaled by volume dimensions
    - Camera space: OpenGL convention (right=+X, up=+Y, forward=-Z)
    
    Args:
        xyz: (N, 3) Gaussian centers in normalized [0,1] coordinates (z, y, x order)
        opacity: (N, 1) Gaussian opacities [0,1]
        scales: (N, 3) Gaussian scales in normalized space
        rotation: (N, 4) Gaussian rotations as quaternions
        volume_shape: (D, H, W) volume shape for coordinate conversion
        angle: Camera rotation angle in degrees
        image_size: (height, width) output image size
        bg_color: Background color (default black)
    
    Returns:
        Rendered image (C, H, W)
    """
    D, H, W = volume_shape
    device = xyz.device
    
    # Convert normalized xyz to world coordinates centered at origin
    # Input format: xyz is (z_norm, y_norm, x_norm) in [0, 1]
    # Output format: means3D is (x, y, z) centered at 0
    # 
    # Transformation steps:
    # 1. Reorder from (z, y, x) to (x, y, z)
    # 2. Subtract 0.5 to center at origin: [0,1] → [-0.5, 0.5]
    # 3. Scale by volume dimensions to get world coordinates
    means3D = torch.zeros((xyz.shape[0], 3), device=device, dtype=torch.float32)
    means3D[:, 0] = (xyz[:, 2] - 0.5) * W  # x: third column of xyz → first column of means3D
    means3D[:, 1] = (xyz[:, 1] - 0.5) * H  # y: second column of xyz → second column of means3D
    means3D[:, 2] = (xyz[:, 0] - 0.5) * D  # z: first column of xyz → third column of means3D
    
    # Convert scales to world space
    # Scales are also in normalized space and need to be scaled by volume dimensions
    # Reorder from (z_scale, y_scale, x_scale) to (x_scale, y_scale, z_scale)
    scales_world = torch.zeros_like(scales)
    scales_world[:, 0] = scales[:, 2] * W  # x scale: third column → first column
    scales_world[:, 1] = scales[:, 1] * H  # y scale: stays in middle
    scales_world[:, 2] = scales[:, 0] * D  # z scale: first column → third column
    
    # Rotation quaternion: must reorder components to match axis permutation
    # Quaternion format: (w, x, y, z) where x, y, z are the axis components
    # Since we're permuting axes from (z, y, x) → (x, y, z), we need to permute
    # the quaternion's axis components accordingly: (w, qx, qy, qz) → (w, qz, qy, qx)
    # This ensures the rotation operates on the correct axes in world space
    rotations = torch.zeros_like(rotation)
    rotations[:, 0] = rotation[:, 0]  # w component unchanged
    rotations[:, 1] = rotation[:, 3]  # x axis in world space = z axis in model space
    rotations[:, 2] = rotation[:, 2]  # y axis unchanged
    rotations[:, 3] = rotation[:, 1]  # z axis in world space = x axis in model space
    rotations = rotations.contiguous()  # Ensure memory layout is compatible with CUDA kernel
    
    print(f"Rendering at angle {angle}°")
    print(f"  Means3D range: x=[{means3D[:,0].min().item():.1f}, {means3D[:,0].max().item():.1f}]")
    print(f"                 y=[{means3D[:,1].min().item():.1f}, {means3D[:,1].max().item():.1f}]")
    print(f"                 z=[{means3D[:,2].min().item():.1f}, {means3D[:,2].max().item():.1f}]")
    print(f"  Scales range: [{scales_world.min().item():.3f}, {scales_world.max().item():.3f}]")
    
    # Create camera
    (world_view_transform, full_proj_transform, camera_center, 
     tanfovx, tanfovy, img_h, img_w) = create_camera(angle, volume_shape, image_size)
    
    print(f"  Image size: {img_h} x {img_w}")
    print(f"  Camera center: {camera_center.cpu().numpy()}")
    
    # Background color: the canvas behind all Gaussians
    # Black (default) for dark-field rendering, white for visibility testing
    if bg_color is None:
        bg_color = torch.zeros(3, device=device)  # RGB black
    else:
        bg_color = torch.tensor(bg_color, device=device, dtype=torch.float32)
    
    # For grayscale/intensity rendering, use opacity as the "color"
    # The rasterizer expects RGB, so replicate the single intensity channel
    # Result: bright Gaussians have high opacity, dark Gaussians have low opacity
    colors_precomp = opacity.repeat(1, 3)  # (N, 1) → (N, 3): [α, α, α]
    
    # Screenspace points: 2D projections of Gaussian centers
    # Initialized to zeros; the rasterizer will compute actual projections
    # Set requires_grad=False since we're doing inference only
    means2D = torch.zeros_like(means3D, requires_grad=False)
    
    # Rasterization settings: configures the CUDA rendering kernel
    # These settings control how Gaussians are projected, culled, and blended
    raster_settings = GaussianRasterizationSettings(
        image_height=int(img_h),  # Output image height in pixels
        image_width=int(img_w),  # Output image width in pixels
        tanfovx=tanfovx,  # tan(fov_x / 2) for screen-space projection
        tanfovy=tanfovy,  # tan(fov_y / 2) for screen-space projection
        bg=bg_color,  # Background color (RGB)
        scale_modifier=1.0,  # Multiplier for all Gaussian scales (1.0 = no change)
        viewmatrix=world_view_transform,  # 4x4 world-to-camera transformation
        projmatrix=full_proj_transform,  # 4x4 combined view-projection matrix
        sh_degree=0,  # Spherical harmonics degree (0 = use precomputed colors)
        campos=camera_center,  # Camera position in world coordinates (for culling)
        prefiltered=False,  # Whether Gaussians are already sorted by depth
        debug=False,  # Enable debug outputs (slower)
        antialiasing=False  # Analytical antialiasing (experimental)
    )
    
    # Create rasterizer: the CUDA kernel wrapper
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Render: execute tile-based rasterization on GPU
    # The rasterizer performs:
    # 1. Frustum culling: discard Gaussians outside view
    # 2. Screen-space projection: compute 2D Gaussian parameters
    # 3. Tile assignment: assign Gaussians to image tiles (16x16 pixels)
    # 4. Per-pixel compositing: front-to-back alpha blending
    with torch.no_grad():  # No gradients needed for inference
        rendered_image, radii, _ = rasterizer(
            means3D=means3D,  # Gaussian centers in world space (N, 3)
            means2D=means2D,  # 2D projections (computed internally if zero)
            opacities=opacity,  # Gaussian opacities (N, 1)
            shs=None,  # Spherical harmonics (not used, using colors_precomp)
            colors_precomp=colors_precomp,  # Precomputed RGB colors (N, 3)
            scales=scales_world,  # Gaussian scales in world space (N, 3)
            rotations=rotations,  # Rotation quaternions (N, 4)
            cov3D_precomp=None  # Precomputed 3D covariance (None = compute from scales/rotations)
        )
    
    print(f"  Rendered image range: [{rendered_image.min().item():.4f}, {rendered_image.max().item():.4f}]")
    # Check rendering results
    # radii: (N,) array where radii[i] > 0 indicates Gaussian i contributed to the image
    # radii[i] = 0 means the Gaussian was culled (outside frustum or too small)
    print(f"  Visible Gaussians: {(radii > 0).sum().item()} / {radii.shape[0]}")
    
    # rendered_image: (3, H, W) RGB image tensor in [0, 1] range
    return rendered_image


def render_mip_cuda(xyz, opacity, scales, rotation, volume_shape, 
                    angle=0, image_size=None):
    """
    Render Gaussians using Maximum Intensity Projection (MIP) with custom CUDA kernel.
    
    MIP is a visualization technique commonly used in medical imaging and microscopy.
    For each pixel, it projects the maximum intensity value along the viewing ray.
    Unlike alpha blending, MIP does not perform occlusion or depth-based compositing.
    
    Algorithm:
    1. Transform Gaussians to world coordinates
    2. Rotate around Y-axis by specified angle
    3. Project to 2D using orthographic projection
    4. For each pixel, compute maximum Gaussian contribution
    
    This implementation is significantly faster than naive Python loops due to:
    - Parallel GPU evaluation of all Gaussian-pixel pairs
    - Optimized memory access patterns
    - Reduced host-device transfers
    
    Args:
        xyz: (N, 3) Gaussian centers in normalized [0,1] coordinates (z, y, x order)
        opacity: (N, 1) Gaussian opacities/intensities [0,1]
        scales: (N, 3) Gaussian scales in normalized space
        rotation: (N, 4) Gaussian rotations as quaternions
        volume_shape: (D, H, W) volume shape for coordinate conversion
        angle: Camera rotation angle in degrees
        image_size: (height, width) output image size
    
    Returns:
        MIP rendered image (H, W)
    """
    from gaussian_mip import mip_render  # Import custom CUDA MIP kernel
    
    D, H, W = volume_shape
    device = xyz.device
    
    if image_size is None:
        img_h, img_w = H, W  # Default to volume dimensions
    else:
        img_h, img_w = image_size  # Custom output size
    
    # Convert normalized xyz to world coordinates centered at origin
    # Same transformation as in render_gaussians()
    z_world = (xyz[:, 0] - 0.5) * D  # z: depth dimension
    y_world = (xyz[:, 1] - 0.5) * H  # y: height dimension  
    x_world = (xyz[:, 2] - 0.5) * W  # x: width dimension
    
    # Apply rotation around y-axis (vertical axis)
    # MIP uses simple rotation rather than full camera matrix
    # This creates a circular orbit around the volume's Y-axis
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # 2D rotation matrix applied to (x, z) coordinates
    # [x']   [cos  -sin] [x]
    # [z'] = [sin   cos] [z]
    x_rot = x_world * cos_a - z_world * sin_a
    y_rot = y_world  # Y coordinate unchanged
    
    # Convert scales to world space (same as render_gaussians)
    scale_z = scales[:, 0] * D  # z-axis scale
    scale_y = scales[:, 1] * H  # y-axis scale
    scale_x = scales[:, 2] * W  # x-axis scale
    
    # Build 3D covariance matrices for proper projection
    # This accounts for both the Gaussian's rotation and the viewing angle
    # For a rotated 3D Gaussian: Σ = R * S * S^T * R^T
    # where R is rotation matrix from quaternion, S is diagonal scale matrix
    
    # Convert quaternion to rotation matrix
    # Quaternion format: (w, x, y, z) - but need to reorder axes like in render_gaussians
    # Since model space is (z, y, x) and we're working in (x, y, z)
    qw = rotation[:, 0]
    qx = rotation[:, 3]  # x in world = z in model
    qy = rotation[:, 2]  # y unchanged
    qz = rotation[:, 1]  # z in world = x in model
    
    # Rotation matrix from quaternion (standard formula)
    # R = [[1-2(y²+z²),  2(xy-wz),    2(xz+wy)  ],
    #      [2(xy+wz),    1-2(x²+z²),  2(yz-wx)  ],
    #      [2(xz-wy),    2(yz+wx),    1-2(x²+y²)]]
    R = torch.zeros((rotation.shape[0], 3, 3), device=device)
    R[:, 0, 0] = 1 - 2 * (qy**2 + qz**2)
    R[:, 0, 1] = 2 * (qx * qy - qw * qz)
    R[:, 0, 2] = 2 * (qx * qz + qw * qy)
    R[:, 1, 0] = 2 * (qx * qy + qw * qz)
    R[:, 1, 1] = 1 - 2 * (qx**2 + qz**2)
    R[:, 1, 2] = 2 * (qy * qz - qw * qx)
    R[:, 2, 0] = 2 * (qx * qz - qw * qy)
    R[:, 2, 1] = 2 * (qy * qz + qw * qx)
    R[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)
    
    # Build scale matrix (diagonal)
    S = torch.zeros((rotation.shape[0], 3, 3), device=device)
    S[:, 0, 0] = scale_x
    S[:, 1, 1] = scale_y
    S[:, 2, 2] = scale_z
    
    # 3D Covariance: Σ = R * S * S^T * R^T = R * S^2 * R^T
    # More efficient: Σ = (R * S) * (R * S)^T
    RS = torch.bmm(R, S)  # (N, 3, 3)
    Cov3D = torch.bmm(RS, RS.transpose(1, 2))  # (N, 3, 3)
    
    # Apply viewing angle rotation (Y-axis rotation)
    # Rotation matrix for angle around Y-axis
    R_view = torch.eye(3, device=device).unsqueeze(0).repeat(rotation.shape[0], 1, 1)
    R_view[:, 0, 0] = cos_a
    R_view[:, 0, 2] = -sin_a
    R_view[:, 2, 0] = sin_a
    R_view[:, 2, 2] = cos_a
    
    # Rotate covariance: Σ' = R_view * Σ * R_view^T
    Cov3D_rot = torch.bmm(torch.bmm(R_view, Cov3D), R_view.transpose(1, 2))
    
    # Project to 2D by marginalizing out z (depth) dimension
    # 2D covariance is the top-left 2x2 block of the 3D covariance
    Cov2D = Cov3D_rot[:, :2, :2]  # (N, 2, 2)
    
    # Extract standard deviations from 2D covariance
    # For visualization, we approximate with axis-aligned ellipse using eigenvalues
    # σ_x ≈ sqrt(Σ_xx), σ_y ≈ sqrt(Σ_yy)
    # This is an approximation but works well for MIP rendering
    sigma_x = torch.sqrt(torch.clamp(Cov2D[:, 0, 0], min=1e-6))
    sigma_y = torch.sqrt(torch.clamp(Cov2D[:, 1, 1], min=1e-6))
    
    # Orthographic projection: map world coordinates to pixel coordinates
    # MIP uses orthographic (parallel) projection, not perspective
    # This preserves relative sizes regardless of depth
    # Transform: world [-W/2, W/2] → normalized [0, 1] → pixels [0, img_w]
    px = (x_rot / W + 0.5) * img_w  # X pixel coordinate
    py = (y_rot / H + 0.5) * img_h  # Y pixel coordinate
    
    # Scale sigmas to pixel space
    # World-space scales need to be converted to pixel-space for rendering
    sigma_x_px = sigma_x * img_w / W  # Pixels per world unit
    sigma_y_px = sigma_y * img_h / H  # Pixels per world unit
    
    # Clamp sigmas to reasonable range
    # Too small (<0.5 pixels): invisible, numerical instability
    # Too large (>100 pixels): performance issues, likely outliers
    sigma_x_px = torch.clamp(sigma_x_px, min=0.5, max=100.0)
    sigma_y_px = torch.clamp(sigma_y_px, min=0.5, max=100.0)
    
    # Prepare inputs for CUDA kernel
    # The MIP kernel expects specific tensor formats for optimal GPU performance
    means2D = torch.stack([px, py], dim=1).contiguous()  # (N, 2) pixel positions
    sigmas = torch.stack([sigma_x_px, sigma_y_px], dim=1).contiguous()  # (N, 2) pixel-space scales
    intensities = opacity.squeeze().contiguous()  # (N,) intensity values
    
    print(f"MIP CUDA Rendering at angle {angle}°")
    print(f"  Image size: {img_h} x {img_w}")
    print(f"  Projected x range: [{px.min().item():.1f}, {px.max().item():.1f}] pixels")
    print(f"  Projected y range: [{py.min().item():.1f}, {py.max().item():.1f}] pixels")
    
    # Call CUDA kernel for parallel MIP computation
    # Kernel iterates over all pixels and finds maximum Gaussian contribution
    # Much faster than Python loops due to GPU parallelization
    image = mip_render(means2D, sigmas, intensities, img_h, img_w)
    
    print(f"  Output intensity range: [{image.min().item():.4f}, {image.max().item():.4f}]")
    
    # Return grayscale image (H, W)
    return image


def render_multiple_views(xyz, opacity, scales, rotation, volume_shape,
                          angles=[0, 45, 90, 135, 180], image_size=None, mode='alpha'):
    """
    Render multiple views from different angles.
    
    Useful for creating comparison images or analyzing the volume from multiple perspectives.
    
    Args:
        xyz, opacity, scales, rotation: Gaussian parameters
        volume_shape: (D, H, W) volume dimensions
        angles: List of rotation angles in degrees
        image_size: Output size (default: volume dimensions)
        mode: 'alpha' for 3DGS rendering or 'mip' for Maximum Intensity Projection
    
    Returns:
        List of rendered images, one per angle
    """
    views = []
    for angle in angles:
        print(f"\n--- Rendering view at {angle}° ---")
        if mode == 'mip':
            # MIP mode: returns grayscale (H, W)
            image = render_mip_cuda(xyz, opacity, scales, rotation, volume_shape,
                                    angle=angle, image_size=image_size)
            views.append(image.cpu().numpy())
        else:
            # Alpha mode: returns RGB (3, H, W), extract first channel
            image = render_gaussians(xyz, opacity, scales, rotation, volume_shape,
                                     angle=angle, image_size=image_size)
            views.append(image[0].cpu().numpy())  # Take first channel
    return views, angles


def create_rotation_gif(xyz, opacity, scales, rotation, volume_shape,
                        num_frames=30, output_path='rotation.gif', fps=10,
                        image_size=None, mode='alpha'):
    """
    Create a GIF animation rotating 360° around the volume.
    
    Perfect for visualizing 3D structure and verifying rendering quality.
    
    Args:
        xyz, opacity, scales, rotation: Gaussian parameters
        volume_shape: (D, H, W) volume dimensions
        num_frames: Number of frames in the animation
        output_path: Path to save the GIF file
        fps: Frames per second (playback speed)
        image_size: Output size (default: volume dimensions)
        mode: 'alpha' for 3DGS rendering or 'mip' for Maximum Intensity Projection
    
    Returns:
        None (saves GIF to disk)
    """
    import imageio
    
    frames = []
    angles = np.linspace(0, 360, num_frames, endpoint=False)  # Evenly spaced angles
    
    for i, angle in enumerate(angles):
        print(f"\n--- Frame {i+1}/{num_frames} (angle={angle:.1f}°) ---")
        if mode == 'mip':
            # MIP rendering: returns (H, W) grayscale
            image = render_mip_cuda(xyz, opacity, scales, rotation, volume_shape,
                                    angle=angle, image_size=image_size)
            img_np = image.cpu().numpy()
        else:
            # Alpha rendering: returns (3, H, W) RGB, take first channel
            image = render_gaussians(xyz, opacity, scales, rotation, volume_shape,
                                     angle=angle, image_size=image_size)
            img_np = image[0].cpu().numpy()  # Extract grayscale channel
        
        # Convert to uint8 grayscale for GIF encoding
        # Clamp to [0, 1] then scale to [0, 255]
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        frames.append(img_np)
    
    # Save GIF: loop=0 means infinite loop
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"\nSaved GIF to {output_path} ({num_frames} frames at {fps} fps)")
    
    return frames


def main():
    """
    Command-line interface for rendering Gaussian splats.
    
    This script supports three main modes:
    1. Single view rendering: Render from one angle
    2. Multi-view rendering: Render from multiple predefined angles
    3. Rotation animation: Create a GIF rotating 360° around the volume
    
    Examples:
        # Single view at 45 degrees using alpha blending
        python render_real_splatting.py --angle 45 --mode alpha
        
        # Create rotation GIF using MIP mode
        python render_real_splatting.py --gif --mode mip --fps 15
        
        # Render multiple views
        python render_real_splatting.py --multiview --mode alpha
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Render 3D Gaussians using official 3DGS rasterizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Rendering Modes:
  alpha  - Standard 3D Gaussian Splatting with alpha blending
           Best for: View synthesis, photorealistic rendering
           
  mip    - Maximum Intensity Projection
           Best for: Medical/microscopy visualization, fluorescence data

Examples:
  # Single view from 45 degrees
  python render_real_splatting.py --angle 45 --output render_45.tif
  
  # Create 360° rotation animation
  python render_real_splatting.py --gif --num_frames 60 --fps 30
  
  # MIP rendering with custom image size
  python render_real_splatting.py --mode mip --image_size 512 512
""")
    parser.add_argument('--model_path', type=str, default='final_model.pth',
                        help='Path to trained model')
    parser.add_argument('--output', type=str, default='render_3dgs.tif',
                        help='Output file path')
    parser.add_argument('--angle', type=float, default=0,
                        help='Camera angle in degrees')
    parser.add_argument('--gif', action='store_true',
                        help='Create rotation GIF')
    parser.add_argument('--num_frames', type=int, default=30,
                        help='Number of frames for GIF')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for GIF')
    parser.add_argument('--multiview', action='store_true',
                        help='Render multiple views')
    parser.add_argument('--image_size', type=int, nargs=2, default=None,
                        help='Output image size (height width)')
    parser.add_argument('--mode', type=str, default='alpha', choices=['alpha', 'mip'],
                        help='Rendering mode: alpha (3DGS blending) or mip (Maximum Intensity Projection)')
    
    args = parser.parse_args()
    
    # Load model from checkpoint
    # This converts stored parameters to renderable format
    xyz, opacity, scales, rotation, config = load_model(args.model_path)
    
    # Get volume shape from training config
    # This tells us the original 3D data dimensions
    if 'img_size' in config:
        volume_shape = tuple(config['img_size'])  # (D, H, W)
    else:
        volume_shape = (100, 650, 820)  # Default fallback
    
    print(f"Volume shape: {volume_shape} (D, H, W)")
    print(f"Rendering mode: {args.mode}")
    
    # Parse image size if provided
    image_size = tuple(args.image_size) if args.image_size else None
    
    if args.gif:
        # Create 360° rotation animation
        # Saves animated GIF showing volume from all angles
        create_rotation_gif(xyz, opacity, scales, rotation, volume_shape,
                            num_frames=args.num_frames, output_path=args.output,
                            fps=args.fps, image_size=image_size, mode=args.mode)
    
    elif args.multiview:
        # Render multiple views at predefined angles
        # Useful for comparing different perspectives side-by-side
        views, angles = render_multiple_views(xyz, opacity, scales, rotation, 
                                              volume_shape, image_size=image_size,
                                              mode=args.mode)
        for view, angle in zip(views, angles):
            # Save each view with angle in filename
            out_path = args.output.replace('.tif', f'_angle{int(angle)}.tif')
            # Convert [0,1] float to 16-bit integer for TIFF
            tiff.imwrite(out_path, (view * 65535).astype(np.uint16))
            print(f"Saved {out_path}")
    
    else:
        # Single view at specified angle
        # Default mode for quick rendering and testing
        if args.mode == 'mip':
            # MIP: returns (H, W) grayscale
            image = render_mip_cuda(xyz, opacity, scales, rotation, volume_shape,
                                    angle=args.angle, image_size=image_size)
            img_np = image.cpu().numpy()
        else:
            # Alpha blending: returns (3, H, W) RGB, extract first channel
            image = render_gaussians(xyz, opacity, scales, rotation, volume_shape,
                                     angle=args.angle, image_size=image_size)
            img_np = image[0].cpu().numpy()  # Grayscale channel
        
        # Save as 16-bit TIFF (preserves dynamic range better than 8-bit)
        tiff.imwrite(args.output, (img_np * 65535).astype(np.uint16))
        print(f"Saved {args.output}")


if __name__ == '__main__':
    main()
