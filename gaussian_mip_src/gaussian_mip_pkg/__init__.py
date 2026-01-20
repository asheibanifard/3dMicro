import torch
from . import _C

def mip_render(means2D: torch.Tensor, sigmas: torch.Tensor, 
               intensities: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Render Gaussians using Maximum Intensity Projection (MIP).
    
    Args:
        means2D: (N, 2) Gaussian centers in pixel coordinates [x, y]
        sigmas: (N, 2) Gaussian sigmas in pixel space [sx, sy]
        intensities: (N,) Gaussian intensities
        H: Output image height
        W: Output image width
    
    Returns:
        (H, W) MIP rendered image
    """
    assert means2D.is_cuda, "means2D must be on CUDA"
    assert sigmas.is_cuda, "sigmas must be on CUDA"
    assert intensities.is_cuda, "intensities must be on CUDA"
    
    return _C.mip_render(
        means2D.contiguous().float(),
        sigmas.contiguous().float(),
        intensities.contiguous().float(),
        H, W
    )
