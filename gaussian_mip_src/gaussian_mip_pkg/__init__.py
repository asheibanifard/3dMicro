import torch
from . import _C


class MIPRenderFunction(torch.autograd.Function):
    """Differentiable MIP rendering using custom CUDA kernels."""
    
    @staticmethod
    def forward(ctx, means2D, sigmas, intensities, H, W):
        """
        Forward pass: Render MIP and save context for backward.
        
        Args:
            means2D: (N, 2) Gaussian centers [x, y]
            sigmas: (N, 2) Gaussian sigmas [sx, sy]
            intensities: (N,) intensities
            H: image height
            W: image width
            
        Returns:
            (H, W) rendered image
        """
        output, max_gaussian_ids = _C.mip_render_forward(
            means2D.contiguous().float(),
            sigmas.contiguous().float(),
            intensities.contiguous().float(),
            H, W
        )
        
        # Save for backward
        ctx.save_for_backward(means2D, sigmas, intensities, max_gaussian_ids)
        ctx.H = H
        ctx.W = W
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Compute gradients.
        
        Args:
            grad_output: (H, W) gradient from loss
            
        Returns:
            Gradients for means2D, sigmas, intensities (and None for H, W)
        """
        means2D, sigmas, intensities, max_gaussian_ids = ctx.saved_tensors
        
        grad_means2D, grad_sigmas, grad_intensities = _C.mip_render_backward(
            grad_output.contiguous().float(),
            max_gaussian_ids,
            means2D,
            sigmas,
            intensities
        )
        
        return grad_means2D, grad_sigmas, grad_intensities, None, None


def mip_render(means2D: torch.Tensor, sigmas: torch.Tensor, 
               intensities: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Differentiable MIP rendering of Gaussians.
    
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
    
    return MIPRenderFunction.apply(means2D, sigmas, intensities, H, W)

