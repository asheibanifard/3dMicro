/*
 * CUDA kernel for Maximum Intensity Projection (MIP) rendering of Gaussians with gradients.
 * 
 * Forward pass: Compute MIP and track which Gaussian contributed max at each pixel
 * Backward pass: Backpropagate gradients through the max operation
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define BLOCK_SIZE 16

// Forward pass: Render MIP and track which Gaussian contributed the max
__global__ void mip_render_forward_kernel(
    const float* __restrict__ means2D,      // (N, 2) projected centers [x, y]
    const float* __restrict__ sigmas,       // (N, 2) projected sigmas [sx, sy]
    const float* __restrict__ intensities,  // (N,) intensity values
    float* __restrict__ output,             // (H, W) output image
    int* __restrict__ max_gaussian_ids,     // (H, W) which Gaussian gave max at each pixel
    const int N,                            // number of Gaussians
    const int H,                            // image height
    const int W                             // image width
) {
    // Pixel coordinates
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (px >= W || py >= H) return;
    
    const int pixel_idx = py * W + px;
    float max_val = 0.0f;
    int max_id = -1;
    
    const float pxf = (float)px + 0.5f;
    const float pyf = (float)py + 0.5f;
    
    // Loop over all Gaussians
    for (int i = 0; i < N; i++) {
        const float cx = means2D[i * 2 + 0];
        const float cy = means2D[i * 2 + 1];
        const float sx = sigmas[i * 2 + 0];
        const float sy = sigmas[i * 2 + 1];
        const float intensity = intensities[i];
        
        // Quick reject: if pixel is too far from Gaussian center
        const float dx = pxf - cx;
        const float dy = pyf - cy;
        
        // Skip if outside 3 sigma
        if (fabsf(dx) > 3.0f * sx || fabsf(dy) > 3.0f * sy) {
            continue;
        }
        
        // Compute Gaussian value
        const float exp_x = (dx * dx) / (2.0f * sx * sx + 1e-8f);
        const float exp_y = (dy * dy) / (2.0f * sy * sy + 1e-8f);
        const float gauss_val = intensity * expf(-(exp_x + exp_y));
        
        // MIP: take maximum and track which Gaussian
        if (gauss_val > max_val) {
            max_val = gauss_val;
            max_id = i;
        }
    }
    
    output[pixel_idx] = max_val;
    max_gaussian_ids[pixel_idx] = max_id;
}

// Backward pass: Backpropagate gradients
__global__ void mip_render_backward_kernel(
    const float* __restrict__ grad_output,      // (H, W) gradient from loss
    const int* __restrict__ max_gaussian_ids,   // (H, W) which Gaussian gave max
    const float* __restrict__ means2D,          // (N, 2) projected centers [x, y]
    const float* __restrict__ sigmas,           // (N, 2) projected sigmas [sx, sy]
    const float* __restrict__ intensities,      // (N,) intensity values
    float* __restrict__ grad_means2D,           // (N, 2) output gradients
    float* __restrict__ grad_sigmas,            // (N, 2) output gradients
    float* __restrict__ grad_intensities,       // (N,) output gradients
    const int N,
    const int H,
    const int W
) {
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (px >= W || py >= H) return;
    
    const int pixel_idx = py * W + px;
    const int max_id = max_gaussian_ids[pixel_idx];
    
    if (max_id < 0) return;  // No Gaussian contributed
    
    const float grad_out = grad_output[pixel_idx];
    const float pxf = (float)px + 0.5f;
    const float pyf = (float)py + 0.5f;
    
    const float cx = means2D[max_id * 2 + 0];
    const float cy = means2D[max_id * 2 + 1];
    const float sx = sigmas[max_id * 2 + 0];
    const float sy = sigmas[max_id * 2 + 1];
    const float intensity = intensities[max_id];
    
    const float dx = pxf - cx;
    const float dy = pyf - cy;
    
    const float sx2 = sx * sx + 1e-8f;
    const float sy2 = sy * sy + 1e-8f;
    const float exp_x = (dx * dx) / (2.0f * sx2);
    const float exp_y = (dy * dy) / (2.0f * sy2);
    const float gauss_val = intensity * expf(-(exp_x + exp_y));
    
    // Gradient w.r.t. intensity: ∂L/∂intensity = grad_out * exp(-r²)
    atomicAdd(&grad_intensities[max_id], grad_out * expf(-(exp_x + exp_y)));
    
    // Gradient w.r.t. means2D
    // ∂gauss/∂cx = intensity * exp(...) * dx / sx²
    atomicAdd(&grad_means2D[max_id * 2 + 0], grad_out * gauss_val * dx / sx2);
    atomicAdd(&grad_means2D[max_id * 2 + 1], grad_out * gauss_val * dy / sy2);
    
    // Gradient w.r.t. sigmas
    // ∂gauss/∂sx = intensity * exp(...) * dx² / sx³
    atomicAdd(&grad_sigmas[max_id * 2 + 0], grad_out * gauss_val * (dx * dx) / (sx2 * sx));
    atomicAdd(&grad_sigmas[max_id * 2 + 1], grad_out * gauss_val * (dy * dy) / (sy2 * sy));
}


// Forward pass wrapper
std::vector<torch::Tensor> mip_render_forward_cuda(
    torch::Tensor means2D,      // (N, 2)
    torch::Tensor sigmas,       // (N, 2)
    torch::Tensor intensities,  // (N,)
    int H,
    int W
) {
    const int N = means2D.size(0);
    
    auto output = torch::zeros({H, W}, means2D.options());
    auto max_gaussian_ids = torch::full({H, W}, -1, torch::TensorOptions().dtype(torch::kInt32).device(means2D.device()));
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((W + BLOCK_SIZE - 1) / BLOCK_SIZE, (H + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    mip_render_forward_kernel<<<grid, block>>>(
        means2D.data_ptr<float>(),
        sigmas.data_ptr<float>(),
        intensities.data_ptr<float>(),
        output.data_ptr<float>(),
        max_gaussian_ids.data_ptr<int>(),
        N, H, W
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return {output, max_gaussian_ids};
}

// Backward pass wrapper
std::vector<torch::Tensor> mip_render_backward_cuda(
    torch::Tensor grad_output,       // (H, W)
    torch::Tensor max_gaussian_ids,  // (H, W)
    torch::Tensor means2D,           // (N, 2)
    torch::Tensor sigmas,            // (N, 2)
    torch::Tensor intensities        // (N,)
) {
    const int N = means2D.size(0);
    const int H = grad_output.size(0);
    const int W = grad_output.size(1);
    
    auto grad_means2D = torch::zeros_like(means2D);
    auto grad_sigmas = torch::zeros_like(sigmas);
    auto grad_intensities = torch::zeros_like(intensities);
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((W + BLOCK_SIZE - 1) / BLOCK_SIZE, (H + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    mip_render_backward_kernel<<<grid, block>>>(
        grad_output.data_ptr<float>(),
        max_gaussian_ids.data_ptr<int>(),
        means2D.data_ptr<float>(),
        sigmas.data_ptr<float>(),
        intensities.data_ptr<float>(),
        grad_means2D.data_ptr<float>(),
        grad_sigmas.data_ptr<float>(),
        grad_intensities.data_ptr<float>(),
        N, H, W
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return {grad_means2D, grad_sigmas, grad_intensities};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mip_render_forward", &mip_render_forward_cuda, "MIP render forward (CUDA)");
    m.def("mip_render_backward", &mip_render_backward_cuda, "MIP render backward (CUDA)");
}
