/*
 * CUDA kernel for Maximum Intensity Projection (MIP) rendering of Gaussians.
 * 
 * Each thread processes one pixel and computes the maximum intensity
 * from all Gaussians projected onto that pixel.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define BLOCK_SIZE 16

__global__ void mip_render_kernel(
    const float* __restrict__ means2D,      // (N, 2) projected centers [x, y]
    const float* __restrict__ sigmas,       // (N, 2) projected sigmas [sx, sy]
    const float* __restrict__ intensities,  // (N,) intensity values
    float* __restrict__ output,             // (H, W) output image
    const int N,                            // number of Gaussians
    const int H,                            // image height
    const int W                             // image width
) {
    // Pixel coordinates
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (px >= W || py >= H) return;
    
    float max_val = 0.0f;
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
        
        // MIP: take maximum
        max_val = fmaxf(max_val, gauss_val);
    }
    
    output[py * W + px] = max_val;
}


// Tiled version for better cache utilization
// Groups nearby Gaussians and processes them together
__global__ void mip_render_tiled_kernel(
    const float* __restrict__ means2D,      // (N, 2) projected centers [x, y]
    const float* __restrict__ sigmas,       // (N, 2) projected sigmas [sx, sy]
    const float* __restrict__ intensities,  // (N,) intensity values
    const int* __restrict__ gaussian_ids,   // sorted Gaussian IDs by tile
    const int* __restrict__ tile_ranges,    // (num_tiles, 2) start/end for each tile
    float* __restrict__ output,             // (H, W) output image
    const int N,                            // number of Gaussians
    const int H,                            // image height
    const int W,                            // image width
    const int tiles_x,                      // number of tiles in x
    const int tiles_y                       // number of tiles in y
) {
    // Pixel coordinates
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (px >= W || py >= H) return;
    
    // Tile coordinates
    const int tile_x = px / BLOCK_SIZE;
    const int tile_y = py / BLOCK_SIZE;
    const int tile_id = tile_y * tiles_x + tile_x;
    
    // Get range of Gaussians for this tile
    const int start = tile_ranges[tile_id * 2];
    const int end = tile_ranges[tile_id * 2 + 1];
    
    float max_val = 0.0f;
    const float pxf = (float)px + 0.5f;
    const float pyf = (float)py + 0.5f;
    
    // Loop over Gaussians in this tile
    for (int idx = start; idx < end; idx++) {
        const int i = gaussian_ids[idx];
        
        const float cx = means2D[i * 2 + 0];
        const float cy = means2D[i * 2 + 1];
        const float sx = sigmas[i * 2 + 0];
        const float sy = sigmas[i * 2 + 1];
        const float intensity = intensities[i];
        
        const float dx = pxf - cx;
        const float dy = pyf - cy;
        
        // Compute Gaussian value
        const float exp_x = (dx * dx) / (2.0f * sx * sx + 1e-8f);
        const float exp_y = (dy * dy) / (2.0f * sy * sy + 1e-8f);
        const float gauss_val = intensity * expf(-(exp_x + exp_y));
        
        max_val = fmaxf(max_val, gauss_val);
    }
    
    output[py * W + px] = max_val;
}


torch::Tensor mip_render_cuda(
    torch::Tensor means2D,      // (N, 2)
    torch::Tensor sigmas,       // (N, 2)
    torch::Tensor intensities,  // (N,)
    int H,
    int W
) {
    const int N = means2D.size(0);
    
    // Allocate output
    auto output = torch::zeros({H, W}, means2D.options());
    
    // Launch kernel
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((W + BLOCK_SIZE - 1) / BLOCK_SIZE, (H + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    mip_render_kernel<<<grid, block>>>(
        means2D.data_ptr<float>(),
        sigmas.data_ptr<float>(),
        intensities.data_ptr<float>(),
        output.data_ptr<float>(),
        N, H, W
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mip_render", &mip_render_cuda, "MIP render Gaussians (CUDA)");
}
