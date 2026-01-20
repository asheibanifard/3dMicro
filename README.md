# 3D Micro - Gaussian Splatting for Microscopy

This repository contains code for 3D Gaussian Splatting applied to microscopy volumetric data, enabling high-quality reconstruction and interactive visualization of 3D microscopy images.

## Features

- **3D Gaussian Splatting Training**: Train Gaussian representations from volumetric microscopy data
- **Multiple Rendering Modes**: MIP (Maximum Intensity Projection), alpha blending, X-ray style
- **Interactive WebGL Viewer**: Browser-based 3D visualization with real-time controls
- **Export Tools**: Convert trained models to various formats for visualization

## Project Structure

```
3Dmicro/
├── gaussian-splatting/       # Core 3DGS implementation
├── r2_gaussian/              # R2-Gaussian implementation
├── gaussian_mip_src/         # Custom MIP CUDA kernels
├── viewer/                   # Interactive WebGL viewer
│   └── index.html           # Main viewer application
├── render_real_splatting.py  # Main rendering script
├── export_for_viewer.py      # Export models for web viewer
├── visualize_gaussians.py    # Gaussian visualization tools
└── ortho_rasterizer.py       # Orthographic projection renderer
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone git@github.com:asheibanifard/3dMicro.git
cd 3dMicro

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy tifffile matplotlib

# Install diff-gaussian-rasterization
cd gaussian-splatting/submodules/diff-gaussian-rasterization
pip install .
cd ../../..

# Install simple-knn
cd gaussian-splatting/submodules/simple-knn
pip install .
cd ../../..
```

## Usage

### Training

```bash
# Train on your volumetric data
python gaussian-splatting/train.py -s /path/to/your/data
```

### Rendering

```bash
# Render with different modes
python render_real_splatting.py --model final_model.pth --mode mip
python render_real_splatting.py --model final_model.pth --mode alpha
```

### Interactive Web Viewer

1. Export your model for the web viewer:

```bash
python export_for_viewer.py final_model.pth --output viewer/
```

2. Start a local server:

```bash
cd viewer
python -m http.server 8000
```

3. Open http://localhost:8000 in your browser

### Viewer Controls

- **Drag**: Rotate the view
- **Scroll**: Zoom in/out
- **Space**: Toggle auto-rotation
- **R**: Reset view

### Viewer Settings

- **Render Mode**: MIP, Alpha Blending, Average, X-Ray
- **Brightness/Contrast**: Adjust display intensity
- **Threshold**: Filter low-intensity values
- **Color Maps**: Grayscale, Viridis, Plasma, Hot, Cool, Bone

## Exporting Models

### For Blender

```bash
python export_for_blender.py model.pth --output gaussians_blender.npz
```

### For WebGL Viewer

```bash
python export_for_viewer.py model.pth --output viewer/ --resolution 64 256 320
```

## File Formats

### Model Checkpoint (.pth)

Contains:
- `xyz`: Gaussian centers in normalized [0,1] space
- `intensity`: Logit-space opacity values
- `scaling`: Log-space scale parameters
- `rotation`: Quaternion rotations (w, x, y, z)
- `config`: Training configuration

### Volume Data (.raw)

Raw 8-bit volume data for WebGL viewer. Dimensions stored in `dims.json`.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{sheibanifard2026,
  title={3D Gaussian Splatting for Microscopy Volume Rendering},
  author={Sheibanifard, Armin},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) - Original implementation
- [R2-Gaussian](https://github.com/Ruyi-Zha/r2_gaussian) - Radiative transfer extension
