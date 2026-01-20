from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gaussian_mip',
    version='0.0.3',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='gaussian_mip_pkg._C',
            sources=['mip_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
