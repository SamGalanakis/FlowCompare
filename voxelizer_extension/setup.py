from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='voxelizer_cpp',
      ext_modules=[cpp_extension.CppExtension('voxelizer_cpp', ['voxelizer.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

