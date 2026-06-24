import sys
import os
from setuptools import setup, Extension
import pybind11

arix_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

core_dir = os.path.join(arix_root, 'src', 'core')
arch_dir = os.path.join(arix_root, 'src', 'arch')
security_dir = os.path.join(arix_root, 'src', 'security')

include_dirs = [
    pybind11.get_include(),
    os.path.join(core_dir, 'include'),
    os.path.join(arch_dir, 'include'),
    os.path.join(security_dir, 'c', 'include'),
    os.path.join(security_dir, 'cpp', 'include'),
]

ext_modules = [
    Extension(
        'arix_algo.arix_algo_core',
        ['bindings.cpp'],
        include_dirs=include_dirs,
        library_dirs=[
            os.path.join(arix_root, 'build', 'Release'),
        ],
        libraries=['arix_arch', 'arix_core'],
        language='c++',
        extra_compile_args=['/std:c++20' if sys.platform == 'win32' else '-std=c++20'],
    ),
]

setup(
    name='arix-algo',
    version='0.1.0',
    packages=['arix_algo'],
    package_dir={'arix_algo': 'arix_algo'},
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.10', 'numpy'],
)
