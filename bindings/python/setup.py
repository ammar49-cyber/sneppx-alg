"""Setup for SneppX-ALG Python package."""
from setuptools import setup, find_packages
import os

# Check if pre-built C extension exists
pkg_dir = os.path.join(os.path.dirname(__file__), "SneppX_ALG")
has_c_ext = any(
    f.startswith(("_SNEPPX_c", "_sneppx_c", "_arix_c")) and f.endswith((".pyd", ".so"))
    for f in os.listdir(pkg_dir)
)

setup(
    name="sneppx-alg",
    version="0.9.5.748",
    description="Next-generation AI architecture with security built into the foundation",
    long_description=open(os.path.join(os.path.dirname(__file__), "..", "..", "README.md"), encoding="utf-8").read() if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "..", "README.md")) else "",
    long_description_content_type="text/markdown",
    author="Ammar [SNEPPX]",
    author_email="algoSNEPPX@gmail.com",
    url="https://github.com/ammar49-cyber/sneppx-alg",
    project_urls={
        "Source": "https://github.com/ammar49-cyber/sneppx-alg",
    },
    packages=find_packages(),
    package_data={
        "SneppX_ALG": [
            "_SNEPPX_c*.pyd", "_SNEPPX_c*.so",
            "_sneppx_c*.pyd", "_sneppx_c*.so",
            "_arix_c*.pyd", "_arix_c*.so",
            "interface_bindings/*.py",
        ],
    },
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "pyyaml>=5.1",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "scipy>=1.7"],
        "hf": ["huggingface_hub>=0.16.0"],
        "serve": ["uvicorn>=0.22.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="ai, machine-learning, neural-networks, security, post-quantum, cryptography",
    entry_points={
        "console_scripts": [
            "sneppx-train=SneppX_ALG.interface_bindings.train_cli:main",
            "sneppx-serve=SneppX_ALG.interface_bindings.serve_cli:main",
            "sneppx-experiment=SneppX_ALG.interface_bindings.experiment_cli:main",
        ],
    },
    zip_safe=False,
)
