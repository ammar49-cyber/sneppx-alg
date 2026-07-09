"""Setup for SneppX-ALG Python package."""
from setuptools import setup, find_packages

setup(
    name="SneppX-ALG",
    version="0.7.8",
    packages=find_packages(),
    package_data={"SneppX_ALG": ["_SNEPPX_c*.pyd", "_SNEPPX_c*.so", "_sneppx_c*.pyd", "_sneppx_c*.so"]},
    include_package_data=True,
    zip_safe=False,
)
