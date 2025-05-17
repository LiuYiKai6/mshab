import os

from setuptools import setup

__version__ = "1.0.0"

setup(
    name="localgrasp",
    version=__version__,
    description="Local Grasp",
    author="THU-VCLab",
    packages=["localgrasp"],
    python_requires=">=3.9",
    setup_requires=["setuptools>=62.3.0"],
    install_requires=[
        "open3d>=0.18.0",
        "numba",
        "cupoch",
        "scikit-image",
        "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@stable",
        f"graspnetAPI @ file://{os.path.abspath('graspnetAPI')}",
        f"pointnet2_ops @ file://{os.path.abspath('pointnet2_ops_lib')}",
    ]
)