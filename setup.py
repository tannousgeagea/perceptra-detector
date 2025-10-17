"""
Setup configuration for perceptra-detector.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read version
version_file = Path(__file__).parent / "perceptra_detector" / "__version__.py"
version = {}
exec(version_file.read_text(), version)

setup(
    name="perceptra-detector",
    version=version['__version__'],
    author=version['__author__'],
    description="Production-ready object detection and segmentation framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tannousgeagea/perceptra-detector",
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "Pillow>=9.0.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "opencv-python>=4.5.0",
        "pydantic>=2.0.0",
        "click>=8.0.0",
        "pyyaml>=6.0",
        "requests>=2.27.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "yolo": [
            "ultralytics>=8.0.0",
        ],
        "detr": [
            "transformers>=4.25.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.20.0",
            "python-multipart>=0.0.5",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "isort>=5.10.0",
        ],
        "all": [
            "ultralytics>=8.0.0",
            "transformers>=4.25.0",
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.20.0",
            "python-multipart>=0.0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "perceptra-detector=perceptra_detector.cli.commands:cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)