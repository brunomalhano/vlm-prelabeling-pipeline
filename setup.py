from setuptools import setup, find_packages

setup(
    name="vlm-pipeline",
    version="0.1.0",
    description="VLM pre-labeling pipeline: PT/EN prompt evaluation with Grounding DINO + SAM on COCO",
    author="Bruno Malhano",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "groundingdino-py>=0.4.0",
        "sam2>=0.4.0",
        "pycocotools>=2.0.7",
        "scipy>=1.11.0",
        "scikit-image>=0.21.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "viz": ["fiftyone>=0.23.0"],
        "dev": ["pytest>=7.4.0"],
    },
    entry_points={
        "console_scripts": [
            "vlm-pipeline=vlm_pipeline.cli:main",
        ],
    },
)
