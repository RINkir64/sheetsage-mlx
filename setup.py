from setuptools import setup, find_packages

setup(
    name="sheetsage-mlx",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mlx>=0.20.0",
        "numpy>=1.20.0",
        "scipy>=1.9.0",
        "safetensors>=0.4.0",
        "mido>=1.2.10",
        "mir_eval>=0.8.0",
        "huggingface_hub>=0.20.0",
    ],
    entry_points={
        "console_scripts": [
            "sheetsage-mlx=sheetsage_mlx.cli:main",
        ],
    },
)
