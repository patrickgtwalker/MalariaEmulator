from setuptools import setup, find_packages
from pathlib import Path

setup(
    name="malaria_emulator",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[line.strip() for line in Path("requirements.txt").read_text().splitlines() if line.strip()],
)
