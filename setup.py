"""
Setup script for RLHF Phi-3 Pipeline
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "RLHF Phi-3 Pipeline - A production-grade RLHF pipeline for Microsoft Phi-3"

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
        return requirements

setup(
    name="rlhf-phi3-pipeline",
    version="0.1.0",
    author="RLHF Pipeline Team",
    author_email="contact@rlhf-pipeline.com",
    description="A production-grade RLHF pipeline for Microsoft Phi-3 Mini",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/rlhf-phi3-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "hypothesis>=6.88.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "pre-commit>=3.4.0",
        ],
        "colab": [
            "google-colab",
            "google-auth>=2.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rlhf-phi3=rlhf_phi3.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "rlhf_phi3": ["configs/*.yaml"],
    },
)