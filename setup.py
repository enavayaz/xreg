from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="xreg",
    version="0.1.0",
    author="Esfandiar Nava-Yazdani",
    description="Manifold Regression Framework for time-series prediction on Riemannian manifolds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/enavayaz/xreg",
    packages=find_packages(include=["timeseries", "timeseries.*", "helpers", "helpers.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "jupyter>=1.0",
            "notebook>=6.5",
        ],
    },
)
