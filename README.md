# xreg: Manifold Regression Framework

This repository contains the implementation for the manifold regression framework 
presented in our paper. In particular, it can be used to run the experiments (forecasting synthetical spherical trajectories) 
from the paper.

> Esfandiar Nava-Yazdani:  
> **[Ridge Regression on Riemannian Manifolds for Time-Series 
Prediction](https://arxiv.org/abs/2411.18339.pdf)**  
> Journal of Information Geometry, 2026.</br>
<!--- >[![Preprint](https://)](https://opus4.kobv.de/opus4-zib/citationExport/index/download/output/bibtex/) --->

## Colab Notebook
You can run the synthetic validation tests directly in your browser (ensure you clone the repository and add 
the root directory to your system path to enable the modules):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/enavayaz/xreg/blob/main/notebooks/synthetic_test.ipynb)

Note: If you encounter import errors, try Runtime → Disconnect and delete runtime, then re-run cells to get a fresh environment.

## Binder
You can also launch the notebook in a fully reproducible environment via Binder (no setup required):
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/enavayaz/xfreg/HEAD?labpath=notebooks%2Fsynthetic_test.ipynb)

## Local Installation
To run the notebooks, ensure you have the dependencies installed:
```bash
pip install -r requirements.txt
```
