# Combining Rule-Enhanced Hammett-Inspired Product Model

This repository contains the script for the combining rule-enhanced (c) Hammett-Inspired Product model by V. Diana Rakotonirina, Marco Bragato, Stefan Heinen, and O. Anatole von Lilienfeld, described in [this paper](https://doi.org/10.48550/arXiv.2405.07747).

## Folders

### `libnew`

This folder builds upon the work of Bragato et al. ([DOI](https://doi.org/10.1039/D0SC04235H)), with the addition of the cHIP script (calc_sigmas_average function) at the end of the `core.py` file. 

*Notes: 
	1. Both the HIP and cHIP models are thoroughly described in our manuscript.
	2. The path of libnew must be updated in the `ml.py` file before use.*

### `example`

This folder contains scripts that demonstrate how to use the cHIP model and reproduce our results and figures.

*Notes: 
	1. A tutorial for the HIP library is available [here](https://doi.org/10.1039/D0SC04235H).
	2. Most scripts in this folder require you to update the path of libnew.
*

