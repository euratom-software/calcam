![Calcam logo](https://euratom-software.github.io/calcam/html/_images/logo.png)

Calcam Python Package
=====================

Calcam is a Python package providing tools for "spatial calibration" of cameras used in science & engineering, i.e. determining the mapping between pixel coordinates in an image and real-world 3D sight lines & coordinates. The calibration method is based on the user matching features between an image to be calibrated and a 3D mesh of the scene being viewed, then using OpenCV’s camera calibration routines to fit a model relating the image and real world coordinates.

It was written for use with camera-based plasma diagnostics on fusion experiments, but may be useful for other applications too. As well as calibrating existing cameras, it can also be used as part of a virtual diagnostic setup, e.g. to evaluate different possible viewing geometries during diagnostic design and/or  setting up synthetic imaging diagnostics for modeling codes. 

Calcam provides a set of GUI tools for calibrating images and doing some basic analysis using the results, and a Python API for making full use of the calibration information as part of a data analysis process.

Documentation
--------------
The full documentation for the latest release version is online at: https://euratom-software.github.io/calcam/

or can be found in the `docs/` directory in the source repo.

License
-------
Calcam is released under the European Union Public Licence (EUPL) v1.1, full details in the included LICENCE.txt

Citing
------
If using this software for published academic work, please cite it! It has a DOI issued by Zenodo; click the DOI badge below for full details of how to cite.

[![DOI](https://zenodo.org/badge/92296352.svg)](https://zenodo.org/badge/latestdoi/92296352)