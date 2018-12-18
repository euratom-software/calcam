[![DOI](https://zenodo.org/badge/92296352.svg)](https://zenodo.org/badge/latestdoi/92296352)
Calcam 
======
Calcam is a Python package providing tools for spatial calibration of cameras, i.e. determining the mapping between pixel coordinates in an image and real-world 3D sight lines & coordinates. The calibration method is based on the user matching features between an image to be calibrated and a 3D mesh of the scene being viewed, then using OpenCV’s camera calibration routines to fit a model relating the image and real world coordinates.

It is primarily written for use with camera-based diagnostics on fusion experiments, but there’s no reason it wouldn’t work more generally. As well as calibrating real cameras, it can also be used as part of a virtual diagnostic setup e.g. to create synthetic diagnostic images from plasma simulations or evaluate different possible viewing geometries during diagnostic design. 

Calcam provides a set of GUI tools for doing the calibration of cameras and some usage of the results, and a Python API for then including camera spatial calibration in your data analysis chain.

Documentation
--------------
The full calcam documentation can be found at: https://euratom-software.github.io/calcam/

or in the docs/ directory in this repository.

For authorship information see AUTHORS.txt, and for details of how to cite the code in academic publications see CITE.txt.
