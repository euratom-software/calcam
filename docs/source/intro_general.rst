=============================
Genral Introduction to Calcam
=============================

Calcam is a Python package providing tools for for spatial calibration of cameras, i.e. mapping between points in a camera image and points and sight-lines in 3D space. It was written primarily for use with camera-based diagnostics on fusion experiments, but may be useful for other applications too. The package provides a GUI application for performing calibrations and a Python API for working with calibration results programatically. As well as calibrating real cameras, it can also be used as part of a synthetic diagnostic setup e.g. to define synthetic camera diagnostics for creating synthetic diagnostic images from plasma simulations, or for aiding in designing viewing geometries for camera diagnostics.

Calibration in Calcam is based on the user matching known points between an image and a CAD model of the scene the camera is viewing, in order to fit a model for the camera view. Under the hood it uses OpenCV's camera calibration routines, and therefore the model used in Calcam is the same as in OpenCV.

The Calcam python package consists of two aspects: a GUI application for performing calibrations or setting up synthetic diagnostics, and a Python API for working with calibrations programatically, e.g. to integrate information about a camera's viewing geometry in to a data analysis workflow. 

If using Calcam for published academic work, the code is citeable via a DOI provided by Zenodo. Citation details of both the latest and previous release versions can be found at the link below:

.. image:: https://zenodo.org/badge/92296352.svg
   :target: https://zenodo.org/badge/latestdoi/92296352

Calcam is released under the `European Union Public License 1.1 <https://opensource.org/licenses/EUPL-1.1>`_.

It was inspired by an IDL code of the same name originally written by James Harrison. 

The GUI toolbar icons used are from `Icons8 <https://icons8.com/>`_.