==============================
General Introduction to Calcam
==============================

Calcam is a Python package for geometric calibration of cameras and for performing related analysis, i.e. for finding the mapping between pixel coordinates in a camera image and physical positions in 3D space. It was created for use with camera diagnostics on fusion experiments, but may be useful in other applications too. As well as calibrating existing cameras, it can also be used to define synthetic camera diagnostics, e.g. to simulate or help define new viewing geometries when designing new camera diagnostics.

The Calcam python package consists of two aspects: a GUI application for performing calibrations or setting up synthetic diagnostics, and a Python API for working with calibrations programatically, e.g. to integrate information about a camera's viewing geometry in to a data analysis workflow. 

The calibration process is based on the user identifying matching positions between a camera image and a CAD model of the scene viewed by the camera, in order to fit a model for the camera geometry. Under the hood it uses OpenCV's camera calibration routines, and therefore the model used in Calcam is the same as in OpenCV.

If using Calcam for published academic work, the code is citeable via a DOI provided by Zenodo. Citation details of both the latest and previous release versions can be found at the link below:

.. image:: https://zenodo.org/badge/92296352.svg
   :target: https://zenodo.org/badge/latestdoi/92296352

Calcam is released under the `European Union Public License 1.1 <https://opensource.org/licenses/EUPL-1.1>`_.

It was inspired by an IDL code of the same name originally written by James Harrison. 

The GUI toolbar button icons used are from `Icons8 <https://icons8.com/>`_.