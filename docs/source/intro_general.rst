==============================
General Introduction to Calcam
==============================

What it is & does
-----------------

Calcam is a Python package for geometric calibration of cameras and for performing related analysis, i.e. for finding the mapping between pixel coordinates in a camera image and physical positions in 3D space. It was created for use with camera diagnostics on magnetic confinement fusion experiments, but may be useful in other applications too. As well as calibrating existing cameras, it can also be used to define synthetic camera diagnostics, e.g. to simulate or help define new viewing geometries when designing new camera diagnostics.

The Calcam python package consists of two aspects: a GUI application for performing calibrations or setting up synthetic diagnostics, and a Python API for working with calibrations programatically, e.g. to integrate information about a camera's viewing geometry in to a data analysis workflow. 

The calibration process is based on the user identifying matching positions between a camera image and a CAD model of the scene viewed by the camera, in order to fit a model for the camera geometry. Under the hood it uses OpenCV's camera calibration routines, and therefore the model used in Calcam is the same as in OpenCV.

Licensing, citation & credits
-----------------------------

Calcam is released under the `European Union Public License 1.1 <https://opensource.org/licenses/EUPL-1.1>`_.

It was inspired by an IDL code of the same name originally written by James Harrison. 

The GUI toolbar button icons used are from `Icons8 <https://icons8.com/>`_.

If using Calcam for published academic work, the code is citeable via a DOI provided by Zenodo. Citation details of both the latest and previous release versions can be found at the link below:

.. image:: https://zenodo.org/badge/92296352.svg
   :target: https://zenodo.org/badge/latestdoi/92296352

The main contributors to Calcam are listed as authors on the Zeonodo page linked above, see also AUTHORS.txt in the code repository and the list of contributors on GitHub.

Version Numbering
-----------------

Calcam (more-or-less) uses semanic versioning. The version number consists of 3 numbers separated by points, in the format ``Major.Minor.Patch``

* The ``major`` version number is incremented if incompatible (i.e. non backwards-compatible) changes to the public API or storage file formats are made. The public API is defined as anything covered by the `API User Guide` section of this documentation.
* The ``minor`` version is incremented when adding new functionality in a backwards-compatible way. Upgrading to a newer minor version of calcam should therefore not break any code which calls calcam as a dependency.
* The ``patch`` version is incremented for bug fixes which do not change the functionality.
