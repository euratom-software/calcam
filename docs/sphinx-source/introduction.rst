=================
What is Calcam?
=================

Calcam is a Python package providing tools for spatial calibration of cameras, i.e. determining the mapping between pixel coordinates in an image and real-world 3D sight lines & coordinates. The calibration method is based on the user matching features between an image to be calibrated and a 3D mesh of the scene being viewed, then using OpenCV's camera calibration routines to fit a model relating the image and real world coordinates.

Calcam is primarily written for use with camera-based diagnostics on fusion experiments, but there's no reason it wouldn't work more generally. As well as calibrating real cameras, it can also be used as part of a virtual diagnostic setup e.g. to create synthetic diagnostic images from plasma simulations or evaluate different possible viewing geometries during diagnostic design. It is built on OpenCV's camera calibration routines and was inspired by an IDL code of the same name originally written by James Harrison.

Calcam is released under the European Union Public License 1.1.


GUI Tools + API
-----------------

The calcam package has two main aspects:

- A set of GUI tools used for interactively performing camera calibrations (and/or creating virtual diagnostic views) and performing some visualisation and very basic use of the results. 

- A Python API which provides the means for more advanced and flexible use of the calibration results, intended to provide an interface for use in image data analysis, synthetic diagnostics etc.

More details about the GUI and API tools are given on their specific documentation pages.