======
Calcam
======
Calcam is a Python package providing tools for spatial calibration of cameras, i.e. determining the mapping between pixel coordinates in an image and real-world 3D sight lines & coordinates. The calibration method is based on the user matching features between an image to be calibrated and a 3D mesh of the scene being viewed, then uses OpenCV's camera calibration routines to fit a model relating the image and real world coordinates.

Calcam is primarily written for use with camera-based diagnostics on fusion experiments, but there's no reason it wouldn't work more generally. As well as calibrating real cameras, it can also be used as part of a virtual diagnostic setup e.g. to create synthetic diagnostic images from plasma simulations or evaluate different possible viewing geometries during diagnostic design. It is built on OpenCV's camera calibration routines and was inspired by an IDL code of the same name originally written by James Harrison.

The package consists of a GUI application for performing calibrations and a Python API for working with calibration results programatically.

Calcam is released under the European Union Public License 1.1. 
GUI icons from `Icons8 <https://icons8.com/>`_.

.. toctree::
   :caption: Introduction
   :maxdepth: 1
   :name: maintoc
   
   intro_theory
   intro_getting_started
   intro_conventions
   

.. toctree::
   :caption: GUI App User Guide
   :maxdepth: 1
   :name: guitoc

   gui_intro
   gui_calib
   gui_alignment_calib
   gui_virtual_calib
   gui_image_analyser
   gui_viewer

.. toctree::
   :caption: API User Guide
   :maxdepth: 1
   :name: apitoc

   api_analysis
   api_cadmodel


.. toctree::
   :caption: Developer Documentation
   :maxdepth: 1
   :name: devtoc

   dev_imsources