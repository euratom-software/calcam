====================
Installation & Setup
====================

The steps to get up & running with Calcam are:

1. Manually install prerequisits: PyQt and VTK
2. Download & Install Calcam
3. Set up CAD models and image sources

The sections below provide details of each step.

Prerequisites
-------------
The calcam package is known to work under Python versions from 2.7 -  3.6 (and probably works with newer versions but has not been explicitly tested). It also requires the following additional Python packages to be installed:

	- NumPy
	- SciPy
	- OpenCV (opencv-python, otherwise known as cv2) 2.4+ [Some features only available with 3.0+]
	- **PyQt [4 or 5]**
	- **VTK 5.10+** [Tested with versions up to 7.1. Must be built with Qt support enabled for the correct Qt version]
	
The first four of these will be installed automatically by the setup script if they are not already. However, PyQt and VTK cannot be installed automatically, and the setup script merely checks if these are available and will issue a warning message if not. On some platforms / environments it can be non-trivial to get PyQt and VTK working and playing nicely together, particularly in the case of VTK. The easiest way to install them is usually through your OS's package manager, if applicable, or to use a Python distibution such as `Enthought Canopy <https://www.enthought.com/product/canopy/>`_ or `Python (x,y) <https://python-xy.github.io/>`_. which can provide these packages.


Download & Installation
-----------------------
If you have Git available, the latest version of the code can be cloned from the GitHub repository::
	
	git clone https://github.com/euratom-software/calcam.git

Alternatively, the latest release version of the code can be downloaded from: `<https://github.com/euratom-software/calcam/releases>`_ .

Once you have the calcam repository files safely on your computer, the package is installed using the included setup script:
::

	python setup.py install 

(Note: this requres the ``setuptools`` python package to work). This will copy Calcam to the appropriate Python library path and create a launcher script for the Calcam GUI. After the setup script is finished you can delete the downloaded calcam files, should you want.

If using a multi-user system where you do not have the relevant permissions to install globally, adding the ``--user`` switch to the above command will install the package just for you.

**Note for Windows users:** If installing Calcam on Windows, the setup script will finish by printing the location of the GUI launcher ``calcam.exe`` which is created during the installation. It is recommended to make a shortcut to this in a place of your choosing for easy access to the Calcam GUI.

Installing in Development mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you are likely to want to poke around (a.k.a. develop) the code without having to reinstall a lot, you can use the alternative command to install calcam in development mode::

	python setup.py develop

In this case the copy of the code you are installing from remains the "live" version and can be used for development.


Starting Calcam
----------------

On unix-based systems, after installation the Calcam GUI can be started simply by typing ``calcam`` at a terminal. On windows, the setup script will create an executable which can be run to start the GUI (see above).

Alternatively, the GUI can be started from inside Python with::

	import calcam
	calcam.start_gui()

Details of how to use the Calcam GUI and Python API are given in other sections of this documentation.

Setting up CAD Model Definitions and Image Sources
---------------------------------------------------
Since Calcam is based on matching features on images to a CAD model, to do any calibrations you will need to set up CAD models to use. Calcam supports ``.stl`` or ``.obj`` format 3D mesh files, and packages these in to a custom file format (.ccm) along with various metadata to create a Calcam CAD model file. You can have several of these CAD model files and easily switch between them at any given time.

CAD Model Setup
~~~~~~~~~~~~~~~

Camera calibration in Calcam is based on feature matching between images and a CAD model of the scene viewed by the camera. As such, it is necessary to define one or more CAD models for use in calcam. The current version supports ``.stl.`` and ``.obj`` format 3D mesh files. It's usually convenient to split the model in to several individual files containing different parts of the scene, and these can then be turned on or off individually when working with the model in Calcam.

CAD model definitions are written as Python classes and stored in the directory ``[calcam_root]/UserCode/machine_geometry/``. A detailed template is provided in this folder and will be created when Calcam is first imported. To define CAD models for use in calcam, please refer to this file to create your CAD model definition(s).


Image Source Setup (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
As standard, Calcam can load camera images from most common image file formats. However, it may also be desirable to plug in your own code for loading images, e.g. fetching images from a central data store based on some uder inputs. This can be achieved by defining custom image sources in python files in ``[calcam_root]/UserCodemage_sources/``. A detailed, working template is provided in that directory.

Upgrading from Calcam 1.x
--------------------------
The update from Calcam 1.x to Calcam 2 includes large overhauls to the file formats, file storage conventions and Python API. This section provides a brief overview of the major changes.

File Storage
~~~~~~~~~~~~
In Calcam 1, CAD model definitions, other user-defined code, calibration input and results files were stored in a pre-prescribed directory structure, and were saved and loaded by name. In Calcam 2 this is no longer the case; these files can be stored wherever you want and are opened either by graphical file browsing in the Calcam GUI or by file path in the Calcam API.

File Formats
~~~~~~~~~~~~
Whereas in Calcam 1, imported images, point pairs, calibration and fit results were all stored in separate files, in Calcam 2 all of these elements are stored together as a calibration.