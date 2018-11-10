====================
Installation & Setup
====================

Dependencies
-------------
Calcam is known to work under Python versions from 2.7 -  3.6, and probably works with newer versions but has not been explicitly tested. In addition to NumPy and SciPy, it also requires the following less standard Python packages to be installed:

	- OpenCV (cv2) 2.4+ [Some features only available with 3.0+]
	- PyQt4 or PyQt5
	- VTK 6.0+ [Tested with versions up to 8.1. Must be built with Qt support enabled for the correct Qt version]
	
These cannot be reliably installed automatically by the setup script on all platforms / environments, so the setup script will merely check if these are working and will issue an error or warning if not. It is highly recommended to get these libraries installed and working before installing Calcam. The easiest way to install them is usually through your OS's package manager, if applicable, or to use a Python distibution such as `Enthought Canopy <https://www.enthought.com/product/canopy/>`_ or `Python (x,y) <https://python-xy.github.io/>`_. which can provide these packages. It can be very tricky to build these libraries and their python bindings from source and get everything working properly together.


Download & Installation
-----------------------
If you have Git available, the very latest version of the code can be cloned from the GitHub repository::

	git clone https://github.com/euratom-software/calcam.git

Alternatively, the latest release version of the code can be downloaded from: `<https://github.com/euratom-software/calcam/releases>`_ .

Once you have the calcam repository files on your computer, the package is installed using the included ``setuptools`` setup script::

	python setup.py install

This will copy Calcam to the appropriate Python library path and create a launcher script for the Calcam GUI. If installing on a system where you do not have the relevant admin/root permissions to install python packages globally, adding the ``--user`` switch to the above command will install the package under your user account specifically.

After the setup script is finished you can delete the downloaded calcam files, should you want.

**Note for Windows users:** If installing Calcam on Windows, the setup script will finish by printing the location of the GUI launcher ``calcam.exe`` which is created during the installation. It is recommended to make a shortcut to this in a place of your choosing for easy access to the Calcam GUI.

Installing in Development mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to be able to edit the Calcam source code without having to run the install again to have your changes go live, you can use the below alternative command to install calcam in development mode::

	python setup.py develop

In this case the copy of the code you are installing from remains the "live" version and can be used for development.


CAD Models and Image Sources
----------------------------

Setting up CAD Model Definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Camera calibration in Calcam is based on matching features between camera images and a CAD model of the scene viewed by the camera. As such, it is necessary to define one or more CAD models for use in calcam. The current version supports importing ``.stl`` or ``.obj`` format 3D mesh files. It's usually convenient to split the model in to several individual mesh files containing different parts of the scene, and these can then be turned on or off individually when working with the model. Calcam packages these mesh files in to a custom zipped file format (.ccm) along with various metadata to create a Calcam CAD model file. You can have several such files and easily switch between them at any time. It is recommended to read the :ref:`cadmodel_intro` section in concepts and conventions, then consult the GUI user guide for the CAD model editor.

Setting up custom image sources (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
As standard, Calcam can load camera images from most common image file formats. If desired, you can set up additional custom "image sources", which are user-defined Python modules for loading camera images in to Calcam. For example you may want to load camera data directly from a central data server, or read images from an unusual file format. This can be done by writing a small python module which plugs in to calcam and handles the image loading. A full guide to writing such modules can be found in the :doc:`dev_imsources` developer documentation page. Once written, they can be added to Calcam with the Settings GUI.



Upgrading from Calcam 1.x
--------------------------
The update from Calcam 1.x to Calcam 2 includes large overhauls to the file formats, file storage conventions and Python API. This section covers the main things users need to know when upgrading from Calcam 1.x to Calcam 2.

File Storage
~~~~~~~~~~~~
In Calcam 1, CAD model definitions, other user-defined code, calibration input and results files were stored in a pre-prescribed directory structure, and were saved and loaded by name. In Calcam 2 this is no longer the case; these files can be stored wherever you want and are opened either by graphical file browsing in the Calcam GUI or by file path in the Calcam API.

File Formats
~~~~~~~~~~~~
Whereas in Calcam 1, imported images, point pairs, calibration and fit results were all stored in separate files, in Calcam 2 all of these elements are stored together as a calibration.