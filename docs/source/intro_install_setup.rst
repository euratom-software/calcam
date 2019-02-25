====================
Getting Up & Running
====================


Downloading
-----------
The latest version of Calcam can be downloaded using the :guilabel:`Clone or Download` button on the `GitHub repository page <https://github.com/euratom-software/calcam>`_, or, if you want to use Git, can be cloned from the GitHub reporepository with the command::

	git clone https://github.com/euratom-software/calcam.git



Installation
-------------
Calcam is a Python package, so the first thing to do is install Python, if you don't have it already. It is highly recommended to use Pyton 3.x, however Calcam will also work with Python 2.7. You will also need the ``pip`` and ``setuptools`` python packages (which should be redily available, if not installed by default, in most OSs / python distributions) to take advantage of Calcam's setup script as described below. 

Once you have Python working and the Calcam repository files on your computer, the package can be installed from the system command line using the setup script included in the root of the Calcam repository with::

	python setup.py install

This will try to install Calcam's dependencies (Scipy, MatPlotLib, OpenCV, PyQt and VTK) if they are not already installed; copy Calcam to the appropriate Python library path; and create a convenient launcher script for the Calcam GUI. On some combinations of platforms and python versions, the PyQt and VTK libraries may not be installable automatically by the setup script, in which case it will show a warning detailing what could not be installed automatically. Some features of Calcam will still work without these dependencies, but they must be installed in order to make the full Calcam API and GUI available. If you find yourself on a platform where one or more dependencies cannot be installed automatically, the easiest way to get install them is probably to see if your OS has pre-built packages available for them, or to use a python distribution such as `Enthought Canopy <https://www.enthought.com/product/canopy/>`_ or `Python (x,y) <https://python-xy.github.io/>`_ which can provide the necessary dependencies via its own package management.

If installing on a system where you do not have the relevant admin/root permissions to install python packages globally (in which case running the above command will produce an error message saying something about permissions), adding the ``--user`` switch to the above installation command will install the package for your user account and will not require admin/root permissions.

After the setup script is finished you can delete the downloaded calcam files, if you want.


Installing in Development mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you plan to do any development on Calcam, and want to be able to edit the Calcam source code without having to run the setup script again to have your changes go live, you can use the below alternative command to install calcam in development mode::

	python setup.py develop

In this case the copy of the code you are installing from remains the "live" version and can be used for development. Again, the ``--user`` switch can be added to install in the current user's library path rather than the system one.


Initial Configuration
---------------------

Setting up CAD Model Definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Camera calibration in Calcam is based on matching features between camera images and a CAD model of the scene viewed by the camera. As such, it is necessary to define one or more CAD models for use in calcam. The current version supports importing ``.stl`` or ``.obj`` format 3D mesh files. It's usually convenient to split the model in to several individual mesh files containing different parts of the scene, and these can then be turned on or off individually when working with the model. Calcam packages these mesh files in to a custom zipped file format (.ccm) along with various metadata to create a Calcam CAD model file. You can have several such files and easily switch between them at any time. It is recommended to read the :ref:`cadmodel_intro` section in concepts and conventions, then consult the user guide for the :doc:`gui_settings` interface for details of how to set up CAD model definitions.

Setting up custom image sources (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
As standard, Calcam can load camera images from most common image file formats. If desired, you can set up additional custom "image sources", which are user-defined Python modules for loading camera images in to Calcam. For example you may want to load camera data directly from a central data server, or read images from an unusual file format. This can be done by writing a small python module which plugs in to calcam and handles the image loading. A full guide to writing such modules can be found in the :doc:`dev_imsources` developer documentation page. Once written, they can be added to Calcam with the :doc:`gui_settings` interface.



Upgrading from Calcam 1.x
--------------------------
The update from Calcam 1.x to Calcam 2 includes large overhauls to the file formats, file storage conventions and Python API. This section covers the main things users need to know when upgrading from Calcam 1.x to Calcam 2.

File Storage
~~~~~~~~~~~~
In Calcam 1, CAD model definitions, other user-defined code, calibration input and results files were stored in a pre-prescribed directory structure. In Calcam 2 this is no longer the case; these files can be stored wherever you want and are opened either by graphical file browsing in the Calcam GUI or by file path in the Calcam API. The main change required to code calling Calcam to accommodate this will be that calibration results will now need to be loaded by supplying the relative or full path to the results file, rather than just the identifying name as before.

File Formats
~~~~~~~~~~~~
Whereas in Calcam 1, imported images, point pairs, calibration and fit results were all stored in separate files, in Calcam 2 all of these elements are stored together as a calibration. This is to maintain better traceability of calcam calibrations and make it easier for users to share data. Except for ``.csv`` point pair files, Calcam 2 is not backwards compatible with Calcam 1 files, therefore to use existing data from Calcam 1 you must convert your Calcam 1 data to the new Calcam 2 formats. This can be done in bulk using the file converter utility provided in the ``calcam1_file_converter`` directory of the calcam 2 repo. Running ``convert_files.py`` from this directory as a script will open the tool, which is shown below:

.. image:: images/screenshots/file_converter.png
   :alt: Calcam 1.x file converter screenshot

At the top of this window, the "Source Directory", where the tool will look for Calcam 1.x files to convert, is displayed. This is typically detected automatically, but you can also manually set the source directory manually using the :guilabel:`Browse...` button (this should be the complete Calcam 1.x data directory, i.e. the location of the ``FitResults``, ``Images``, ``PointPairs`` etc directories). 

Below this are 2 main sections: the top section for converting existing calibrations, and the bottom section for converting existing CAD model definitions. When the :guilabel:`Convert!` button is clicked in the relevant section, the large status bar at the bottom of the window will show the current progress during the conversion. The three text boxes containing file paths are used to specify where the output Calcam 2 calibration files should be saved to, since in Calcam 2 this can be wherever you want.

When converting calibrations, if the :guilabel:`Try to match with image files based on name` checkbox is ticked, the tool will try to match up calibration results with images by looking for Calcam image save files whose name also appears in the name of the calibration result being converted. If such an image is found, the image will be added to the resulting Calcam 2 save file. To disable this auto-matching, un-tick this checkbox, and Calcam 2 calibration results converted from Calcam 1 files will simply not contain any images.

**Note:** the conversion process does not alter or remove any of the original Calcam 1 data, so if anything goes wrong and you have to, or want to, go back to using Calcam 1.x, the data will still be intact, and it is left to the user to remove the old Calcam 1 data when you feel sufficiently comfortable to do so.


API Changes Summary
~~~~~~~~~~~~~~~~~~~
The change from Calcam 1 to Calcam 2 includes several compatibility breaking API changes. The main changes to the API are:

* The old :class:`calcam.CalibResults` class has been superceded by the new :class:`calcam.Calibration` class. This maintains the methods for working with calibration results which existed in :class:`calcam.CalibResults`, with the addition that :class:`calcam.Calibration` now contains data on the entire calibration process: image, point pairs, fit results and metadata. 

* The old :class:`calcam.VirtualCalib` class has been removed: virtual calibration results are now represented by the new :class:`calcam.Calibration` class, meaning all types of calibration use the same class in Calcam 2.

* The :class:`RayCaster` class has been removed. This is because although more functionality was originally envisaged for this class, that additional functionality is no longer planned for Calcam and therefore only a single method of this class was ever useful. In addition, the important element of this class' state was already being held by other objects. The functionality of the :class:`RayCaster` class has been moved to the function :func:`calcam.raycast_sightlines()`

* Naming conventions: to be more Pythonic, throughout the API argument or function names which previously used capital letters and ``PascalCase`` or ``camelCase`` have been changed to ``lowercase_with_underscores``, while class names keep ``PascalCase``.

For more information, see the API documentation in :doc:`api_analysis` and the :doc:`api_examples` .