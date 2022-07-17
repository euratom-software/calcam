====================
Getting Up & Running
====================


Calcam works with Python 3.5 or newer on Windows, Linux or OSX. You can download Python installers for Windows or OSX from `Python.org <https://www.python.org/downloads/>`_ , or get it from your favourite software repository on Linux.

As of June 2022, suitable versions of VTK (one of Calcam's major dependencies) are not available from the Python Package Index for Python >3.9, so it currently it is much easier to install Calcam in Python 3.5 - 3.9 than anything newer (for newer versions you will have to build and install VTK and its Python bindings yourself, which can be quite involved).

The calcam setup script requires the ``pip`` and ``setuptools`` packages to do the installation, and will tell you if either of these are missing (on most configurations they should be installed as standard). Documentation for how to get pip can be found `here <https://pip.pypa.io/en/stable/installing/>`_, and you can then use ``pip`` to install ``setuptools``.


Installing using pip
--------------------
The easiest way to install Calcam is using ``pip``.

Option 1: From PyPi
~~~~~~~~~~~~~~~~~~~
As of version 2.9, the current release version of Calcam is available on the Python Package Index (PyPi), so can be installed with the single command::

    pip install -v calcam

The option ``-v`` is not necessary for the installation to work, but will display useful messages from the Calcam setup script which would otherwise be hidden by ``pip``.

Option 2: From GitHub
~~~~~~~~~~~~~~~~~~~~~
If you want to get the "cutting edge" development / pre-release version of Calcam from GitHub rather than the release version, you can install this with the command::

    pip install -v https://github.com/euratom-software/calcam/zipball/master

Or if you need the version from a different ``git`` branch, simply replace ``master`` in the above URL with the name of the branch. Note that doing this with ``release`` should have the same results as installing from PyPi, as described above.


Option 3: From manually downloaded source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer to download the calcam source code manually, the latest release version can be downloaded from the `Releases page on GitHub <https://github.com/euratom-software/calcam/releases>`_.

Alternatively, the "cutting edge" live development version (which is not guaranteed to be in a fully working state at all times) can be downloaded using the green :guilabel:`Code` button on the `GitHub repository page <https://github.com/euratom-software/calcam>`_.

If you prefer to use Git, which is recommended if you want to do any development on Calcam, the source can be cloned from the GitHub reporepository with the command::

	git clone -b release https://github.com/euratom-software/calcam.git

for the latest release version, or::

	git clone https://github.com/euratom-software/calcam.git

for the development version (which may contain some bugs or incomplete features at any given time).


Once you have a copy of the source files on your computer, navigate to the directory where Calcam has been unzipped or cloned and open a terminal / command prompt. To install Calcam, use the command::

	pip install -v .

Once the setup is complete, you can delete the downloaded source code.

Installing for Development
--------------------------
If you plan to make any modifications to /  do any development work on Calcam, and want to be able to edit the Calcam source code without having to run the setup script again to have your changes take effect, you can install Calcam in development / eidtable mode.

Option 1: Using Git
~~~~~~~~~~~~~~~~~~~
If you want to clone the project directly from GuitHub and install in editable mode, this can be done with the command::

	pip install -v -e git+https://github.com/euratom-software/calcam.git@master#egg=calcam

This will clone the calcam git repository and install in editable mode, so you can make changes to the downloaded code which will take effect without reinstalling.

Option 2: From manually downloaded source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you prefer to download the calcam source manually, you can get the source as described in the above section, and modify the ``pip`` installation command to::

    pip install -v -e .

The source you downloaded then remains the "live" copy and changes you make will take effect without re-installing.


Initial Configuration
---------------------
If you will be using the Calcam GUI often, it is highly suggested to make a shortcut to the calcam GUI executable for covenience. If you did not see a message telling you where this executable is during installation, or need to check it later, you can find out the executable location using the following Python code:

.. code-block:: python

    import calcam
    print(calcam.gui.executable_path)


Setting up CAD Model Definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Camera calibration in Calcam is based on matching features between camera images and a CAD model of the scene viewed by the camera. As such, it is necessary to define one or more CAD models for use in calcam.

The current version supports importing ``.stl`` or ``.obj`` format 3D mesh files. It's usually convenient to split the model in to several individual mesh files containing different parts of the scene, and these can then be turned on or off individually when working with the model. Calcam packages these mesh files in to a custom zipped file format (.ccm) along with various metadata to create a Calcam CAD model file. You can have several such files and easily switch between them at any time.

When you first start one of the Calcam GUI tools which requires CAD models, you will be prompted to either browse for a folder containing existing Calcam CAD model files or create a new one by importing mesh files. For creating CAD model definitions from mesh files, it is recommended to read the :ref:`cadmodel_intro` section in concepts and conventions, then consult the user guide for the :ref:`cad_editor` for details of how to use the CAD model definition editing tool.

Setting up custom image sources (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
As standard, Calcam can load camera images from most common image file formats. If desired, you can set up additional custom "image sources", which are user-defined Python modules for loading camera images in to Calcam. For example you may want to load camera data directly from a central data server, or read images from an unusual file format. This can be done by writing a small python module which plugs in to calcam and handles the image loading. A full guide to writing such modules can be found in the :doc:`dev_imsources` developer documentation page. Once written, they can be added to Calcam with the :doc:`gui_settings` interface.

File type associations (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Since version 2.6, it is possible to open .ccc files directly with the Calcam GUI executable / launch script to make opening calibrations more convenient. To take advantage of this, follow your operating system's normal procedure to associate the Calcam executable with opening files with extension `.ccc`.

.. note::
    Calcam calibration files with extension ``.ccc`` have the MIME type ``application/zip``. Therefore on platforms which manage application / file type associations based on MIME type, rather than filename extension (i.e. Linux), associating calcam files with the calcam executable may have the side effect of associating all ZIP files to calcam too.


System-wide default configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If installing Calcam on a multi-user system, it may be desirable to provide a default configuration (CAD model file locations, image sources) for users running Calcam for the first time on that system. Since version 2.9, Calcam supports this by placing a suitable default configuration file in the Calcam installation directory.

The default configuration file is a json file with the same format as Calcam's normal user configuration file. Therefore the easiest way to set up a default configuration file is to configure Calcam how you want it on your own user account, then copy the configuration file ``~/.calcam_config`` (where ``~`` is your home directory e.g. ``/home/username`` on Unix or ``C:\Users\username`` on windows) to the relevant location.

The place Calcam will look for the default configuration file - where you need to place it to be effective - can be checked with:

.. code-block:: python

    import calcam
    print(calcam.config.default_cfg_path)

In a default calcam installation this file will not exist; if you place a configuration file of your choice there, it will be picked up as the default for new users who do not yet have their own user-specific conifguration file.

Troubleshooting
---------------

At the end of the installation you should see a mesage saying Calcam was successfully installed. If you used the ``-v`` option with ``pip`` (or installed by directly running ``setup.py install``), you should also see an additional message like this::

	***************************************************************

	Calcam installation complete.

	It can be imported as a Python module with "import calcam"

	The GUI can be launched using the executable:
	C:\Users\username\AppData\Roaming\Python\Python37\Scripts\calcam.exe

	***************************************************************

You should then be able to import the calcam module in Python and start the GUI via the executable or via Python (see the GUI user guide). If instead you get error messages, or get errors when trying to start or import calcam, the following sections provide some guidance on fixing common problems.

Insufficient Persmissions to install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The setup process may try to install Calcam in the python library paths for your system. If your user account does not have permissions to write to these paths (e.g. root or admin permissions), the setup will fail with an error about permissions. In this case, adding the ``--user`` option to the installation command will install the package for your user account only, which does not require root or admin permissions.


Problems trying to start Calcam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you cannot import or start Calcam after installation, the most common problems are due to one or more of Calcam's dependencies not working properly. In most cases, Calcam should give an error message which makes it clear which dependency is not working properly. If this is not clear, open a python prompt and try the following import commands, which all need to work for Calcam to be able to work:

.. code-block:: python

    from vtk import vtkVersion
    from cv2 import __version__
    from scipy import __version__
    import matplotlib.pyplot

In addition to these, at least one of the following PyQt imports must work for the Calcam GUI to be available ( it doesn't matter which - as long as one works Calcam will be able to use it):

.. code-block:: python

    from PyQt6 import QtCore
    from PyQt5 import QtCore
    from PyQt4 import QtCore

If any of the required imports fail with errors, you will need to fix the relevant Python module installation before Calcam will work (re-installing the relevant module is a good first thing to try). If all of the required imports work properly, there could be a bug or issue with Calcam.

If troubleshooting dependencies or strange / broken behaviour of Calcam, the table below gives some information on known issues with some versions of Calcam's dependencies:

+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Module / Library    | Versions Tested / comments                                                                                                                                         |
+=====================+====================================================================================================================================================================+
| SciPy               | Up to 1.5.2                                                                                                                                                        |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| NumPy               | Up to 1.19.1                                                                                                                                                       |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| MatPlotLib          | Up to 3.3.0                                                                                                                                                        |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| OpenCV (a.k.a. cv2) | Tested up to to 4.6                                                                                                                                                |
|                     |                                                                                                                                                                    |
|                     | Fisheye camera model only available if using 3.x or newer.                                                                                                         |
|                     |                                                                                                                                                                    |
|                     | If running under OSX older than 10.12, versions of OpenCV newer than 3.2 may cause crashes on import (downgrade to OpenCV < 3.3 to fix this).                      |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| VTK                 | Requires >6, OK with versions < 9.1. With VTK 9.1+, setting CAD models to wireframe can cause Python to exit unexpectedly (see GitHub issue #78)                   |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| PyQt                | Works with PyQt4, PyQt5 or PyQt6 (tested up to 6.2.2)                                                                                                              |
|                     |                                                                                                                                                                    |
|                     | PyQt5 versions 5.11 and older are known to cause unreadable text in the GUI on OSX when using dark theme.                                                          |
|                     |                                                                                                                                                                    |
|                     | Some versions can result in click positions being registsred wrong on OSX using High DPI mode; not clear what version ranges this affects (see GitHub issue #79)   |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+

You can check which versions OpenCV, VTK and PyQt you are using in the :doc:`gui_settings` interface.

OpenGL related error messages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If the Calcam GUI fails to start with a message about OpenGL environment etc, either there is a problem with your installation of VTK, or the graphics setup of your system. Sometimes this can be a result of using Calcam on a remote system using some remote connection software. If you have a different way to connect to the computer running Calcam, try that - if the results don't change, see the section below on graphics problems.

Black screen / corrupted graphics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you get blank / black displays in Calcam where the CAD model and image are supposed to be displayed, or get a corrupted view of the CAD model, this is an issue with VTK (the OpenGL visualisation library which Calcam uses to display the CAD and some images). To confirm if your VTK installation is working, you can try running the VTK example code on `this page <https://kitware.github.io/vtk-examples/site/Python/GeometricObjects/CylinderExample/>`_ to check if it gives a result like the picture. If you do have a problem with VTK, the easiest thing to try is installing a different version. You can check the current version of VTK in the calcam :doc:`gui_settings` interface. For example, if you have VTK 9.0.3 installed but are getting non-working graphics, you can try installing VTK 8 instead with the command::

    pip install "vtk<9"

If you cannot get VTK working properly, you may need to try using Calcam on a different computer with a different hardware / software environment.

Reporting Problems
~~~~~~~~~~~~~~~~~~
If you find bugs / problems, please check the `GitHub issues page <https://github.com/euratom-software/calcam/issues>`_ and report the problem there if it isn't already listed.


Updating
--------

Version Cross-Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calcam uses something close to semanic versioning, to try to make it clear for users to decide when to update. The version number consists of 3 numbers separated by points, in the format ``major.minor.patch``:

* The ``patch`` version is incremented for bug fixes which do not change the functionality.
* The ``minor`` version is incremented when adding new functionality in a backwards-compatible way. Upgrading to a newer minor version of calcam should therefore not break any code which calls calcam as a dependency.
* The ``major`` version number is incremented if incompatible (i.e. non backwards-compatible) changes to the public API or storage file formats are made. The public API is defined as anything covered by the `API User Guide` section of this documentation.

Therefore if you are using Calcam integrated in to some analysis toolchain, it should be safe to upgrade to a newer ``minor`` version but not to a newer major version.

File Compatibility
******************
Newer ``minor`` versions of Calcam will maintain backwards compatibility with files created by earlier versions, but forward compatibility is not guaranteed i.e. files created with newer versions of Calcam may not work properly with older versions.

.. warning::
    Calibration files created with Calcam 2.9 or newer which make use of the image masking feature will cause errors if used with Calcam versions < 2.9

    Calibration files created with Calcam 2.6 or newer cannot be loaded properly in Calcam versions < 2.6

Updating using pip
~~~~~~~~~~~~~~~~~~
To update to the latest release version of calcam using ``pip``, use the command::

    pip install --upgrade calcam

From source
~~~~~~~~~~~
To upgrade from manually downloaded source, follow the instructions near the top of this page to download the version you want and install.

.. note::
    If installing older versions of Calcam < 2.9, installing with pip may not take care of Calcam's dependencies properly. If you have problems with the instructions on this page for older versions, refer to the offline version of this documentation in the ``docs/html/`` folder of the particular code version.

Updating from Calcam 1.x
~~~~~~~~~~~~~~~~~~~~~~~~
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

* The :class:`machine_geometry` module has been removed. Now instead of every CAD modeling having its own class inside calcam.machine_geometry, the :class:`calcam.CADModel` class is used for all CAD models and is instanciated with string arguments to specify the name of the model you want. Also there have been various changes to method names and call signatures in the CAD model class.

* Naming conventions: to be more Pythonic, throughout the API argument or function names which previously used capital letters and ``PascalCase`` or ``camelCase`` have been changed to ``lowercase_with_underscores``, while class names keep ``PascalCase``.
