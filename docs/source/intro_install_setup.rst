====================
Getting Up & Running
====================
.. highlight:: none

Calcam runs under Python 3.5 or newer on Windows, Linux or OSX. As of August 2023, Python 3.11 or older is recommended, because suitable versions of VTK (one of Calcam's major dependencies) are not available from the Python Package Index (PyPi) for newer versions (for newer Python versions you will have to build and install VTK and its Python bindings yourself, which can be quite involved).

You can download Python installers for Windows or OSX from `Python.org <https://www.python.org/downloads/>`_ ; get it from your favourite software repository on Linux; get it from the Microsoft Store on Windows; or use a python environment manager such as `Anaconda <https://www.anaconda.com>`_. 


Installing using pip
--------------------
Once you have Python installed, the recommended way to install Calcam is to use ``pip`` (unless you are using Anaconda, in which case see :ref:`below <conda_install>`), which on most configurations should come installed with Python as standard . If not, documentation for how to get ``pip`` can be found `here <https://pip.pypa.io/en/stable/installing/>`_.


Option 1: From PyPi (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The current release version of Calcam is available on the Python Package Index (PyPi), so can be installed with the single command::

    pip install calcam


Option 2: From GitHub
~~~~~~~~~~~~~~~~~~~~~
If you want to get the "cutting edge" development / pre-release version of Calcam from GitHub rather than the release version, you can install this with the command::

    pip install https://github.com/euratom-software/calcam/zipball/master

Or if you need the version from a different ``git`` branch, simply replace ``master`` in the above URL with the name of the branch. Note that doing this with ``release`` should have the same results as installing from PyPi, as described above.


Option 3: From manually downloaded source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer to download the calcam source code manually, the latest release version can be downloaded from the `Releases page on GitHub <https://github.com/euratom-software/calcam/releases>`_.

Alternatively, the "cutting edge" live development version (which is not guaranteed to be in a fully working state at all times) can be downloaded using the green :guilabel:`Code` button on the `GitHub repository page <https://github.com/euratom-software/calcam>`_.

If you prefer to use Git, which is recommended if you want to do any development on Calcam, the source can be cloned from the GitHub reporepository with the command::

	git clone https://github.com/euratom-software/calcam.git

for the cutting edge / development version (which may contain some bugs or incomplete features at any given time), or::

	git clone -b release https://github.com/euratom-software/calcam.git

for the latest release version.

Once you have a copy of the source files on your computer, navigate to the directory where Calcam has been unzipped or cloned and open a terminal / command prompt. To install Calcam, use the command::

	pip install .

Once the setup is complete, you can delete the downloaded source code.

.. _conda_install:

Installing with Anaconda
------------------------
If you are using `Anaconda <https://www.anaconda.com>`_ to manage your Python environment, you can install Calcam as a conda package with the command::

	conda install -c calcam -c conda-forge calcam
	
This will install Calcam in your current conda environment, and add a launcher for the Calcam GUI to the Anaconda Navigator.


Installing for Development
--------------------------
If you plan to make any modifications to /  do any development work on Calcam, and want to be able to edit the Calcam source code without having to run the setup script again to have your changes take effect, you can install Calcam in development / eidtable mode.

Option 1: Using Git
~~~~~~~~~~~~~~~~~~~
If you want to clone the project directly from GuitHub and install in editable mode, this can be done with the command::

	pip install -e git+https://github.com/euratom-software/calcam.git@master#egg=calcam

This will clone the calcam git repository and install in editable mode, so you can make changes to the downloaded code which will take effect without reinstalling.

Option 2: From manually downloaded source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you prefer to download the calcam source manually, you can get the source as described in the above section, and modify the ``pip`` installation command to::

    pip install -e .

The source you downloaded then remains the "live" copy and changes you make will take effect without re-installing.


Initial Configuration
---------------------
If you will be using the Calcam GUI often, it is highly recommended to make a shortcut to the calcam GUI executable for covenience, and/or make sure the executable is included in your ``PATH`` environment variable. You can find out the executable location using the following Python code:

.. code-block:: python

    import calcam
    print(calcam.gui.exe_path)

Calcam is also provided with icons which can be used for program shortcuts or icons for associated file types. You can find the location of these icons similarly with:

.. code-block:: python

    import calcam
    print(calcam.gui.icons_path)

Setting up CAD Model Definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Camera calibration in Calcam is based on matching features between camera images and a CAD model of the scene viewed by the camera. As such, it is necessary to define one or more CAD models for use in calcam.

When you first start one of the Calcam GUI tools which requires CAD models, you will be prompted to either browse for a folder containing existing Calcam CAD model files or create a new one by importing mesh files. For creating CAD model definitions from mesh files, it is recommended to read the :ref:`cadmodel_intro` section in concepts and conventions, then consult the user guide for the :doc:`gui_cad_editor` for details of how to use the CAD model definition editing tool.

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

This section contains advice on how to troubleshoot any problems you may encounter getting up & running with Calcam.

Insufficient Persmissions to install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If installing on a multi-user system, your account may not have permissions to install calcam in the system-wide python library paths. Typically ``pip`` will handle this for you and install Calcam just for your user account if this is the case. If this does not happen and the setup fails with an error about permissions, adding the ``--user`` option to the installation command will try to install the package for your user account only, which does not require root or admin permissions.


Dependencies
~~~~~~~~~~~~
Installation may fail, or you may encounter errors when first trying to import or run Calcam, if one of the Python modules that Calcam depends on cannot be installed or is not working properly. Calcam requires the following Python modules to be available to work:

+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Module / Library    | Versions Tested / comments                                                                                                                                         |
+=====================+====================================================================================================================================================================+
| SciPy               | Tested up to v1.11.1                                                                                                                                               |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| NumPy               | Tested up to v1.26                                                                                                                                                 |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| MatPlotLib          | Tested up to v3.7.2                                                                                                                                                |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| OpenCV (a.k.a. cv2) | Tested up to to v4.8                                                                                                                                               |
|                     |                                                                                                                                                                    |
|                     | Fisheye camera model only available if using 3.x or newer.                                                                                                         |
|                     |                                                                                                                                                                    |
|                     | If running under OSX older than 10.12, versions of OpenCV newer than 3.2 may cause crashes on import (downgrade to OpenCV < 3.3 to fix this).                      |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| VTK                 | Requires version =>7, tested up to 9.2.6. Note Versions 9.1.x cause crashes when setting large CAD models to wireframe rendering.                                  |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| PyQt                | Works with PyQt4 or PyQt5; pip will try to install PyQt5.                                                                                                          |
|                     |                                                                                                                                                                    |
|                     | PyQt6 support will be added at some point but currently the combination of PyQt6 + VTK9 often causes problems.                                                     |
|                     |                                                                                                                                                                    |
|                     | PyQt5 versions 5.11 and older are known to cause unreadable text in the GUI on OSX when using dark theme.                                                          |
|                     |                                                                                                                                                                    |
|                     | Some versions can result in click positions being registsred wrong on OSX using High DPI mode; not clear what version ranges this affects (see GitHub issue #79)   |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| h5py                | Tested with 3.10.0. Used for MATLAB 7.3 file support in calcam.gm.GeometryMatrix.                                                                                  |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| triangle            | Tested with 20230923. Used for generating triangular meshes in calcam.gm module.                                                                                   |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Also check the  `GitHub issues page <https://github.com/euratom-software/calcam/issues>`_ for any more known compatibility issues not yet updated here.

In most cases with dependency issues, the installation process or Calcam itself should give an error message which makes it clear which dependency is not working properly. In such cases, you will have to install or fix the relevant module yourself before you can continue installing or using Calcam. The sections below give some advice on how to force Calcam to install ignoring dependencies and to troubleshoot them manually.

Installing without dependencies
*******************************
If you encounter problems due to dependencies during the installation, and you believe these are erroneous or want to try to fix them manually, you can force ``pip`` to install Calcam without trying to install any dependencies by adding the ``--no-deps`` option to the installation command.

Manually troubleshooting dependencies
*************************************
If it is not clear that a dependency is the problem, or which it might be, open a python prompt and check if all of the following import commands work without errors:


.. code-block:: python

    from vtk import vtkVersion
    from cv2 import __version__
    from scipy import __version__
    import matplotlib.pyplot

In addition to these, at least one of the following PyQt imports must work for the Calcam GUI to be available ( it doesn't matter which - as long as one works Calcam will be able to use it):

.. code-block:: python

    from PyQt5 import QtCore
    from PyQt4 import QtCore

If any of these required imports fail with errors, you will need to fix the relevant Python module installation before Calcam will work (re-installing the relevant module is a good first thing to try). If all of the required imports work properly, there could be a bug or issue with Calcam.


Black screen / corrupted graphics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you get blank / black displays in Calcam where the CAD model and image are supposed to be displayed, or get a corrupted view of the CAD model, this could be an issue with VTK (the OpenGL visualisation library which Calcam uses to display the CAD and some images). To confirm if your VTK installation is working, you can try running the VTK example code on `this page <https://kitware.github.io/vtk-examples/site/Python/GeometricObjects/CylinderExample/>`_ to check if it gives a result like the picture. If you get correct display testing VTK on its own but not in Calcam, it could be caused by your particilar combination of VTK, PyQt and graphics drivers - see the above section about dependencies. If you do have a problem with VTK, the easiest thing to try is installing a different version (you can check the current version of VTK in the calcam :doc:`gui_settings` interface). You can try installing different versions using `pip`, for example if VTK 9 is causing issues, you can install an older version with the command::

    pip install "vtk<9"

If you cannot get VTK working properly, you may need to try using Calcam on a different computer with a different graphics hardware / software environment.

Reporting Problems
~~~~~~~~~~~~~~~~~~
If you find bugs / problems, please check the `GitHub issues page <https://github.com/euratom-software/calcam/issues>`_ and report the problem there if it isn't already listed.


Updating
--------

Updating using pip
~~~~~~~~~~~~~~~~~~
To update to the latest release version of calcam using ``pip``, use the command::

    pip install --upgrade calcam

From source
~~~~~~~~~~~
To upgrade from manually downloaded source, follow the installation instructions near the top of this page to download the version you want and install.

.. note::
    If installing older versions of Calcam < 2.9, installing with pip may not take care of Calcam's dependencies properly. If you have problems with the instructions on this page for older versions, refer to the offline version of this documentation in the ``docs/html/`` folder of the particular code version.

Updating with Anaconda
~~~~~~~~~~~~~~~~~~~~~~

For Anaconda users, you can update the Calcam package with the command::

	conda update -c calcam calcam


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
