=============================
Calcam GUI tools and launcher
=============================

Calcam consists of 5 main GUI tools both for performing calibrations and visualising the results / doing some basic image position analysis, which can all be conveniently accessed via the Calcam GUI launcher interface. The launcher is shown below:

.. image:: images/screenshots/launcher.png
   :alt: Calcam Launcher Screenshot
   :align: left

The buttons to the right-hand side of the Calcam logo each open a new python instance running the selected GUI, and a link to the online copy of this documentation is provided under the logo. The launcher will also check whether you are using the most recent release version of Calcam and will display a message at the bottom of the window alerting you if a newer version is available. More details of each of the tools are given below.

Starting the launcher
---------------------
At installation, the setup script will create a launch script for the Calcam GUI in your Python environment's script directory, which ideally will be in your OS's ``PATH`` environment variable. If this is the case, the Clacam GUI can be started simply by typing::

	calcam

at a terminal / command prompt. On Windows, the launch script takes the form of an executable, so it is easy to make e.g. a desktop or start menu shortcut to start Calcam if you wish (the setup script will tell you where this exe has been created).

Alternatively, the Calcam GUI launcher can be opened by calling the start_gui function in the Calcam Python package::

	>>>import calcam
	>>>calcam.start_gui()