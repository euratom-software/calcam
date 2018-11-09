====================================
Calcam GUI Launcher
====================================

Calcam includes 5 different GUI tools, which can all be easily accessed from the GUI launcher. On Unix-based systems the launcher can be opened from a terminal simply with::

	calcam

On Windows, a ``calcam.exe`` launcher is created on installation, and the setup script will tell you where this file is located at the end of the installation.

Alternatively, the Calcam GUI launcher can be invoked as a function call in Python::

	>>>import calcam
	>>>calcam.gui.start_gui()

The launcher window is shown below, it simply has buttons to start each GUI tool and short descriptions of the tools, and a button to access an offline copy of this documentation. 

Note: when the GUI tools are opened from the launcher they open in their own new Python instances. This means that (1) The Launcher can be closed without affecting any other open Calcam windows, and (2) If editing the code, you can simply leave the launcher open and re-launch the relevant tool for the code changes to become effective.


For more information aboutwhat each tool does and how to use it, please consult the other pages in this section.


.. toctree::
   :caption: GUI User Guides
   :maxdepth: 1
   :name: guitoc2

   calib_tool_points
   calib_tool_alignment
   virtualcalib_editor
   image_analyser
   model_viewer