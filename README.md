# Calcam
Calcam is a python package for spatial calibration of cameras (i.e. mapping the relationship between pixel coordinates in an image and real-world 3D coordinates), written for use with camera diagnostics on fusion experiments (but would work more generally). It is built on OpenCV's camera calibration routines and is inspired by an IDL code of the same name by James Harrison.

Calcam provides a set of GUI tools for doing the calibration of cameras and some usage of the results, and a Python API for then including camera spatial calibration in your data analysis chain.

For more authorship information see AUTHORS.txt

For somewhat out of date documentation and user guide see docs/ directory; better documentation in the pipleine.

## How to Install

1. Make sure you have the required dependencies installed & working (note: this can be quite non-trivial). These are:
	- Python 2.7+ [Tested with versions up to 3.5]
	- SciPy / NumPy
	- MatPlotLib
	- OpenCV (cv2) 2.4+ [Some features only available with 3.0+]
	- PyQt [4 or 5]
	- VTK 5.10+ [Tested with versions up to 7.1. Must be built with Qt support enabled for the correct Qt version]

   The easiest way to get a python environment satisfying these dependencies is to use a Python distibution such as [Enthought Canopy](https://www.enthought.com/product/canopy/) or [Python (x,y)](https://python-xy.github.io/). 


 2. Get the latest version of the code by cloning or downloading from: https://github.com/euratom-software/calcam

3. In the directory where you downloaded the code, run the setup script:<br>`python setup.py install` (normal installation; you can then delete the downloaded folder if you want)<br>or<br>`python setup.py develop` (development mode; the copy you downloaded becomes the live copy so you can work on the live version of the code. Recommended if you want to poke around inside calcam).<br> If on a multi-user machine where you don't have persmissions to install it globally, add the `--user` option to the above commands to install it just for you.
4. Set up CAD model definitions and custom image sources as reuiqred; refer to the full documentation for how to do this.

## How to Use
The calcam GUI for preparing calibrations (amongst other things) can be run straight from a terminal on UNIX based systems simply by running: `calcam`
On windows, a calcam.exe is created to serve the same prupose, and the setup script will tell you where this .exe is when installing.

Then for analysing the results or making use of the calibration results in your data analysis, Calcam can be imported as a Python package: `import calcam`. Please refer to the full documentation for details of using this package.