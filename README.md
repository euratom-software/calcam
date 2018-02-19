Calcam
======
Calcam is a Python package providing tools for spatial calibration of cameras, i.e. determining the mapping between pixel coordinates in an image and real-world 3D sight lines & coordinates. The calibration method is based on the user matching features between an image to be calibrated and a 3D mesh of the scene being viewed, then using OpenCV’s camera calibration routines to fit a model relating the image and real world coordinates.

It is primarily written for use with camera-based diagnostics on fusion experiments, but there’s no reason it wouldn’t work more generally. As well as calibrating real cameras, it can also be used as part of a virtual diagnostic setup e.g. to create synthetic diagnostic images from plasma simulations or evaluate different possible viewing geometries during diagnostic design. 

Calcam provides a set of GUI tools for doing the calibration of cameras and some usage of the results, and a Python API for then including camera spatial calibration in your data analysis chain.

Documentation
--------------
The full calcam documentation can be found at: https://euratom-software.github.io/calcam/

or in the docs/ folder of this project. The installation & setup instructions are reproduced below for convenience.

For authorship information see AUTHORS.txt

Installation & Setup
--------------------
Installation & Setup
The steps to get up & running with Calcam are:

1. Install prerequisites
2. Download & Install Calcam
3. Import calcam for the first time
4. Set up CAD model definitions
5. (Optional) Set up image source definitions

The sections below provide details of each step.

### Prerequisites
The calcam package works with Python 2.7+ (tested up to 3.5) and requires the following other Python packages to be available:

* NumPy
* SciPy
* MatPlotLib
* OpenCV (cv2) 2.4+ (Some features only available with 3.0+)
* PyQt (4 or 5)
* VTK 5.10+ (Tested with versions up to 7.1. Must be built with Qt support enabled for the correct Qt version)
* Setuptools (for installation)

It can be rather non-trivial to get all of these up & running and playing nicely together if installing everything from source, particularly in the case of VTK. The easiest way to satisfy all of these dependencies is to use a Python distibution such as Enthought Canopy or Python (x,y).

### Download & Installation
If you have Git available, the latest version of the code can be cloned from GitHub using:

```git clone https://github.com/euratom-software/calcam.git```

Alternatively, the code can be downloaded from: https://github.com/euratom-software/calcam .

Once you have the calcam repository files safely on your computer, the package is installed using the included setup script:

```python setup.py install```

This will copy Calcam to the appropriate Python library path and create a launcher script for the Calcam GUI. After the setup script is finished you can delete the downloaded calcam files, should you so choose.

If using a multi-user system where you do not have the relevant permissions to install globally, adding the ```--user``` switch to the above commands will install the package just for you.

Note for Windows users: If installing Calcam on Windows, the setup script will finish by printing the location of the GUI launcher calcam.exe which is created during the installation. It is recommended to make a shortcut to this in a place of your choosing for easy access to the Calcam GUI.

#### Installing in Development mode
If you are likely to want to poke around (a.k.a. develop) the code without having to reinstall a lot, you can use the alternative command to install calcam in development mode:

```python setup.py develop```

In this case the copy you are installing from remains the “live” version and can be used for development.

### First Import & File Storage
Calcam stores CAD model definitions, calibration results etc in a specific storage directory which is created the first time the module is importted. It is therefore necessary to import or start calcam before continuing setting things up. Thie can be done by importing calcam in Python, i.e.:
```python
import calcam
```
The default location for Calcam’s storage folder is ~/calcam, i.e. typically /home/username/ on unix and C:\Users\username\ on Windows. The storage rolder location can be changed using a function in calcam:
```python
import calcam
calcam.paths.change_save_location(new_path,migrate=True)
```
The optional argument migrate specifies whether Calcam should try to copy any existing files it finds in the storage folder to the new location.

Throughout this documentation the calcam storage directory will be denoted [calcam_root].

### CAD Model Setup
Camera calibration in Calcam is based on feature matching between images and a CAD model of the scene viewed by the camera. As such, it is necessary to define one or more CAD models for use in calcam. The current version supports .stl. and .obj format 3D mesh files. It’s usually convenient to split the model in to several individual files containing different parts of the scene, and these can then be turned on or off individually when working with the model in Calcam.

CAD model definitions are written as Python classes and stored in the directory ```[calcam_root]/UserCode/machine_geometry/```. A detailed template is provided in this folder and will be created when Calcam is first imported. To define CAD models for use in calcam, please refer to this file to create your CAD model definition(s).

### Image Source Setup (Optional)
As standard, Calcam can load camera images from most common image file formats. However, it may also be desirable to plug in your own code for loading images, e.g. fetching images from a central data store based on some uder inputs. This can be achieved by defining custom image sources in python files in ```[calcam_root]/UserCodemage_sources/```. A detailed, working template is provided in that directory.
