'''
* Copyright 2015-2017 European Atomic Energy Community (EURATOM)
*
* Licensed under the EUPL, Version 1.1 or - as soon they
  will be approved by the European Commission - subsequent
  versions of the EUPL (the "Licence");
* You may not use this work except in compliance with the
  Licence.
* You may obtain a copy of the Licence at:
*
* https://joinup.ec.europa.eu/software/page/eupl
*
* Unless required by applicable law or agreed to in
  writing, software distributed under the Licence is
  distributed on an "AS IS" basis,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
  express or implied.
* See the Licence for the specific language governing
  permissions and limitations under the Licence.
'''


"""
CalCam package.
"""

# Make sure Calcam is in the user's PYTHONPATH
import sys
import inspect
import os
calcampath = sys.path.append(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
if calcampath not in sys.path:
    sys.path.append(calcampath)

version = '1.2'

# List of user-exposed modules which are part of the CalCam package.
module_list = ['paths','machine_geometry','pointpairs','fitting','roi','render','raytrace','image','gui','geometry_matrix','image_filters']

# This will be a list of things which aren't working
missing_modules = []

# Try to import each module, and tell the user if some dependencies are missing.
for ModuleName in module_list:
    try:
        exec('import ' + ModuleName)
    except ImportError as error:
        if 'No module named ' in error.args[0]:
            missing_modules.append(error.args[0][16:])
        else:
            raise

if len(missing_modules) > 0:
    missing_modules = list(set(missing_modules))
    raise ImportError('The following required python modules could not be imported: ' + ', '.join(missing_modules))


from fitting import CalibResults, VirtualCalib
from image import Image
from raytrace import RayData
from pointpairs import PointPairs
from roi import ROI, ROISet
from geometry_matrix import RectangularGeometryMatrix
from gui import start_gui

if int(fitting.cv2.__version__[0]) < 3:
    print('[Calcam Import] Using OpenCV ' + fitting.cv2.__version__ + '; fisheye model features will not be available (requires OpenCV 3)')