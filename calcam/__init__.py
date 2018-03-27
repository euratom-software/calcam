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

# Calcam version
__version__ = '2.0.0-dev'


# Make sure we have the right thing in sys.path to be able to import the modules.
import sys
import inspect
import os
calcampath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if calcampath not in sys.path:
    sys.path.insert(0,calcampath)

# List of user-exposed modules which are part of the CalCam package.
module_list = ['paths','machine_geometry','pointpairs','fitting','roi','render','raytrace','image','gui','geometry_matrix','image_filters']


# Try to import each module
for ModuleName in module_list:
    exec('import ' + ModuleName)


from fitting import CalibResults, VirtualCalib
from image import Image
from raytrace import RayData, RayCaster
from pointpairs import PointPairs
from roi import ROI, ROISet
from geometry_matrix import RectangularGeometryMatrix
from gui import start_gui