'''
* Copyright 2015-2018 European Atomic Energy Community (EURATOM)
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
__version__ = '2.x'

# Some stuff will only work if we have VTK.
# For some unknown reason, the GUI breaks if the GUI
# imports are not done before geometry_matrix (?!?)
try:
    import vtk
except ImportError:
    vtk = None

if vtk:
    from . import gui
    from .cadmodel import CADModel
    from .raycast import raycast_sightlines
    from .gui import start_gui
    from .render import render_cam_view


# Import the top level "public facing" classes & functions
from .calibration import Calibration
from .raycast import RayData
from .pointpairs import PointPairs
from . import geometry_matrix
from . import geometry_matrix_old


