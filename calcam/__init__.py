'''
* Copyright 2015-2022 European Atomic Energy Community (EURATOM)
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
import os
import warnings

# Calcam version
with open(os.path.join(os.path.split(os.path.abspath(__file__))[0],'__version__'),'r') as ver_file:
    __version__ = ver_file.readline().rstrip()

# Calcam supports a "headless" mode which does not require VTK / PyQt. This way it can still be used for analysis
# using calibration or raydata objects even if the GUI libraries are not available.
no_gui_reason = None

try:
    import vtk
    vtk.vtkObject.GlobalWarningDisplayOff()

    try:
        from . import gui
        from .gui import start_gui

    except Exception as e:
        warnings.warn('Error importing GUI modules (error: {:}) - the calcam.gui module will not be available.'.format(e),ImportWarning)
        no_gui_reason = str(e)

    from .cadmodel import CADModel
    from .raycast import raycast_sightlines
    from .render import render_cam_view,render_unfolded_wall

except Exception as e:
    warnings.warn('Cannot import VTK python package (error: {:}) - the calcam.gui, calcam.raycast, calcam.render and calcam.cadmodel modules will not be available.'.format(e),ImportWarning)
    no_gui_reason = 'Cannot import VTK python package (error: {:})'.format(e)

# Import the top level "public facing" classes & functions
# which do not rely on VTK.
from .calibration import Calibration
from . import movement
from .raycast import RayData
from .pointpairs import PointPairs

from . import config
from . import gm

# If we have no GUI available, put a placeholder function to print a message about why where the GUI launcher would normally be.
if no_gui_reason is not None:
    start_gui = lambda : print('Could not start calcam GUI: {:s}'.format(no_gui_reason))
