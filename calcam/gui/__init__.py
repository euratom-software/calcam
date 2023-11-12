'''
* Copyright 2015-2023 European Atomic Energy Community (EURATOM)
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
import sys
import os
import sysconfig

from .viewer import Viewer
from .launcher import Launcher
from .virtual_calib import VirtualCalib
from .fitting_calib import FittingCalib
from .alignment_calib import AlignmentCalib
from .image_analysis import ImageAnalyser
from .settings import Settings
from .cad_edit import CADEdit
from .movement_correction import ImageAlignDialog
from . import qt_wrapper as qt
from ..calibration import Calibration


def open_window(window_class,*args):
    """
    Open a GUI window.

    Parameters:
         window_class : Calcam window class.

    Returns:
        PyQt return value after GUI execution
    """
    app = qt.QApplication([])
    win = window_class(app,None,*args)
    if qt.QDialog in window_class.__bases__:
        return win.exec(),win
    else:
        return app.exec(),win

def start_gui():
    """
    Starts the Calcam launcher GUI.
    """
    try:
        cal = Calibration(sys.argv[1])
        if cal._type == 'alignment':
            open_window(AlignmentCalib,cal.filename)
        elif cal._type == 'virtual':
            open_window(VirtualCalib,cal.filename)
        elif cal._type == 'fit':
            open_window(FittingCalib,cal.filename)
    except:
        open_window(Launcher)


# Locate the calcam executable and put its path in a handy string variable for the user.
exe_path = None
if sys.platform == 'win32':
    exe_name = 'calcam.exe'
else:
    exe_name = 'calcam'

script_path = sysconfig.get_path('scripts')
if os.access(os.path.join(script_path,exe_name),os.X_OK):
    exe_path = os.path.join(script_path, exe_name)
else:
    script_path = sysconfig.get_path('scripts','{:s}_user'.format(os.name))
    if os.access(os.path.join(script_path, exe_name), os.X_OK):
        exe_path = os.path.join(script_path, exe_name)

if exe_path is not None:
    exe_path = os.path.realpath(exe_path)

del exe_name, script_path

icons_path = os.path.realpath(os.path.join(os.path.split(os.path.abspath(__file__))[0],'icons'))