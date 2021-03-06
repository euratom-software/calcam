'''
* Copyright 2015-2020 European Atomic Energy Community (EURATOM)
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
    app = qt.QApplication([])
    win = window_class(app,None,*args)
    if qt.QDialog in window_class.__bases__:
        return win.exec_(),win
    else:
        return app.exec_(),win

def start_gui():

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