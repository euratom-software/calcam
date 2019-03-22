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

import sys
try:
    from calcam import gui
except:
    gui = False

if __name__ == '__main__':

    if not gui:
        print('Could not start Calcam GUI.')
        exit()
        
    try:
        arg = sys.argv[1]
    except:
        arg = '--launcher'

    if arg == '--launcher':
        gui.open_window(gui.Launcher)
    elif arg == '--fitting_calib':
        gui.open_window(gui.FittingCalib)
    elif arg == '--alignment_calib':
        gui.open_window(gui.AlignmentCalib)
    elif arg == '--virtual_calib':
        gui.open_window(gui.VirtualCalib)
    elif arg == '--viewer':
        gui.open_window(gui.Viewer)
    elif arg == '--image_analyser':
        gui.open_window(gui.ImageAnalyser)
    elif arg == '--settings':
        gui.open_window(gui.Settings)
    elif arg == '--cad_edit':
        try:
            filepath = sys.argv[2]
        except:
            filepath = None
        gui.open_window(gui.CADEdit,filepath)
