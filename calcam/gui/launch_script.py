import sys
from calcam import gui

if __name__ == '__main__':
    try:
        arg = sys.argv[1]
    except:
        arg = '--launcher'

    if arg == '--launcher':
        gui.open_gui(gui.LauncherWindow)
    elif arg == '--fitting_calib':
        gui.open_gui(gui.FittingCalibrationWindow)
    elif arg == '--alignment_calib':
        gui.open_gui(gui.AlignmentCalibWindow)
    elif arg == '--virtual_calib':
        gui.open_gui(gui.VirtualCalibrationWindow)
    elif arg == '--viewer':
        gui.open_gui(gui.ViewerWindow)