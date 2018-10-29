import sys
from calcam import gui

if __name__ == '__main__':
    try:
        arg = sys.argv[1]
    except:
        arg = '--launcher'

    if arg == '--launcher':
        gui.open_window(gui.LauncherWindow)
    elif arg == '--fitting_calib':
        gui.open_window(gui.FittingCalibrationWindow)
    elif arg == '--alignment_calib':
        gui.open_window(gui.AlignmentCalibWindow)
    elif arg == '--virtual_calib':
        gui.open_window(gui.VirtualCalibrationWindow)
    elif arg == '--viewer':
        gui.open_window(gui.ViewerWindow)
    elif arg == '--image_analyser':
        gui.open_window(gui.ImageAnalyserWindow)
    elif arg == '--settings':
        gui.open_window(gui.SettingsWindow)
    elif arg == '--cad_edit':
        try:
            filepath = sys.argv[2]
        except:
            filepath = None
        gui.open_window(gui.CADEditorWindow,filepath)