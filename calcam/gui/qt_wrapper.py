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
This module provides a little wrapping of PyQT,
to enable the GUI module to work easily with either
PyQt 4 or 5. Also makes sure matplotlib is using the
correct backend.
"""

# Import all the Qt bits and pieces from the relevant module
try:
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    from PyQt6.QtWidgets import *
    from PyQt6.QtWidgets import QTreeWidgetItem as QTreeWidgetItem_class
    from PyQt6 import uic
    qt_ver = 6

except Exception:
    try:
        import PyQt6
        pyqt6_broken=True
    except Exception:
        pyqt6_broken=False

    try:
        from PyQt5.QtCore import *
        from PyQt5.QtGui import *
        from PyQt5.QtWidgets import *
        from PyQt5.QtWidgets import QTreeWidgetItem as QTreeWidgetItem_class
        from PyQt5 import uic
        qt_ver = 5

    except Exception:
        try:
            import PyQt5
            pyqt5_broken=True
        except:
            pyqt5_broken=False
        try:
            from PyQt4.QtCore import *
            from PyQt4.QtGui import *
            from PyQt4.QtGui import QTreeWidgetItem as QTreeWidgetItem_class
            from PyQt4 import uic
            qt_ver = 4
        except Exception:

            if pyqt6_broken:
                raise ImportError('Could not import required GUI library: Python package "PyQt6" is present but seems to be broken.')

            if pyqt5_broken:
                raise ImportError('Could not import required GUI library: Python package "PyQt5" is present but seems to be broken.')

            raise ImportError('Could not import required GUI library: could not import either "PyQt6", "PyQt5" or "PyQt4" python packages successfully.')


if qt_ver == 6:

    # The PyQt6 API has squirredled away a bunch of flags which used to be available
    # in the root of Qt. So we can have code that works across Qt versions, here I
    # expand a bunch of the ones I use back to where they were in previous Qt versions.
    # I feel bad about this, especially the use of __members__, but not enough to do it another way, for now.

    enums_to_unwrap = {
                        Qt:[Qt.ItemFlag,Qt.CheckState,Qt.TextFormat,Qt.ShortcutContext,Qt.WindowType,Qt.AlignmentFlag,Qt.Orientation],
                        QMessageBox:[QMessageBox.StandardButton,QMessageBox.Icon,QMessageBox.ButtonRole],
                        QAbstractSpinBox:[QAbstractSpinBox.ButtonSymbols],
                        QFileDialog:[QFileDialog.FileMode,QFileDialog.AcceptMode],
                        QSizePolicy:[QSizePolicy.Policy],
                        QDialog:[QDialog.DialogCode],
                        QDialogButtonBox:[QDialogButtonBox.StandardButton],
                        QKeySequence:[QKeySequence.StandardKey],
                        QTextBrowser:[QTextBrowser.LineWrapMode]
                       }

    for parent,enums in enums_to_unwrap.items():
        for enum in enums:
            for item in enum.__members__.items():
                setattr(parent,item[0],item[1])


# Import our local version of QVTKRenderWindowInteracor.
# This avoids a problem with the version shipped with Enthought
# Canopy + PyQt5. Also allows me to work around an annoying rendering issue.
from .qvtkrenderwindowinteractor import QVTKRenderWindowInteractor, QVTKRWIBase
if QVTKRWIBase == 'QGLWidget':
    qt_opengl = True
else:
    qt_opengl = False

# Due to hangover from Python 2 code, the GUI classes  might expect to find
# a class called QString in here, which is in fact the same as Python's string class
QString = str

# Here's a little custom constructor for QTreeWidgetItems, which will
# work in either PyQt4 or 5. For PyQt4 we have to make string list
# arguments in to QStringLists.
def QTreeWidgetItem(*args):
    if qt_ver == 4:
        for i in range(len(args)):
            if type(args[i]) == str:
                args[i] = QStringList(args[i])

    return QTreeWidgetItem_class(*args)

