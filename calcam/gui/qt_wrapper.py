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
This module provides a little wrapping of PyQT,
to enable the GUI module to work easily with either
PyQt 4 or 5. Also makes sure matplotlib is using the
correct backend.
"""

# Import all the Qt bits and pieces from the relevant module
try:
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
    from PyQt5.QtWidgets import QTreeWidgetItem as QTreeWidgetItem_class
    from PyQt5 import uic
    qt_ver = 5
except:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
    from PyQt4.QtGui import QTreeWidgetItem as QTreeWidgetItem_class
    from PyQt4 import uic
    qt_ver = 4


# Import our local version of QVTKRenderWindowInteracor.
# This avoids a problem with the version shipped with Enthought
# Canopy + PyQt5. Also allows me to work around an annoying rendering issue.
from .qvtkrenderwindowinteractor import QVTKRenderWindowInteractor, QVTKRWIBase

# If we're on Python 3, there is no QString class because PyQt uses 
# python 3's native string class. But for compatibility with the rest 
# of the code we always want there to be a string class called QString
try:
    QString
except:
    QString = str

# Make sure Matplotlib is using the right backend for
# whichever version of Qt we managed to load.
import matplotlib
try:
    matplotlib.use('Qt{:d}Agg'.format(qt_ver),warn=False,force=True)
except:
    print('WARNING: Error forcing Matplotlib to use Qt{:d} backend; there may be GUI issues!'.format(qt_ver))

# Here's a little custom constructor for QTreeWidgetItems, which will
# work in either PyQt4 or 5. For PyQt4 we have to make string list
# arguments in to QStringLists.
def QTreeWidgetItem(*args):
    if qt_ver == 4:
        for i in range(len(args)):
            if type(args[i]) == str:
                args[i] = self.QStringList(args[i])

    return QTreeWidgetItem_class(*args)

