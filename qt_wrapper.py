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
This module provides a little wrapping of PyQT,
to enable the GUI module to work easily with either
PyQt 4 or 5.
"""

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


# Here's a function which creates a new instance of QTreeWidgetItem,
# which will work in either PyQt4 or 5. For PyQt4 we have to make string list
# arguments in to QStringLists.
def QTreeWidgetItem(*args):
	if qt_ver == 4:
		for i in range(len(args)):
			if type(args[i]) == str:
				args[i] = self.QStringList(args[i])

	return QTreeWidgetItem_class(*args)



try:
	if qt_ver == 4:
		from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
	else:
		from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except:
	raise ImportError('VTK Qt module could not be imported. Check your VTK library has been built with Qt support.')

