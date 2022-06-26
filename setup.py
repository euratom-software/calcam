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

'''
Calcam Setup script.
Uses setuptools.
'''
import sys
import os

try:
    from setuptools import setup,find_packages
except Exception:
    print('\nThe calcam installation process requires Python setuptools, but setuptools does not appear to be available. Giving up.')
    exit()

# Dependencies without alternate names
dependencies = ['scipy','matplotlib','opencv-python-headless','vtk>=6,<9.1']


try:
    # If we already have PyQt6, specify that as the dependency.
    from PyQt6.QtCore import *
    dependencies.append('PyQt6')
except:
    try:
        # If we already have PyQt5, specify that as the dependency.
        from PyQt5.QtCore import *
        dependencies.append('PyQt5')
    except Exception:
        # Check if PyQt5 is present and broken, or just completely absent.
        try:
            import PyQt5
            pyqt5_broken=True
        except Exception:
            pyqt5_broken = False

        try:
            # If we already have PyQt4, specify that as the dependency.
            from PyQt4.QtCore import *
            dependencies.append('PyQt4')
        except:
            # Case where we have no working PyQt at all.
            if pyqt5_broken:
                dependencies.append('PyQt4')
            else:
                dependencies.append('PyQt5')


# Actually do the requested setup actions
s = setup(
          name='Calcam',
          version='2.8.3',
          url='https://euratom-software.github.io/calcam/',
          license='European Union Public License 1.1',
          author='Scott Silburn et.al.',
          packages=find_packages(),
          package_data={'calcam':['gui/icons/*','gui/qt_designer_files/*.ui','gui/logo.png','builtin_image_sources/*.py']},
          entry_points={ 'gui_scripts': [ 'calcam = calcam:start_gui'] },
          zip_safe=False,
          install_requires=dependencies,
          extras_require={'docs':['Sphinx','sphinx-rtd-theme']}
         )

# Offer the user some useful and informative statements about what just happened.
if 'install' in sys.argv or 'develop' in sys.argv:

    if 'install' in sys.argv:
        script_dir = s.command_obj['install'].install_scripts
    elif 'develop' in sys.argv:
        script_dir = s.command_obj['develop'].script_dir

    try:
        env_path = os.environ['PATH']
    except KeyError:
        env_path = ''

    extra_msg = '\n\nIt can be imported as a Python module with "import calcam"'

    extra_msg = extra_msg + '\n\nThe GUI can be launched using the executable:\n{:s}'.format(os.path.join(script_dir.replace('/',os.path.sep),'calcam'))
    if sys.platform == 'win32':
        extra_msg = extra_msg + '.exe'
    if script_dir in env_path:
        extra_msg = extra_msg + '\nor just by typing "calcam" at a terminal.'

    print('\n***************************************************************\n\nCalcam installation complete.{:s}\n\n***************************************************************\n'.format(extra_msg))

