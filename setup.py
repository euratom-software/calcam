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
Uses setuptools and pip.
'''

try:
    import pip
except ImportError:
    print('\nERROR: PIP is not installed.\nPlease install the PIP package before using this setup script (see https://pip.pypa.io/en/stable/installing/)\n')
    exit()

import sys
import os
import subprocess
from setuptools import setup,find_packages


def check_dependency(pkg_name):
    '''
    Check if a module is importable, using an external python process.
    The reason for doing this in a separate process is because otherwise
    if we install a module in aseparate process then try to import it
    on a subsequent check in this process,there be segfaults. So to avoid 
    segfaults, use this.
    '''
    test_process = subprocess.Popen([sys.executable,'-c','import {:s}'.format(pkg_name)],stderr=subprocess.PIPE)
    stderr = test_process.communicate()[1]
    return len(stderr) == 0


def pip_install(pkg_name):
    '''
    A small utility function to call pip externally to install a given package.
    Sadly, this seems to be needed because setuptools.setup cannot be trusted to 
    locate some dependencies, or their correct versions, and pip works more reliably.
    '''    
    if '--user' in [arg.lower() for arg in sys.argv]:
        extra_opts = ['--user']
    else:
        extra_opts = []

    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',pkg_name] + extra_opts)
        return True
    except:
        return False



if 'install' in sys.argv or 'develop' in sys.argv:
    # Organise dependencies, in a sadly manual way (I can't 
    # find a suitably portable way to do it more automatically)

    # Essential dependencies
    for prettyname,pkgname,importname in [ ('SciPy','scipy','scipy') ,('MatPlobLib','matplotlib','matplotlib'),('OpenCV','opencv-python','cv2')]:

        if check_dependency(importname):
            print('Dependency {:s}: OK!'.format(prettyname))
        else:
            print('Dependency {:s}: trying to install using pip...\n'.format(prettyname))
            if not pip_install(pkgname):
                print('Could not install essential dependency {:s}. Please install it before installing Calcam.'.format(prettyname))
                exit()
            else:
                print('\n{:s} installed OK!'.format(prettyname))


    # Slightly less essential dependencies
    vtk = False
    if check_dependency('vtk'):
        import vtk
        if vtk.vtkVersion.GetVTKMajorVersion() >= 6:
            if vtk.vtkVersion.GetVTKMajorVersion() >= 8 and vtk.vtkVersion.GetVTKMinorVersion() > 1:
                print('Dependency VTK: Installed version >8.1 is known to cause wrong results!')
            else:
                print('Dependency VTK: OK!')
                vtk = True

    if not vtk:
        print('Dependency VTK: trying to install using pip...\n')
        if pip_install('vtk>=6,<8.2'):
            print('\nVTK installed OK!')
            vtk = True
        else:
            print('\nFailed to install VTK :(')


    pyqt = True
    if check_dependency('PyQt5'):
        print('Dependency PyQt: PyQt5 OK!')
    elif check_dependency('PyQt4'):
        print('Dependency PyQt: PyQt4 OK!')
    else:
        print('Dependency PyQt: trying to install PyQt5 using pip...\n')  
        if not pip_install('PyQt5'):
            print('\nDependency PyQt: trying to install PyQt4 using pip...\n')
            if not pip_install('PyQt4'):
                print('\nFailed to install PyQt :(')
                pyqt = False
            else:
                print('\nPyQt4 installed OK!')
        else:
            print('\nPyQt5 installed OK!')



# Actually do the requested setup actions
s = setup(
          name='Calcam',
          version='2.6.0',
          url='https://euratom-software.github.io/calcam/',
          license='European Union Public License 1.1',
          author='Scott Silburn et.al.',
          packages=find_packages(),
          package_data={'calcam':['gui/icons/*','gui/qt_designer_files/*.ui','gui/logo.png','builtin_image_sources/*.py']},
          entry_points={ 'gui_scripts': [ 'calcam = calcam:start_gui'] },
          zip_safe=False
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
    if pyqt and vtk:
        extra_msg = extra_msg + '\n\nThe GUI can be launched using the executable:\n{:s}'.format(os.path.join(script_dir.replace('/',os.path.sep),'calcam'))
        if sys.platform == 'win32':
            extra_msg = extra_msg + '.exe'
        if script_dir in env_path:
            extra_msg = extra_msg + '\nor just by typing "calcam" at a terminal.'

    if not vtk:
        extra_msg = extra_msg + '\n\nNOTE: Dependency VTK (6.0+; < 8.2) is not installed and could\n      not be installed automatically; the Calcam GUI, rendering\n      and ray casting features will not work until this is installed.'

    if not pyqt:
        extra_msg = extra_msg + '\n\nNOTE: Dependency PyQt (4 or 5) is not installed and\n      could not be installed automatically; the Calcam \n      GUI will not work until this is installed.'


    print('\n***************************************************************\n\nCalcam installation complete.{:s}\n\n***************************************************************\n'.format(extra_msg))
