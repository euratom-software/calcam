'''
* Copyright 2015-2019 European Atomic Energy Community (EURATOM)
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
    vtk = True
    if check_dependency('vtk'):
        import vtk
        if vtk.vtkVersion.GetVTKMajorVersion() >= 6:
            print('Dependency VTK: OK!')
            
        else:
            print('Dependency VTK: trying to install using pip...\n')
            if pip_install('vtk>=6'):
                print('\nVTK installed OK!')
            else:
                print('\nFailed to install VTK :(')
                vtk = False           
    else:
        print('Dependency VTK: trying to install using pip...\n')
        if pip_install('vtk>=6'):
            print('\nVTK installed OK!')
        else:
            print('\nFailed to install VTK :(')
            vtk = False
           

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
          version='2.2.3',
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
        env_path = None
    
    extra_msg = '\nIt can be imported within python with "import calcam"'
    if script_dir in env_path and pyqt and vtk:
        extra_msg = extra_msg + '\nThe Calcam GUI can be started by typing "calcam" at a terminal.'

    if not vtk:
        extra_msg = extra_msg + '\n\nNOTE: Dependency VTK (6.0+) is not installed and could\n      not be installed automatically; the Calcam GUI, rendering\n      and ray casting features will not work until this is installed.'

    if not pyqt:
        extra_msg = extra_msg + '\n\nNOTE: Dependency PyQt (4 or 5) is not installed and\n      could not be installed automatically; the Calcam \n      GUI will not work until this is installed.'
    if env_path is not None:
        if script_dir not in env_path:
            extra_msg = extra_msg + '\n\nNOTE: The path containing the Calcam GUI launch script:\n\n      {:s}\n\n      is not in your PATH environment variable; consider\n      adding it to enable launching the Calcam GUI directly!'.format(script_dir)
    else:
        extra_msg = extra_msg + '\nLocation of Calcam GUI launcher:\n\n{:s}'.format(script_dir)

    print('\n***************************************************************\nCalcam installation complete.{:s}\n***************************************************************\n'.format(extra_msg))