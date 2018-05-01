'''
Calcam Setup script.

Fairly usual setuptools setup and then tell Windows users where the launcher exe was created.
'''

from setuptools import setup,find_packages
import sys

setup(
	  name='Calcam',
	  version='2.0.0-dev',
	  url='https://github.com/euratom-software/calcam',
	  license='European Union Public License 1.1',
	  packages=find_packages(),
	  package_data={'calcam':['ui/*.ui','ui/*.png','usercode_examples/*.py_']},
          entry_points={ 'gui_scripts': [ 'calcam = calcam.gui:start_gui'] },
          zip_safe=False
	  )
	 

if len(sys.argv) > 1:
    if sys.platform == 'win32' and (sys.argv[1] == 'install' or sys.argv[1] == 'develop'):
        import sysconfig
        import os
        script_dir = sysconfig.get_path('scripts')
        print('\n****\nPath to Calcam GUI launcher:\n"{:s}"\n****\n'.format(os.path.join(script_dir,'calcam.exe')))