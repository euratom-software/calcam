'''
Calcam Setup script.

Pretty trivial; Calcam is a pure python module with a few data files and a script to start it.
'''

from setuptools import setup,find_packages

setup(
	  name='Calcam',
	  version='1.9.0',
	  url='https://github.com/euratom-software/calcam',
	  license='European Union Public License 1.1',
	  packages=find_packages(),
	  package_data={'calcam':['ui/*.ui','ui/*.png','usercode_examples/*.py_']},
      entry_points={ 'gui_scripts': [ 'calcam = calcam.gui:start_gui'] },
      zip_safe=False
	  )