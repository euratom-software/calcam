'''
Calcam Setup script.

Pretty trivial; Calcam is a pure python module with a few data files and a script to start it.
'''

from distutils.core import setup

setup(
	  name='Calcam',
	  version='1.9.0',
	  url='https://github.com/euratom-software/calcam',
	  license='European Union Public License 1.1',
	  packages=['calcam'],
	  package_data={'calcam':['ui/*.ui','ui/*.png','usercode_examples/*.py_']},
	  scripts=['bin/calcam']
	  )