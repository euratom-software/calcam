===============================
3D Coordinate Formatters
===============================

Custom 3D coordinate formatters are used to change the format of information displayed by Calcam about a point in 3D on the CAD model. A custom coordinate formatter takes the form of a python module or package. The module or package is required to have a function at its top level called :func:`format_coord` which takes as its input argument a 3 element sequence containing :math:`X,Y,Z` coordinates in metres. The function must return a single string containing the information to be displayed in the Calcam GUI. It is highly recommended that different elements of information should be separated by newline characters i.e. `\n`.


Example
-------
Below is a simple example of a coordinate formatter

.. code-block:: python

  import numpy as np

  # MAST coordinate formatter, includes sector number.
  def format_coord(coords):

      # Toroidal angle
      phi = np.arctan2(coords[1],coords[0])
      if phi < 0.:
          phi = phi + 2*3.14159
      phi = phi / 3.14159 * 180

      # MAST is divided in to 12 sectors; we want to know what segment we're in.
      sector = (3 - np.floor(phi/30)) % 12
      
      # Build the output string
      formatted_coord = 'X,Y,Z: ( {:.3f} m , {:.3f} m , {:.3f} m )'.format(coords[0],coords[1],coords[2])

      formatted_coord = formatted_coord + u'\nR,Z,\u03d5: ( {:.3f} m , {:.3f}m , {:.1f}\xb0 )'.format(np.sqrt(coords[0]**2 + coords[1]**2),coords[2],phi)

      formatted_coord = formatted_coord + '\nSector {:.0f}'.format(sector)

      return  formatted_coord

Note on sub-module and function names
-------------------------------------
If the custom coordinate formatter takes the form of a package (i.e. a folder containing ``__init__.py`` along with other python files), some care should be taken not to import from a sub-module anything with the same name as that sub-module. This can cause problems if the coordinate formatter has to be re-loaded e.g. re-loading the CAD model or using the :guilabel:`Refresh` button in the CAD model editor, due to the way Calcam re-loads coordinate formatter code.  For example, doing the following in ``__init__.py`` :

.. code-block:: python

  from .do_something import do_something

Would mean that if the coordinate formatting code is re-loaded, the ``do_something`` module would not be re-loaded with the rest of the code, which may or may not cause problems when the code is executed (depending on the contents of the ``do_something`` module). To avoid this issue, ensure functions (or any other object) imported in the coordinate formatting code do not have the same name as their parent modules. For instance, the above example could be fixed by renaming the ``do_something`` module so that the same line looks like:

.. code-block:: python

  from .DoSomething import do_something


Adding to a CAD model
----------------------
Once written, a custom coordinate formatter can be added to Calcam CAD model definitions using the :ref:`CAD model definition editor<cad_editor>` .