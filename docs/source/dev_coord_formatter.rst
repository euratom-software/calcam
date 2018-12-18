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

Note on package structure
-------------------------
If the custom coordinate formatter takes the format of a package (i.e. a folder containing ``__init__.py`` along with other python files), care must be taken if you wish to be able to edit the code "live" i.e. see changes in the code take effect when using the :guilabel:`Refresh` button in the CAD model editor. In order for this to work properly, the coordinate formatter code must be able to be recursively reloaded by the calcam. For example, the following minimal ``__init__.py`` would not work properly:

.. code-block:: python

  from MySubModule import format_coord

In this case because the ``MySubModule`` module is not a member of your package, but only its function :func:`format_coord` is, Calcam will not know to reload the ``MySubModule`` source and therefore changes in the code will not come in to effect when using the :guilabel:`Refresh` feature.

Instead, either write the :func:`format_coord` function directly in ``__init__.py``, or import the entire ``MySubModule`` module and then make a reference to the correct function, e.g.:

.. code-block:: python

  import MySubModule 

  format_coord = MySubModule.format_coord

Adding to a CAD model
----------------------
Once written, a custom coordinate formatter can be added to Calcam CAD model definitions using the :ref:`CAD model definition editor<cad_editor>` .