===============================
3D Coordinate Formatters
===============================

Custom 3D coordinate formatters are used to change the format of information displayed by Calcam about a point in 3D on the CAD model. A custom coordinate formatter takes the form of a python module or package. The module or package is required to have a function at its top level called :func:`format_coord` which takes as its input argument a 3 element sequence containing :math:`X,Y,Z` coordinates in metres. The function must return a single string containing the information to be displayed in the Calcam GUI. It is highly recommended that for clarity, different elements of information should be separated by newline characters i.e. ``\n``.


Example
-------
Below is a simple example of a coordinate formatter, which will display the position in cartesian and cylindircal coordinates, and which vacuum vessel segment a point is in:

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

Adding to a CAD model
----------------------
Once written, a custom coordinate formatter can be added to Calcam CAD model definitions using the :ref:`CAD model definition editor<cad_editor>` .