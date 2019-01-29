============================
Tomography Geometry Matrices
============================


.. autoclass:: calcam.GeometryMatrix
	:members: plot_coverage, save, load
	
Reconstruction grids
--------------------
.. autoclass:: calcam.PoloidalPolyGrid
    :members: n_cells,n_segments,verts_per_cell,n_vertices,get_extent,get_cell_intersections,plot,remove_cells

Creating commonly used grid types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following functions are provided for convenience to easily generate some commonly used types of grid.

.. autofunction:: calcam.gm.squaregrid

.. autofunction:: calcam.gm.trigrid