============================
Tomography Geometry Matrices
============================

Background
----------
It is common to want to recover the distribution of plasma emissivity in the poloidal plane, assuming toroidal symmetry, from cameras viewing approximately tangentially. Such tomographic inversion is usually done by constructing a grid in the poloidal plane, on which we want to reconstruct the emission. Then, the relationship between the brightness of emission from each grid cell and the signal at each camera pixel is a large system of linear equations, where the brightness of a pixel is given by a weighted sum of the brightness of each grid cell along that line of sight. This is most compactly expressed by a matrix multiplication :math:`\mathbf{Ax = b}` where :math:`\mathbf{A}` is the geometry matrix, :math:`\mathbf{x}` is a vector of the brightnesses of each grid element (which we want to recover), and :math:`\mathbf{b}` is the image pixel data re-formatted in to a vector. The :module:calcam.gm module provides tools for generating the matrix :math:`\mathbf{A}`.

Model assumptions and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In calculation of the geometry matrices, camera sight-lines are assumed to be infinitely thin pencil beams. The values of matrix element :math:`i,j` is given by the length, in metres, of the :math:`i^{th}` sight line which passes through the :math:`j^{th}` grid cell.


The Geometry Matrix class
-------------------------
.. autoclass:: calcam.gm.GeometryMatrix
	:members: grid,data,get_los_coverage,get_mean_rcc,set_binning,set_included_pixels,get_included_pixels,save,format_image,load

Reconstruction grids
--------------------
.. autoclass:: calcam.gm.PoloidalVolumeGrid
    :members: n_cells,n_segments,verts_per_cell,n_vertices,get_extent,get_cell_intersections,plot,remove_cells

Creating commonly used grid types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following functions are provided for convenience to easily generate some commonly used types of grid.

.. autofunction:: calcam.gm.squaregrid

.. autofunction:: calcam.gm.trigrid