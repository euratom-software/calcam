============================
Tomography Geometry Matrices
============================

Background
----------
It is common to want to recover the distribution of light emission from the plasma in the poloidal plane, assuming toroidal symmetry, from cameras viewing approximately tangentially. The emission is reconstructed on a grid defined the poloidal :math:`(R,Z)` plane. The relationship between the brightness of emission from each grid element and the signal at each camera pixel is then a large system of linear equations, where the brightness of a pixel is given by a weighted sum of the brightness of each grid element along that pixel's line of sight. This is most compactly expressed by a matrix multiplication :math:`\mathbf{Ax = b}` where :math:`\mathbf{A}` is the *geometry matrix*, :math:`\mathbf{x}` is a vector of the brightnesses of each grid element (which we want to find), and :math:`\mathbf{b}` is a vector of the pixel brightnesses. The calcam.gm module provides tools for generating the geometry matrix :math:`\mathbf{A}`.

For examples of using the features documented on this page, see the :doc:`api_examples` page.

Model assumptions and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In calculation of the geometry matrices, camera sight-lines are assumed to be infinitely thin pencil beams, i.e. finite etendue and depth-of-field effects are not included. If ray casting with binning = 1, each image pixel is characterised by a single pencil beam sight-line at the image centre, i.e. the finite size of the pixels is not accounted for. The values of matrix element :math:`i,j` is given by the length, in metres, of the :math:`i^{th}` sight line which passes through the :math:`j^{th}` grid cell.


The Geometry Matrix class
-------------------------
.. autoclass:: calcam.gm.GeometryMatrix(grid,raydata,pixel_order='C',calc_status_callback=calcam_status_printer)
    :members: grid,data,get_los_coverage,set_binning,set_included_pixels,get_included_pixels,save,format_image,fromfile


Reconstruction grids
--------------------
.. autoclass:: calcam.gm.PoloidalVolumeGrid
    :members: n_cells,n_segments,n_vertices,extent,get_cell_intersections,plot,remove_cells,interpolate

Creating commonly used grid types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following functions are provided for convenience to easily generate some commonly used types of grid.

.. autofunction:: calcam.gm.squaregrid

.. autofunction:: calcam.gm.trigrid