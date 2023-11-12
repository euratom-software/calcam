=============
File Formats
=============

This page gives information about the file formats used by Calcam to store data. This may be useful if you want to read the contents of these files from other software or manually manipulate them. The table below summarises the types of files used by Calcam; click on the file contents in the left-hand row of the table to go to the section below with more details.

+----------------------------+-----------------+---------------------+
| File Contents              |  File Extension |  Format Description |
+============================+=================+=====================+
| :ref:`calfiles`            | .ccc            | Zip file            |
+----------------------------+-----------------+---------------------+
| :ref:`pp`                  | .csv            | CSV table           |
+----------------------------+-----------------+---------------------+
| :ref:`cad`                 | .ccm            | Zip file            |
+----------------------------+-----------------+---------------------+
| :ref:`raydata`             | .nc             | NetCDF              |
+----------------------------+-----------------+---------------------+
| :ref:`gm`                  | .npz            | NuMPy zip format    |
|                            +-----------------+---------------------+
|                            | .mat            | MATLAB binary       |
|                            +-----------------+---------------------+
|                            | .zip            | Zipped ASCII        |
+----------------------------+-----------------+---------------------+
| :ref:`mov`                 | .cmc            | JSON                |
+----------------------------+-----------------+---------------------+

.. _calfiles:

Calibrations
------------

Calcam calibration files are zip archive files, i.e. can be opened with a zip archive manager, and have file extension `.ccc`. The structure inside the zip file can contain the following::


    Calibration File (.ccc)
    |
    ├── image.png               # Camera image being calibrated, in display orientation (present for fitting or manual alignment calibrations)
    ├── subview_mask.png        # Mask image showing which image pixels belong to which sub-views. Pixels with no image contain value 255.
    ├── pointpairs.csv          # Calibration point pair coordinates, see section below for format (present for point fitting calibrations)
    ├── calibration.json        # JSON file containing calibration metadata and configuration
    |
    ├── calib_params_0.json     # For each image sub-view 0 to N, JSON files containing the
    ├── ...                     # fitted intrinsic and extrinsic calibration paremeters
    ├── calib_params_N.json     #
    |
    ├── cad_config.json         # JSON file containing last-used CAD model configuration (name of CAD model, viewport, enabled features).
    |
    └── intrinsics_constraints  # Folder containing additional fitting constraints (e.g. chessboard images or images from other calibrations)
        |
        ├── im_000.png          # For each additional intrinsics constraint 0...N, an image PNG file and pointpairs CSV file.
        ├── points_000.csv
        ├── ...
        ├── im_NNN.png
        └── points_NNN.png

.. _pp:

Point Pairs
-----------

Point pairs are stored in human-readable comma-separated tables in `.csv` files. These give the correspondence between 3D coordimates on the CAD model and image pixel coordinates in each image sub-view. An example layout of a point pairs file for an image with 2 sub-views is shown below:

+-----------------------+-----------+-----------+--+------------+---------+--+------------+---------+
| World Coordinates [m] |           |           |  | Sub-view 0 |         |  | Sub-view 1 |         |
+-----------------------+-----------+-----------+--+------------+---------+--+------------+---------+
| Machine X             | Machine Y | Machine Z |  | Image X    | Image Y |  | Image X    | Image Y |
+-----------------------+-----------+-----------+--+------------+---------+--+------------+---------+
| -3.7296               | -0.2063   | -0.4603   |  |            |         |  | 201.23     | 339.66  |
+-----------------------+-----------+-----------+--+------------+---------+--+------------+---------+
| -3.7358               | 0.2128    | -0.4593   |  |            |         |  | 227.2      | 336.2   |
+-----------------------+-----------+-----------+--+------------+---------+--+------------+---------+
| -3.887                | -0.2041   | 0.4694    |  | 192.35     | 216.91  |  |            |         |
+-----------------------+-----------+-----------+--+------------+---------+--+------------+---------+
| -3.8964               | 0.2144    | 0.4628    |  | 218.54     | 223.7   |  |            |         |
+-----------------------+-----------+-----------+--+------------+---------+--+------------+---------+

There are 2 header rows, then each numerical row consists of, from left-to-right: 3D X, Y, Z of each point, then for each sub-view a blank cell followed by the image X and Y. Each point can have image coordinates in 1 or more sub-views (i.e. if a 3D point is visible in multiple sub-views, the image coordinates can be populated in the columns corresponding to all of those sub-views).

.. _cad:

CAD Models
----------

Similarly to calibrations, CAD model definition files are zip archives but with extension `.ccm`. The easiest (and recommended) way to construct or edit them is with the :doc:`gui_cad_editor`. The contents of the zip file follow the structure shown below::

    CAD Definition File (.ccm)
    |
    ├── model.json              # JSON file containing metadata including model variants, mapping of mesh files to named features, default colours, pre-set viewports.
    ├── wall_contour.txt        # If present, 2-column ASCII table of R,Z locations tracing out the wall contour in the R,Z plane.
    |
    └── usercode                # Folder containing python code for custom format coordinate display. May contain .py files and/or subfolders
    |
    └── .large                  # Folder containing the actual mesh data
        |
        ├── Model variant 1     # Typically (but not necesserily) 1 folder per model variant
        |   |
        |   ├── featureA.stl    # Individual CAD mesh files in .stl or .obj format.
        |   └── featureB.obj
        |
        └── Model variant 2
            |
            ├── featureA.stl
            └── featureB.obj


.. _raydata:

Raycast Results
---------------
Ray casting results are stored in NetCDF files with extension ``.nc``. This file contains:

NetCDF Attributes
~~~~~~~~~~~~~~~~~
* ``history``
    Human readable provenance of the data.
* ``image_transform_actions``
    List of geometrical transformations to convert the camera image corresponding to this calibration from Original to Display coordinates.


NetCDF Variables
~~~~~~~~~~~~~~~~
* ``PixelXLocation``
    Horizontal (x) position on the detector, in pixels, at each point which has been ray-casted. Note that these coordinates are `Display` coordinates, regardless of how the raycasting was called. Shape depends on how the ray-casting was called.
* ``PixelYLocation``
    Vertical (y) position on the detector, in pixels, at each point which has been ray-casted. Note that these coordinates are `Display` coordinates, regardless of how the raycasting was called. Shape depends on how the ray-casting was called.
* ``RayStartCoords``
    The 3D starting position of the rays (i.e. the camera pupil position) in metres. This array has the same shape as ``PixelXLocation`` and ``PixelYLocation`` with an additional dimension to store the X,Y,Z components of the location.
* ``RayEndCoords``
    The 3D ending position of the rays (i.e. the point where the ray intersects the CAD model) in metres. This array has the same shape as ``PixelXLocation`` and ``PixelYLocation`` with an additional dimension to store the X,Y,Z components of the location.
* ``ModelNormals``
    The 3D surface normal vectors at the points of intersection between the sight-lines and CAD model. This array has the same shape as ``PixelXLocation`` and ``PixelYLocation`` with an additional dimension to store the X,Y,Z components of the vectors. This is only present if the `calc_normals=True` argument was used when doing the ray-casting.
* ``Binning``
    If the raycast was for the entire image, this contains the image binning factor used when raycasting.
* ``image_original_shape``
    2-element array containing the (width,height) shape of the image (in original orientation)
* ``image_offset``
    2-element array giving the (x,y) offset of the (0,0) pixel position in this data compared with the full camera detector (i.e. used if the raycast is for a sub-window of the detector)
* ``image_original_pixel_aspect``
    The pizel aspect ratio in the original image.

.. _gm:

Geometry Matrices
-----------------

Geometry matrices can be saved in 3 different formats: NumPy zip files (``.npz``) which can be loaded with ``numpy.load()``; matlab ``.mat`` files, or ``.zip`` files containing ASCII data files. Whichever format the files are saved in, they contain the following variables / data:

.. note::
    When saved in ``.zip`` format, the follwing variables are saved as fields in a ``JSON`` file called ``metadata.json``: ``mat_shape, binning, pixel_order, grid_type, history , im_transform_actions , im_px_aspect, im_coords``. The other variables listed below are saved in individual `.txt` files.


* ``mat_shape``
    2-element array giving the shape (n_rows, n_columns) of the geometry matrix
* ``im_shape``
    2-element array giving the (width,height) of the camera image
* ``im_coords``
    Whether the pixel layout in this matrix is based on starting with an image in 'Display' or 'Original' orientation
* ``mat_row_inds``
    Row indices of each matrix element stored in ``mat_data``. This is a 1D array with number of elements = number of non-zero entries in the geometry matrix.
* ``mat_col_inds``
    Column indices of each matrix element stored in ``mat_data``.  This is a 1D array with number of elements = number of non-zero entries in the geometry matrix.
* ``mat_data``
    The actual geometry matrix data. The position of each element of this array in the geometry matrix is given by the corresponding entries in ``mat_row_inds`` and ``mat_col_inds``.  This is a 1D array with number of elements = number of non-zero entries in the geometry matrix.
* ``grid_verts``
    Nx2 array, where N is the number of vertices in the reconstruction grid. Each row of this array is the R,Z position of a vertex in the reconstruction grid.
* ``grid_cells``
    N_cells x N_verts_per_cell array defining the reconstruction grid cells. Each row of this array corresponds to a single grid cell and gives the indices in to the first dimension of ``grid_verts`` for all of the vertices of that grid cell.
* ``grid_wall``
    Array containing the device wall contour in the R, Z plane. This is an Nx2 array where N is the number of points in the wall contour.
* ``binning``
    Image binning factor the matrix is to be used with
* ``pixel_order``
    The order of 2D -> 1D array unwrapping when re-shaping the 2D image to a 1D vector for use with this matrix. This is a string, either 'C' for left-to-right, then row-by-row, or 'F' for top-to-bottom, then column-by-column.
* ``pixel_mask``
    Boolean array the same size as the (un-binned) image indicating whether each pixel has been included (if True) or not included (if False) in the matrix.
* ``history``
    Human readable provenance of the geometry matrix
* ``grid_type``
    Name of the calcam class identifying the type of reconstruction grid
* ``im_transforms``
    List of geometrical transforms to transform the camera image from Original to Display orientation
* ``im_px_aspect``
    The pixel aspect ratio of the raw camera image.


.. _mov:

Camera Movement Corrections
---------------------------

Movement corrections are saved as `JSON` formatted ASCII files with extension ``.cmc``. The `JSON` data contains the following fields:

* ``transform_matrix``
    The affine transform matrix to transform "moved" coordinates in to alignment with "reference" coordinates.
* ``im_array_shape``
    The (x,y) dimensions of the image to which this correction applies.
* ``ref_points``
    Pixel coordinates (x,y in display coordinates) of the matching points used to determine the transform in the "reference" image.
* ``moved_points``
    Pixel coordinates (x,y in display coordinates) of the matching points used to determine the transform in the "moved" image.
* ``history``
    Human-readable provenance of the data.
