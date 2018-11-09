===========
Ray Casting
===========
While working with calibration fitting results alone can give the 3D viewing geometry of the camera, it does not provide information about where the camera sight-lines intersect physical objects in the scene. This is important e.g. for determining sight-line lengths when integrating for synthetic diagnostics, or producing geometry matrices for tomographic inversion. It is also useful for infrared cameras observing heat loaded surfaces, where it is desirable to associate temperature measurements with a real position on a component.

Ray casting can be performed using the calcam.raytrace module. This was originally envisaged to include ray tracing of reflections, however better tools are available for this (e.g. RaySect) and for now only basic ray casting to find sight-line lengths is implemented.

The main tools for ray casting in calcam are the :class:`calcam.RayCaster` class for performing raytracing and :class:`calcam.Raydata` class for storing ray tracing results.

.. py:class:: calcam.RayCaster

	A class for performing ray casting. Can be initialised with no input arguments or with:

	:param FitResults: Camera calibration result defining the camera view for ray casting.
	:type FitResults: :class:`calcam.CalibResults`
	:param CADModel: CAD model for ray casting.
	:type CADModel: :class:`calcam.CADModel`
	:param bool Verbose: Whether to print status when performing operations, the default is True.

	.. py:method:: set_cadmodel(CADModel)

		Sets the CAD model for ray casting to the passed CAD model object.

		:param CADModel: CAD model for ray casting.
		:type CADModel: :class:`calcam.CADModel`

	.. py:method:: set_calibration(FitResults)

		Sets the camera calibration to use for raycasting to the passed result.

		:param FitResults: Calibration result for ray casting.
		:type FitResults: :class:`calcam.CalibResults`

	.. py:method:: raycast_pixels(x=None,y=None,binning=1,Coords='Display')

		Perform ray casting either for each detector pixel or at the provided image coordinates.

		:param array-like x: Image X coordinates at which to do the ray casting. If this is not specified, rays will be cast at the centre of every detector pixel.
		:param array-like y: Image Y coordinates at which to do the ray casting. Must be the same shape as x. If this is not specified, the ray casting will be performed at the centre of each detector pixel.
		:param float binning: If not explicitly providing x and y image coordinates, pixel binning for ray casting. This specifies NxN binning, i.e. for a value of 2, one ray is cast at the centre of every 2x2 cluster of pixels.
		:param str Coords: 'Display' or 'Original'. If providing x and y coordinate inputs, this specifies whether the input is in display or original coordinates. If not providing image coordinates, specifies in what orientation the output should be.
		:return: Ray data object containing the ray cast results.
		:rtype: :class:`calcam.RayData`


.. py:class:: calcam.RayData

	Class for representing the results of ray casting, and providing some convenience methods for working with the results.

	:param str filename: Name of a saved raycast result to load.

	The ray casting results are stored in the following attributes of the RayData object:

	.. py:attribute:: x

		Numpy array containing the image X coordinates at which ray casting was performed.

	.. py:attribute:: y

		Numpy array containing the image Y coordinates at which ray casting was performed.

	.. py:attribute:: ray_start_coords

		Numpy array containing the 3D coordinates of the start of the sight lines, i.e. the camera pupil position. This is the same shape as x and y with an additional dimension; the X,Y,Z components of the ray start coordinates are stored along the extra dimension.

	.. py:attribute:: ray_end_coords

		Numpy array containing the 3D coordinates of the sight line ends, i.e. the locations where the sight lines intersect the CAD geometry. This is usually the most interesting data from the ray cast. This is the same shape as x and y with an additional dimension; the X,Y,Z components of the ray end coordinates are stored along the extra dimension.

	In addition, the following methods are provided for quickly extrating useful information:

	.. py:method:: save(SaveName)

		Save the ray casting results to disk for later use.

		:param str SaveName: Name with which to save the results.

	.. py:method:: get_ray_length(x=None,y=None,PositionTol = 3,Coords='Display')

		Get the sight-line lengths either of all casted sight-lines or at the specified image coordinates.

		:param array x: Image X coordinates at which to get the sight-line lengths. If not specified, the lengths of all casted sight lines will be returned.
		:param array y: Image Y coordinates at which to get the sight-line lengths. Must be the same shape as x. If not specified, the lengths of all casted sight lines will be returned.
		:param float PositionTol: If specifying x and y, it is possible that ray casting was not performed exactly at the specified x and y coordinates. In such a case the length of the nearest cast ray is returned instead, provided its distance is not more than PositionTol pixels away from the requested position.
		:param str Coords: 'Original' or 'Display', specifies whether any x and y coordinates provided are in original or display coordinates. When working with a full-frame raycast, specifies whether the results should be returned in original or display orientation.
		:return: Numpy array containing the sight-line lengths. If the ray cast was for the full detector this array will be the same shape as the camera image, otherwise it will be the same shape as the input image coordinate arrays.

	.. py:method:: get_ray_directions(x,y,PositionTol=3,Coords='Display')

		Return unit vectors specifying the sight-line directions. Note that ray casting is not required to get this information: see :func:`calcam.CalibResults.get_los_direction` for the same functionality, however this can be useful if you have the RayData but not CalibResults objects loaded when doing the analysis.

		:param array x: Image X coordinates at which to get the sight-line directions. If not specified, the directions of all casted sight lines will be returned.
		:param array y: Image Y coordinates at which to get the sight-line directions. Must be the same shape as x. If not specified, the directions of all casted sight lines will be returned.
		:param float PositionTol: If specifying x and y, it is possible that ray casting was not performed exactly at the specified x and y coordinates. In such a case the direction of the nearest cast ray is returned instead, provided its distance is not more than PositionTol pixels away from the requested position.
		:param str Coords: 'Original' or 'Display', specifies whether any x and y coordinates provided are in original or display coordinates. When working with a full-frame raycast, specifies whether the results should be returned in original or display orientation.
		:return: Numpy array containing the sight-line directions. If the ray cast was for the full detector this array will be the same shape as the camera image plus an extra dimension, with the X,Y,Z components of the sight-line directions stored along the extra dimension. Otherwise, it will be the same shape as the input image coordinate arrays with the extra dimension added.