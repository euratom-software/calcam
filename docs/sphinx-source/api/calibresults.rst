=====================================
Working with Calibration Results
=====================================

Once you have created calibration results using the Calcam GUI, you can import calcam as a Python package and use it to extract the information you want from the calibrations and/or integrate the results in to your data analysis.

This page documents the classes available for loading saved calibration results and extracting information about the camera's sight-line geometry. Note that these only provide information about the sight-line origin and directions, and not intersections between the camera's sight lines and CAD model geometry. For that see `Ray Casting`.

.. py:class:: calcam.CalibResults

	Class representing a calibration result. 

	This class is used for loading a saved calibration result and provides various methods for getting information about the calibrated camera's viewing geometry.

	:param str SaveName: Name of the calibration result to load, as entered when creating the result in the Calcam GUI. 


	.. py:method:: get_pupilpos(x_pixels=None, y_pixels=None,field=0,Coords='Display')
	
		Get the camera pupil position in 3D space. Can be used together with :func:`get_los_direction` to obtain a full description of the camera's sight line geometry.

		For non-split-field images (i.e. most normal images) this can simply be called with no arguments; the input arguments are typically only relevant when different pixels in the image could be looking through different pupils.

		:param array-like x_pixels: Image X coordinates at which to return the pupil positions.
		:param array-like y_pixels: Image Y coordinates at which to return the pupil positions(must be the same shame as x_pixels).
		:param int field: This can be used to specify which sub-field-of-view to get the pupil position for, rather than specifying a set of pixel coordinates. If both field and pixel coordinates are specified the field parameter will be ignored.
		:param str Coords: 'Display' or 'Original'; If specifying x_pixels and y_pixels, this specifies whether the pixel coordinate inputs are in Display or Original coordinates.
		:return: Camera pupil position in 3D space. If not specifying x_pixels or y_pixels inputs, this will be a 3x1 array containing the [X,Y,Z] coordinates of the pupil position. If using x_pixels and y_pixels inputs, the output array will be the same shape as the x_pixels and y_pixels input arrays with an additional dimension added; the X, Y and Z components are then given along the new array dimension.


	.. py:method:: get_fov(field=None)

		Get the horizontal and vertical field-of-view (or more correctly, angle-of-view) of the camera. 

		For images with split fields-of-view, get the horizontal and vertical FOV angles for the specified sub-field.

		:param int field: For images with split fields-of-view, index of the sub-field whose FOV to return.
		:return: 1x2 array containing the horizontal and vertial view angles in degrees.


	.. py:method:: get_los_direction(x_pixels=None,y_pixels=None,Coords='Display')

		Get unit vectors representing the directions of the camera's sight-lines in 3D space. Can be used together with :func:`get_pupilpos` to obtain a full description of the camera's sight-line geometry.

		:param array-like x_pixels: Image X coordinates at which to return the sight-line vectors. If not specified, the method will return sight-lines at the centre of every detector pixel.
		:param array-like y_pixels: Image Y coordinates at which to return the sight-line vectors. Must be the same shape as x_pixels. If not specified, the method will return sight-lines at the centre of every detector pixel. 
		:param str Coords: 'Display' or 'Original', specifies whether the x_pixels and y_pixels inputs are in original or display coordinates. If x_pixels and y_pixels are not used, specifies whether the returned sight-line array orientation should be in display or original coordinates.
		:return: Array of sight-line vectors. If specifying x_pixels and y_pixels, the output array will be the same shape as the input arrays but with an extra dimension added. The extra dimension contains the [X,Y,Z] components of the sight-line vectors. If not specifying x_pixels and y_pixels, the output array shape will be HxWx3 where H and W are the detector width and height in pixels. 


	.. py:method:: project_points(ObjPoints,CheckVisible=False,OutOfFrame=False,RayData=None,RayCaster=None,VisibilityMargin=0,Coords='Display')

		Get the image coordinates corresponding to given real-world 3D coordinates. Optionally can also check whether the 3D points are hidden from the camera's view by part of the CAD model.

		:param array-like ObjPoints: 3D point coordinates to project to image coordinates. Can be EITHER an Nx3 array, where N is the number of points and each array row represents a 3D point, or an array-like of 3 element array-likes, where each 3 element array specifies a point.
		:param bool CheckVisible: Whether to check if the 3D points are visible to the camera or hidden behind CAD geometry. If set to True, either RayData or RayCaster inputs must also be specified.
		:param bool OutOfFrame: Whether to force returning pixel coordinates even if the point is not within the camera field-of-view. If set to True, even if an input point is outside the field-of-view its image coordinates according to the calibration model will still be returned. This option only has an effect for non split-field images.
		:param RayCaster: If using CheckVisible=True, you should provide a RayCaster object which will be used for checking whether the 3D points are hidden from view. If both this and RayData are specified, this takes precident. 
		:type RayCaster: :class:`calcam.RayCaster`
		:param RayData: If using CheckVisible=True and not specifying a RayCaster, this must contain raycasting results for this camera. If both RayCaster and RayData objects are provided, the RayData is ignored.
		:type RayData: :class:`calcam.RayData`
		:param float VisibilityMargin: An optional "fudge factor" to adjust the tolerance of CheckVisibility (I can't remember why this exists). It is a distance in metres such as points this far behind a CAD model surface will still be considered visible to the camera.
		:param str Coords: 'Display' or 'Original', whether to return the results in display or original coordinates.
		:return: A list of Nx2 NumPY arrays containing the image coordinates of the given 3D points (N is the number of input 3D points). Each element of this list corresponds to a single sub field-of-view, so for non-split field images this will return a single element list containing an Nx2 array. Each row of the array contains the [X,Y] image coordinates of the corresponding input point. Points not visible to the camera, either because they are outisde the camera field-of-view or hidden by CAD geometry and CheckVisible is enabled, have their coordinates set to ``[np.nan, np.nan]``.

	.. py:method:: export_sightlines(filename,x_pixels=None,y_pixels=None,Coords='Display',binning=1)

		Export 3D camera sight-line geometry to a NetCDF file. This was originally written for interfacing with RaySect, which can load these files using XX, although hopefully a better interface for that will soon exist. For a description of the file format see the File Formats page.

		:param str filename: File name to save the netCDF file to (relative path to current working directory). It does not matter is the .nc file extension is included or not.
		:param array x_pixels: Image X coordinates at which to export the sight lines. If this is not provided, the method exports sight lines for the centre of every detector pixel.
		:param array y_pixels: Image Y coordinates at which to export the sight lines. Must be the same shape as x_pixels. If this is not provided, the method exports sight lines for the centre of every detector pixel.
		:param str Coords: 'Display' or 'Original'. If specifying x_pixels and y_pixels input, this specifies whether the inputs are in original or display coordinates. Otherwise, determines whether the output is oriented as the original or display image orientation.
		:param float binning: If exporting sight-lines for the entire detector, this specifies NxN pixel binning. For example setting this to 2 will export 1 sight-line for every 2x2 cluster of pixels, while setting it to <1 will result in sub-sampling of each pixel.

	.. py:method:: undistort_image(image,coords=None)

		Un-distort an image from the calibrated camera. Only supported for non split-field cameras.

		:param image: Camera image to be un-distored.
		:type image: array or :class:`calcam.Image`
		:param str coords: 'Display' or 'Original', specifies whether the input image is in 'original' orientation (i.e. directly out of camera) or 'display' orientation. This is only really required for square images where the display and original orientation images have the same dimensions; in all other cases it is determined automatically from the image dimensions.
		:return: Un-distorted image.
		:rtype: NumPY array or :class:`calcam.Image` (same as input image type)

.. py:class:: calcam.VirtualCalib

	Class representing a virtual camera calibration, created with the Virtual Calibration editor. This type of object has the same interface and can be used interchangably with :class:`calcam.CalibResults`, but provides a distinction when loading calibrations between real and virtual cameras (and has some internal implementation differences).

	:param str SaveName: Name of the virtual calibration to load, as entered when saving the virtual calibration from the GUI.