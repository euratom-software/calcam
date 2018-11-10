*************************
Using Calibration Results
*************************

Once you have created calibrations using the Calcam GUI tools, you can import calcam as a Python package and use it to extract information from the calibrations and/or integrate use of the results in to your workflow in Python. This page documents the features of the Calcam API for working with calibration results.

For examples of using these features, see the examples page.

The Calibration Class
~~~~~~~~~~~~~~~~~~~~~
The calibration class is used to represent Calcam calibrations and provides various methods for getting information about the camera viewing geometry.

.. autoclass:: calcam.Calibration
	:members: get_cam_matrix,get_cam_roll,get_cam_to_lab_rotation,get_fov,get_image,get_los_direction,get_pupilpos,project_points,undistort_image

Rendering images
~~~~~~~~~~~~~~~~
Calcam can be used to render images of a CAD model from the calibrated camera's point of view. The render includes all lens distortion in the camera model, so should match exactly with a well calibrated real camera image.

.. autofunction:: calcam.render_cam_view

Ray casting
~~~~~~~~~~~
The calibration class can provide line-of-sight directions for given pixel coordinates, however it is often necessary to know where these sight lines terminate at a surface in the CAD model. For example in calculation of geometry matricies for camera data inversion, the length of each camera sight line needs to be known. The function :func:`calcam.raycast_sightlines` is provided for this purpose; it determines the 3D coordinates where the given pixels' sight-lines intersect the CAD model.

.. autofunction:: calcam.raycast_sightlines

.. autoclass:: calcam.RayData
	:members: