********************************
Working with Calibration Results
********************************

Once you have created calibrations using the Calcam GUI tools, you can import calcam as a Python package and use it to extract the information you want from the calibrations and/or integrate use of the results in to your workflow. This page documents the features of the Calcam API for working with calibration results.

For examples Python code, skip down the page to Examples.

.. autoclass:: calcam.Calibration
	:members: get_cam_matrix,get_cam_roll,get_cam_to_lab_rotation,get_fov,get_image,get_los_direction,get_pupilpos,project_points,undistort_image