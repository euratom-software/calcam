============================
Camera Movement Correction
============================

Background
----------
A common issue to deal with is that after a camera has been spatially calibrated, the camera then moves and needs to be re-calibrated. The calcam.movement module contains tools for dealing with this. The problem is framed as movement of the camera image (a "moved image") with respect to a "reference image" e.g. an image that has already been calibrated. There are two approaches to deal with this:

* Apply transformations to warped images, or pixel coordinates in warped images, to make them conform to a reference calibration.

* Update the reference calibration to apply to the moved image.

Which approach is preferable will depend on how you are using calcam results. See the detailed documentation below for details of how these methods are implemented in calcam.movement, or see the :doc:`api_examples` page for examples of how to use this functionality.

Determining Camera Movement
---------------------------
Calcam can try to detect the movement between two images automatically, or there is a GUI tool which allows manual feature matching by the user or automatic detection but user validation.

.. autofunction:: calcam.movement.detect_movement

.. autofunction:: calcam.movement.manual_movement

The MovementCorrection class
----------------------------
The above functions return movement correction objects, which represent the geometrical transform between the reference and moved images. These objects then have various methods for transforming between reference and moved images and coordinates.

.. autoclass:: calcam.movement.MovementCorrection
    :members:

Adjusting calibrations for image movement
-----------------------------------------
Having obtained a movement correction object describing the movement, you can update an exicting calcam calibration to apply to the new image:

.. autofunction:: calcam.movement.update_calibration
