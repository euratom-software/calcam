===========
Ray Casting
===========
While the :class:`calcam.Calibration` class can provide information about the camera line-of-sight geometry based on a calibration, it is often necessary to also know where these sight lines intersect with a surface in the CAD model. For example for infrared thermography cameras, the mapping of image pixels to positions on the physical surfaces (of the CAD model) is usually of most interest. The function :func:`calcam.raycast_sightlines` is provided for this purpose; it determines the 3D coordinates where the given pixels' sight-lines intersect the CAD model. Results of these calculations are represented by the :class:`calcam.RayData` class. Both of these are documented on this page. For examples of usage, see the :doc:`api_examples` page.

.. autofunction:: calcam.raycast_sightlines

.. autoclass:: calcam.RayData
	:members:
