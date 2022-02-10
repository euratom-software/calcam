================
Rendering Images
================

Calcam can be used to render images of a CAD model from a calibrated camera's point of view. This includes all lens distortion in the camera model, so for calibrated real camera images, the rendered images should match exactly with the camera images. The appearance of the CAD model (colour, wireframe, which parts are loaded etc) is configured using the features of the :doc:`api_cadmodel` class.

For examples of using these features, see the :doc:`api_examples` page.

.. autofunction:: calcam.render_cam_view

.. autofunction:: calcam.render_unfolded_wall
