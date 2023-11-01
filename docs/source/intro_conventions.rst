**********************
Concepts & Conventions 
**********************


Image Pixel Coordinates
-----------------------
Calcam follows the convention of using matrix/linear algebra style pixel coordinates for images, which is consistent with the way 2D arrays are handled by default in Python. In this convention, the origin ``(0,0)`` is in the centre of the top-left pixel in the image. The *y* axis runs from top to bottom down the image and the *x* axis runs horizontally left to right.

It is important to note that since 2D arrays are indexed ``[row, column]``, arrays containing images are indexed as ``[y,x]``. However, Calcam functions which deal with image coordinates are called as ``function(x,y)``. This is consistent with the way image coordinates are delt with in other Python libraries including OpenCV.


"Original" and "Display" Pixel Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In camera diagnostics, the data comes out of the camera with some intrinsic orientation which may not be the "right way up" in terms of easily looking at or understanding the images, e.g. if the camera is mounted upside down or the image is flipped by a mirror on the optical path. It is therefore desirable to work with the image transformed to be the "right way round". However, when performing bulk, programmatic analysis of video data it is not necessary or advantageous to perform this transformation, and is more efficient to work with the raw data as it comes out of the camera or is stored. Calcam therefore uses the concept of "display" and "original" image coordinates (thanks to Valentina Huber for this idea, and coining the phrase). Consider an object at a specific point in an image. Its coordinates in the camera image, as read straight from the camera, are the "original" coordinates. If the image is transformed to be the "right way up", the object's coordinates in the image will now be different. These coordinates in the "right way up" image are called "Display" coordinates. Calcam keeps track of the transofmration between original and display coordinates, so that the results can be used with raw data as it comes from the camera as well as processed "right way up" data. It is therefore highly recommended that you do not perform any geometrical transformations to images before loading them in to calcam for calibration, but make all these adjustments within calcam. This will leave the option open for using the raw data more efficiently in your analysis.

By default, calcam functions work with display coordinates unless specified otherwise. This is because the underlying openCV camera model fitting requires that the image being calibrated is not flipped, horizontally or vertically (this would add a minus sign to the pinhole projection which is not supported by the code). This is also an important point when calibrating images: ensure the image is not horizontally or vertically flipped when performing calibrations!

.. _subviews_intro:

Detector / Image Offset
~~~~~~~~~~~~~~~~~~~~~~~
Many scientific CMOS cameras can be configured to readout only a specific region-of-interest (ROI) within the image sensor area to achieve higher frame rates. In such cases, the recorded image area can be specified by the width and height in pixels, and the offset in pixels of the top-left corner of the recorded image from the top-left corner of the whole detector. Calcam supports keeping track of this offset when working with images from this type of camera. This allows, for example, a camera to be calibrated for one readout ROI, but then the same calibration can still be used if the ROI setting is changed. This requires that you tell calcam the camera ROI offset settings (top left of the ROI relative to the top left of the whole detector) when calibrating or analysing such images. The offset is always specified in original coordinates (as it would be in the camera settings) and in Calcam is referred to as the image offset or detector offset. If using a camera where the readout ROI never changes, there is no need to pay attention to these settings or values.

Sub-views
---------
Some camera diagnostic systems include optics such that multiple views share the same detector. For example, systems including a mirror or prism which result in two distinct camera views each occupying different parts of the image. To deal with these cases, Calcam has the concept of sub-views. A sub-view is an individual camera view, and an image can consist of one more more sub-views each occupying different parts of the image. Each sub-view is individually calibrated, and a mask is used to keep track of which image pixels belong to each sub-view. 

.. _cadmodel_intro:

CAD Models
----------
In Calcam, a CAD model consists of a collection of different 3D mesh files plus metadata. The current version supports importing ``.stl`` or ``.obj`` format 3D mesh files. It's usually convenient to split the model in to several individual mesh files containing different parts of the scene, and these can then be turned on or off individually when working with the model. The features can be collected in to groups to make models with many features easier to work with. Calcam packages all the mesh files in to a custom zipped file format (.ccm) along with various metadata to create a Calcam CAD model file. You can have several such files and easily switch between them at any time. A CAD model definition can have multiple variants, which can be used to represent different machine configurations (e.g. if some machine features were changed over time), or to provide different levels of detail in the mesh files to trade off with performance.

3D coordinate system
~~~~~~~~~~~~~~~~~~~~~
For working with machine CAD, Calcam uses a right-handed cartesian coordinate system with the vertical 'up' direction along the +Z axis.