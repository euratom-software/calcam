**********************
Concepts & Conventions 
**********************


Image Pixel Coordinates
-----------------------
Calcam follows the convention of using matrix/linear algebra style pixel coordinates for images, which is consistent with the way images are stored and addressed in 2D arrays in Python. In this convention, the origin ``(0,0)`` is in the centre of the top-left pixel in the image. The *y* axis runs from top to bottom down the image and the *x* axis runs horizontally left to right. 

It is important to note that since 2D arrays are indexed ``[row, column]``, arrays containing images are indexed as ``[y,x]``. However, Calcam functions which deal with image coordinates are called as ``function(x,y)``. This is consistent with the way image coordinates are delt with in OpenCV.


"Original" and "Display" Coordinates
------------------------------------
In camera diagnostics, the data comes out of the camera with some intrinsic orientation which may not be the "right way up" in terms of easily looking at or understanding the images, e.g. if the camera is mounted upside down or the image is flipped by a mirror on the optical path. It is therefore desirable to work with the image transformed to be the "right way round". However, when performing bulk, programmatic analysis of video data it is not necessary or advantageous to perform this transformation, and is more efficient to work with the raw data as it comes out of the camera or is stored. Calcam therefore uses the concept of "display" and "original" image coordinates (thanks to Valentina Huber for this idea, and coining the phrase). Consider an object at a specific point in an image. Its coordinates in the camera image, as read straight from the camera, are the "original" coordinates. If the image is transformed to be the "right way up", the object's coordinates in the image will now be different. These coordinates in the "right way up" image are called "Display" coordinates. Calcam keeps track of the transofmration between original and display coordinates, so that the results can be used with raw data as it comes from the camera as well as processed "right way up" data. It is therefore highly recommended that you do not perform any geometrical transformations to images before loading them in to calcam for calibration, but make all these adjustments within calcam. This will leave the option open for using the raw data more efficiently in your analysis.

By default, calcam functions work with display coordinates unless specified otherwise. This is because the underlying openCV camera model fitting requires that the image being calibrated is not flipped, horizontally or vertically (this would add a minus sign to the pinhole projection which is not supported by the code). This is also an important point when calibrating images: ensure the image is not horizontally or vertically flipped when performing calibrations!


Sub-views
---------
Some camera diagnostic systems include optics such that multiple views share the same detector. For example, systems including a mirror or prism which result in two distinct camera views each occupying different parts of the image. To deal with these cases, Calcam has the concept of sub-views. A sub-view is an individual camera view, and an image can consist of one more more sub-views each occupying different parts of the image. Each sub-view is individually calibrated, and a mask is used to keep track of which image pixels belong to each sub-view. 

.. _cadmodel_intro:

CAD Models
----------
In Calcam, a CAD model consists of a collection of different 3D mesh files plus metadata. A CAD model can have multiple variants, which can be used to represent different machine configurations or to provide different levels of detail in the mesh files to trade off with performance. Each variant consists of model features or parts, each of which comes from a single 3D mesh file and represents part of the machine. The features can be collected in to groups to make models with many features easier to work with.