===============================
Calibration by manual alignment
===============================

While the best and most accurate calibration technique is :doc:`gui_calib`, in some cases this is simply not possible because well defined points cannot be identified in the image (e.g. narrow angle camera views which can see only round or curved features on the machine). In this case, Calcam provides a tool for calibration by manually moving the camera view with the mouse until it lines up with the camera image to be calibrated. This will be much more successful if the camera intrinsics can be calibrated accurately, e.g. using chessboard calibration images in the lab, leaving only the extrinsics to be determined "by hand". Manual calibration only currently supports images with a single sub-view. The manual alignment calibration tool can be started from the Calcam launcher. The layout of the window is shown below:

.. image:: images/screenshots/alignment_calib_annotated.png
   :alt: Calibration tool screenshot
   :align: left

Loading an Image to Calibrate
-----------------------------------------
At the top of the :guilabel:`Camera Image` control tab is a group of controls for loading an image you want to calibrate. The :guilabel:`From` dropdown list selects the source from which you want to load the image. The options available as standard are loading from an image file (default) or loading an image from another Calcam calibration. If you define any custom image sources (see :doc:`dev_imsources`), they will also appear in this dropdown menu. Once an image source is selected, the relevant inputs to set up the image loading appear below the dropdown list. Once the relevant fields are completed, click the :guilabel:`Load` button to load the image. The image is then displayed in the CAD + Image display on the left of the window. It is not possible to zoom or pan on the image; it is always fit to the window. The CAD model will be displayed behind the image.


Current Image Settings
-----------------------
With an image loaded, the :guilabel:`Current Image` section appears on the :guilabel:`Camera Image` tab, containing information and settings for the current image. Controls include

* **Known Pixel Size**: If you know the pixel size of the camera, it can be entered here. This does not make any difference to the calibration except that focal lengths can be displayed in mm instead of pixels, which can be useful if you know the expected effective focal length of the camera optics.
* **Geometrical Transformations**: Controls for transforming the image to get it the "right way up". It is recommended to always load images in to Calcam the way they come out of the camera as raw, then use these controls to get the image right-way-up for calibration. The :guilabell:Stretch Vertically by' button is provided for cameras with non-square pixels or anamorphic optics.

Image display effects
~~~~~~~~~~~~~~~~~~~~~
Applying effects to the image can make it easier to align the image and CAD. The :guilabel:`Image Display` box appears below the :guilabel:`Current Image` box when an image is loaded. Settings available are:
* **No effect**: Display the image, without further processing, semi-transparently on top of the CAD view.
* **Histogram equilisation**: Apply local histogram equilisation to improve the contrast of image features.
* **Edge Detection**: Apply a Canny edge detector to the image and display the detected edges only overlaid on the CAD image. When this option is selected, the following extra controls are available: sliders for the edge detection thresholds which can be adjusted to improve the edge detection, and a colour picker to choose the colour of the displayed edges.
* **Display opacity**:  For manual calibration, the image is displayed semi-transparently over the CAD model. When no effect or histogram equilisation are detected, this slider controls the image opacity: slide to the left to make the image less visible, slide to the right to make the image more opaque over the CAD. This slider is not displayed when edge detecion is used, when the edges are displayed at full opacity.


Loading a CAD model
-------------------
At the top the :guilabel:`Machine Model` control tab are controls for loading a CAD model to calibrate against. Two dropdown boxes are provided to select the CAD model to load and the :ref:`model variant <cadmodel_intro>`. The :guilabel:`Load` button to the right of the model variant then loads the model. Underneath the model variant dropdown box is a checkbox labeled `Enable default model features on load`: if checked, the default parts of the CAD model are all loaded as soon as the model itself is loaded. If not checked, the model definition will be loaded but none of the 3D mesh data will be loaded, and instead you can turn on features individually. This is useful if working with a large model on a slow computer where you don't want to wait for the whole model to load if not needed. You can change to a different CAD model or variant at any time by selecting a different item from the dropdown boxes and clicking :guilabel:`Load` again.

Turning CAD model features On/Off
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For CAD models composed of multiple parts in separate mesh files, individual mesh files can be turned on or off by ticking/unticking them in the :guilabel:`Enable / Disable Features` panel. This can be done for individual parts, groups of parts (if defined in the CAD model) or the entire model. This can be helpful to improve performance when working with large models if not all parts are necessary for a particular calibration, or for removing parts which are in the way and make the calibration more difficult.

CAD model rendering type
~~~~~~~~~~~~~~~~~~~~~~~~
In some cases it may be easier to judge the alignment between image and CAD model if the model is rendered in wireframe / outline rather than the usual solid body appearance. For this purpose, at the bottom of the :guilabel:`Machine Model` tab, the CAD model appearance can be switched between solid body and wireframe outline. In addition, the colour of the selected CAD model part can be set.


Performing the alignment calibration
------------------------------------
In contrast to point fitting calibrations, for manual alignment calibrations the camera intrinsics (focal length & distortion) are set separately from the extrinsics (position and viewing direction). The sections below explain how to set each.

Camera Intrinsics
~~~~~~~~~~~~~~~~~
The camera intrinsics are set using the top part of the :guilabel:`Alignment Calibration` control tab. Camera intrinsics can be set 3 different ways: using intrinsics from an existing calibration (e.g. to calibrate an existing camera & lens setup moved to a new view), using chessboard calibration pattern images from lab measurements, or using a simple pinhole camera model. It is highly recommended to use chessboard images, if possible, or another calcam calibration since this is likely to give much better results and will probably be easier.

Calibration Intrinsics
**********************
To use intrinsics from an existing Calcam calibration, select :guilabel:`Use intrinsics from existing calibration` and browse for the calibration you want to use. The loaded calibration can be changed using the :guilabel:`Load...` button.

Chessboard Calibration Intrinsics
*********************************
To prepare chessboard images: make a flat chessboard target with known square size (there are various printable PDFs available by searching online). Then take a number of images with this chessboard target in front of the camera at a variety of positions, orientations and distances to the camera. The example below shows thumbnails of a set of chessboard calibration images:

.. image:: images/chessboard_example.png
   :alt: Chessboard image example thumbnails
   :align: left

To use the chessboard images to define the camera intrinsics, select :guilabel:`Calibrate from chessboard images`. tab. The first time this option is selected it will open the following window:

.. image:: images/screenshots/chessboard_intrinsics_dialog.png
   :alt: Chessboard dialog screenshot
   :align: left

Chessboard loading consists of 4 steps, done in order by working down the right hand side of this window. First, browse for and select all of the chessboard images to use. Then, enter the details of the chessboard pattern: number of squares and square size. Next, select the :guilabel:`Detect Chessboard Corners` button to run an automatic detection of the boundaries between the chessboard squares. If the automatic detection fails on some images, a dialog box will open telling you which images the detection failed for, and that those cannot be used. If all images fail, check that the number of squares input is correct. Once the corner detection has been completed, cursors will be added to the image displayed on the left hand side of the window. You can pan and zoom to inspect the cursor positions using the usual image mouse controls, and look at different images using the :guilabel:`<<` and :guilabel:`>>` buttons above the image. Finally, select whether to use the perspective distortion model or fisheye distortion model. To complete loading of the images and use these to define the camera intrinsics constraints, click :guilabel:`Apply`.