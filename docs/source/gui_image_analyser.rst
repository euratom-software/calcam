==============
Image Analyser
==============
The Calcam image analyser tool provides a convenient interactive way to convert between 2D image positions and 3D real space positions, given a Calcam calibration. While this is only the most simple use of camera spatial calibration, it can be convenient to have a quick, easy way to quickly find the correspondance between image and 3D coordinates.

The image analyser window is shown below:

.. image:: images/screenshots/image_analyser_annotated.png
   :alt: Image analyser tool screenshot
   :align: left


Loading an image to analyse
---------------------------
At the top of the :guilabel:`Image and Calibration` control tab is a group of controls for loading an image you want to analyse. The :guilabel:`From` dropdown list selects the source from which you want to load the image. The options available as standard are loading from an image file (default) or loading an image from another Calcam calibration. If you define any custom image sources (see :doc:`dev_imsources`), they will also appear in this dropdown menu. Once an image source is selected, the relevant inputs to set up the image loading appear below the dropdown list. Once the relevant fields are completed, click the :guilabel:`Load` button to load the image. Note: when the images is loaded it will always be displayed in its original orientation until a calibration is loaded.

Basic information about the current image id displayed in the :guilabel:`Image` box. This also provides an option to apply histogram equilisation to improve the contrast of image features.


Loading a Calibration
---------------------
The calibration to use for the analysis is also controlled on the :guilabel:`Image and Calibration` control tab. To load a calibration to use for the analysis, click the :guilabel:`Load...` button in the :guilabel:`Calibration for analysis` box. This box also displays the name of the currently loaded calibration, and full details of the calibration can be viewed by clicking the :guilabel:`Properties...` button in this box.

If the loaded calibration contains information about the CAD model settings last used when the calibration was edited, the applicable CAD model will be automatically loaded, if available. The CAD view will also be set to match the camera view automatically when the calibration is loaded.


Loading a CAD model
--------------------
If the CAD model is not loaded automatically when loading the calibration, or you want to change the CAD model used, this is controlled on the :guilabel:`Machine Model` control tab. At the top of this tab are controls for loading a CAD model. Two dropdown boxes are provided to select the CAD model to load and the :ref:`model variant <cadmodel_intro>`. The :guilabel:`Load` button to the right of the model variant then loads the model. Underneath the model variant dropdown box is a checkbox labeled `Enable default model features on load`: if checked, the default parts of the CAD model are all loaded as soon as the model itself is loaded. If not checked, the model definition will be loaded but none of the 3D mesh data will be loaded, and instead you can turn on features individually. This is useful if working with a large model on a slow computer where you don't want to wait for the whole model to load if not needed. You can change to a different CAD model or variant at any time by selecting a different item from the dropdown boxes and clicking :guilabel:`Load` again.


Turning CADmodel features On/Off
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For CAD models composed of multiple parts in separate mesh files, individual mesh files can be turned on or off by ticking/unticking them in the :guilabel:`Enable / Disable Features` panel. This can be done for individual parts, groups of parts (if defined in the CAD model) or the entire model. This can be helpful to improve performance when working with large models if not all parts are necessary for a particular calibration, or for removing parts which are in the way and make the calibration more difficult.


Performing Position Analysis
----------------------------
With an image, calibration and CAD model loaded, clicking on either the CAD model or image will place a green cursor at the clicked position. A green cursor will also appear at the corresponding point on the other view. Quantitative information about the current cursor position is displayed on the :guilabel:`Position Analysis` tab. 

If the cursor is placed on the CAD model at a position not visible in the image, the cursor turns red on the CAD model view. If the cursor position is within the image field of view but hidden behind part of the CAD model, a red cursor will also appear on the image at the position where that point would be if it was not hidden from view.

By default, the CAD model view will also show a solid green line in 3D representing the line of sight from the camera to the cursor position. A red line shows the continuation of the sight-line after it has passed the cursor. If the cursor is hidden from the camera's view, a green sight-line is shown up to the point where the sight-line hits the object hiding the cursor, then the remainder of the sight-line is shown in red.

The :guilabel:`Position Analysis` control tab has boxes showing details of the image coordinates, 3D coordinates and sight-line coordinates corresponding to the current cursor position. It also contains the following additional controls:

* :guilabel:`Show CAD wireframe overlay on image` checkbox: displays the CAD wireframe overlay on top of the image. This can be used to confirm the accuracy of the calibration or provide context for the image.
* :guilabel:`Show cursor close-up` button: If the cursor is visible to the camera, this button sets the 3D CAD view to show a close-up of the current cursor position.
* :guilabel:`Reset View` button: resets the 3D CAD view to match the image.
* :guilabel:`Show line-of-sight on 3D view` checkbox: turn on or off display of the camera sight-line on the 3D view. If turned off, only the cursor is shown.


Image mouse navigation
~~~~~~~~~~~~~~~~~~~~~~
The image can be manipulated at any time using the following mouse controls:

- :kbd:`Scroll Wheel` - Zoom in or out, centred at the current mouse position.
- :kbd:`Middle Click + Drag` - Drag the image around.


CAD view navigation
~~~~~~~~~~~~~~~~~~~~~~~~~
You can interactively navigate around the CAD model using the following mouse controls:

- :kbd:`Right Click + Drag` - Look around (first-person shooter style control; default) or rotate CAD model depending on settings
- :kbd:`Middle Click + Drag` - Pan (translate) sideways i.e. in the plane of the monitor.
- :kbd:`Scroll Wheel` - Move forwards or backwards.
- :kbd:`Ctrl + Scroll Wheel` Reduce or increase the CAD field-of-view angle (i.e. Zoom)
- :kbd:`Ctrl + Right Click + Drag` - Roll the camera (rotate about the direction of view)


In addition to the mouse controls, the :guilabel:`3D Viewport` tab  can be used to control the current view of the CAD model. At the top of this tab is a list of viewports defined in the CAD model definition. Clicking on a view in this pane immediately changes the CAD viewport to that view. You can also set the view to match an existing calibrated camera by clicking the :guilabel:`Add from calibration(s)...` button below the viewport list. You can then select one or more Calcam calibration files to load, and the views defined by the calibration will be added to the viewport list on the 3D Viewport tab. 

If you want to save the current view of the CAD model in to the model definition so you can easily return to it, enter a name in the :guilabel:`Name` box under the heading :guilabel:`Save current view as preset` and click :guilabel:`Save`. The view will then be added to the viewport list, and if the model definition you are using is not read-only, will be saved to the model definition for future use.

Near the bottom of the tab are editable boxes showing the current viewport's camera position, camera view target, field of view and roll. These update automatically to reflect the current viewport, and you can manually set up the CAD view by editing these.

At the bottom of this tab are options for configuring the mouse controls for CAD navigation. The :kbd:`Right Click + Drag` behaviour can be toggled between looking around and rotating the model about a point in front of the camera, and the mouse sensitivity can be adjusted.