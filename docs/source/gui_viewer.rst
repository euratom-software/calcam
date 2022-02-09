=========
3D Viewer
=========
The Calcam 3D viewer can be used for visualising machine CAD models, camera positions and fields-of-view, and for creating high quality figures. As with the other Calcam GUI tools, the window consists of the 3D view on the left of the window and controls on tabs to the right:

.. image:: images/screenshots/viewer_annotated.png
   :alt: 3D Viewer Screenshot
   :align: left

Loading a CAD model
--------------------
At the top the :guilabel:`Machine Model` control tab are controls for loading a CAD model to calibrate against. Two dropdown boxes are provided to select the CAD model to load and the :ref:`model variant <cadmodel_intro>`. The :guilabel:`Load` button to the right of the model variant then loads the model. Underneath the model variant dropdown box is a checkbox labeled `Enable default model features on load`: if checked, the default parts of the CAD model are all loaded as soon as the model itself is loaded. If not checked, the model definition will be loaded but none of the 3D mesh data will be loaded, and instead you can turn on features individually. This is useful if working with a large model on a slow computer where you don't want to wait for the whole model to load if not needed. You can change to a different CAD model or variant at any time by selecting a different item from the dropdown boxes and clicking :guilabel:`Load` again.

Turning CAD model parts On/Off
------------------------------
For CAD models composed of multiple parts in separate mesh files, individual mesh files can be turned on or off by ticking/unticking them in the :guilabel:`Enable / Disable Features` panel. This can be done for individual parts, groups of parts (if defined in the CAD model) or the entire model. This can be helpful to improve performance when working with large models if not all parts are necessary for a particular calibration, or for removing parts which are in the way and make the calibration more difficult.

CAD model appearance settings
-----------------------------
The CAD model apprearance can be highly customised in order to create figures as desired. These options are in the :guilabel:`Appearance` box on the :guilabel:`Machine Model` control tab. This box contains controls to change the model appearance between solid body and wireframe / edges only. The colour of individual model feature(s) can also be changed by selecting the feature(s) in the :guilabel:`Enable / Disable Features` panel, and using the buttons in the :guilabel:`Appearance` box to change the part colour or reset it to the default value. You can also save the current colour configuration to be the default in the model definition by clicking :guilabel:`Save current colours as default (whole model)`.


Wall contour display
--------------------
If the CAD model dfinition includes an :math:`(R,Z)` wall contour, this can be displayed using the controls in the :guilabel:`Show R,Z Wall Contour` box. Here you can select between not showing the wall contour, showing it as a line at the toroidal position of the cursor, or revolving the R,Z contour in 3D to show a simplified, toroidally symmetric wall shape.

Mouse Navigation
-----------------
You can interactively navigate around the 3D view using the following mouse controls:

- :kbd:`Left Click` - Place a cursor on the 3D model, or move the cursor if one already exists. This will display information about the cursor position in the window status bar and can be used by cross-sectioning / wall contour display features.
- :kbd:`Right Click + Drag` - Look around (first-person shooter style control; default) or rotate CAD model depending on settings
- :kbd:`Middle Click + Drag` - Pan (translate) sideways i.e. in the plane of the monitor.
- :kbd:`Scroll Wheel` - Move forwards or backwards.
- :kbd:`Ctrl + Scroll Wheel` Reduce or increase the CAD field-of-view angle (i.e. Zoom)
- :kbd:`Ctrl + Right Click + Drag` - Roll the camera (rotate about the direction of view)


3D ViewPort Tab
---------------
In addition to the mouse controls, the :guilabel:`3D Viewport` tab  can be used to control the current view of the CAD model. In addition, this tab contains settings for the mouse controls and other options which can be used to change the rendering settings to adjust the appaerance of the model.

Rendering Settings
~~~~~~~~~~~~~~~~~~
At the top of the :guilabel:`3D Viewport` tab are settings which control the rendering style. The :guilabel:`3D Projection` options allow switching between a perspective projection view of the model and an orthographic view (where objects appear the same size regardless of their distance from the viewier). This can be helpful for making e.g. cross-section figures.

In the :guilabel:`Cross-Sectioning` box are options which allow the CAD model to be cross-sectioned in the view. If there is a cursor placed, the cross-section can be set to either cut through the cursor or through the origin. Cross-sectioning is turned on and off using the checkbox on the left of these options. Note: cross-sectioning is implemented by adjusting the clipping planes of the 3D rendering, so the cross-section is always cut in a plane whose normal is the viewing direction.

CAD Viewport Adjustment
~~~~~~~~~~~~~~~~~~~~~~~~
In the :guilabel:`Select pre-defined viewport` box is a list of viewports defined in the CAD model definition. Clicking on a view in this pane immediately changes the CAD viewport to that view. You can also set the view to match an existing calibrated camera by clicking the :guilabel:`Add from calibration(s)...` button below the viewport list. You can then select one or more Calcam calibration files to load, and the views defined by the calibration will be added to the viewport list on the 3D Viewport tab. In addition there are two :guilabel:`Auto Cross-Sections` views which will position the camera and set cross-sectioning options to view the entire model, cut in cross-section through the origin.

If you want to save the current view of the CAD model in to the model definition so you can easily return to it, enter a name in the :guilabel:`Name` box under the heading :guilabel:`Save current view as preset` and click :guilabel:`Save`. The view will then be added to the viewport list, and if the model definition you are using is not read-only, will be saved to the model definition for future use.

Near the bottom of the tab are editable boxes showing the current viewport's camera position, camera view target, field of view and roll. These update automatically to reflect the current viewport, and you can manually set up the CAD view by editing these.

Mouse Control Settings
~~~~~~~~~~~~~~~~~~~~~~
At the bottom of this tab are options for configuring the mouse controls for 3D navigation. The :kbd:`Right Click + Drag` behaviour can be toggled between looking around and rotating the model about a point in front of the camera, and the mouse sensitivity can be adjusted.


Visualising camera fields of view from calibrations
---------------------------------------------------
Calcam calibration files can be loaded by the 3D viewer to visualise the camera sight-lines / fields of view on the 3D model. This is done on the :guilabel:`Calibrations` tab. Clicking the :guilabel:`Add...` button below the list of currently loaded calibrations will open a file browsing dialog to load a calibration file. Once loaded, the calibrated camera's field of view is shown as a shaded volume in the 3D view. The sight-lines for a given calibration can be turned on or off using the tick-box next to its name. With multiple calibrations loaded, a legend showing which camera corresponds to which colour can be turned on or off using the checkbox next to the  :guilabel:`Add...` button. To change the name of a calibration in the legend, single-click the calibration name in the list box to edit the name.

Field-of-view appearance options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :guilabel:`Appearance` box at the bottom of the :guilabel:`Calibrations` tab allows customisation of how the selected camera field of view is displayed. In addition to controls for the transparency and colour, there are 3 different options for how fields of view appear:

* Shaded volume (default) - shows the camera field of view as a shaded cone eminating from the camera's pupil.
* Sight-line fan - shows a 2D fan of discrete sight-lines eminating out from the camera.
* CAD model shading (WARNING - slow to calculate!) - useful for cameras such as IR thermography cameras designed to observe objects which are part of the CAD model, this option shows the camera field of view by shading parts of the CAD model which the camera can see. Note: when turning on this option it can be quite slow to calculate since the calculation involves ray casting over the camera field of view.


Displaying Arbitrary 3D Data
-----------------------------
In addition to showing camera fields-of-view, additional arbitrary 3D points and lines can be added to the visualisation, e.g. representing other diagnostic sight-lines, trajectories, magnetic field lines, flux surfaces or other 3D coordinates of interest. Controls for this are on the :guilabel:`3D Data` tab. The 3D data are loaded from ASCII files with .txt, .csv or .dat extensions.

Once loaded, data sets are added to the list in the box on the :guilabel:`3D Data` tab, with names corresponding to the file name they were loaded from. The display of each data set can be turned on and off using the checkboxes next to the dataset names. To edit the name of a data set, single click its name in the list. 

Preparation of ASCII files
~~~~~~~~~~~~~~~~~~~~~~~~~~
The ASCII data to be displayed can take one of two formats, depending on whether you want to display a single continuous line defined by a set of 3D points, or a collection of individual straight lines each with a start and end point. To display a single continuous 3D line, the file should contain a list of 3D coordinates along the line, with each point along the line taking up one line of the text file. Each line of the text file must therefore contain 3 numbers, which can be delimited with commas, spaces or tabs. If you wish to display a set of disconnected line segments, each line of the text file must contain 6 numbers: the 3D coordinates for the start of the 3D line segment followed by the 3D coordinates for the end of the line segment. The coordinates can be given either in cartesian coordinates :math:`X,Y,Z` or :math:`R,Z,\phi` where :math:`\phi` is the toroidal angle in radians.

Appearance Options
~~~~~~~~~~~~~~~~~~
The 3D data can be displayed as solid lines and/or spheres at each point in the data. With a data set selected on the :guilabel:`3D Data` tab, the lines and points for that data set can be turned on and off, and their thickness, size and colour changed using the :guilabel:`Appearance` options at the bottom of the tab. Legend entries can also be shown for the loaded data sets, which can be turned on and off with the :guilabel:`Show in legend` checkbox below the list of loaded data sets. 


Rendering and saving images
---------------------------
The 3D viewer can be used to save high resolution PNG images using the controls on the :guilabel:`Render / Save Image` tab. At the top of this tab are the 3 main options for the types of image which can be saved:

Exporting the current view
~~~~~~~~~~~~~~~~~~~~~~~~~~~
If :guilabel:`Current View` is selected at the top of the :guilabel:`Render / Save Image` tab, the output image will exactly match what is currently displayed in the 3D view in the window. This can be used to prepare illustrative figures. By default the output image size in pixels will match the size of the window on the screen, however you can choose to render the output at higher resolution with the :guilabel:`Output resolution` dropdown box in the :guilabel:`Render Settings` box. Here you can also change the level of anti-aliasing to eliminate sharp edges in the rendered image, which is implemented by rendering the output at higher resolution than desired and then down-sampling again. You can also choose whether to use a black background, as in the display window, or to make any black areas of the image transparent in the output image. If there is a cursor placed in the current view, you can choose whether or not to show the cursor on the output image.

Rendering calibrated camera views
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The tool can also be used to render the CAD model, in the current configuration set up in the window, from the point of view of a calibrated camera. This will exactly match the position, orientation, focal length and distortion of the calibrated camera, so the rendered image should exactly match with real camera images to within the accuracy of the calibration. 

This is done by selecting :guilabel:`Calibration Result` at the top of the :guilabel:`Render / Save Image` tab. Then, click the :guilabel:`Select...` button in the :guilabel:`Render Settings` box to browse for a calibration file to use. You can then choose whether the output image should be in original or display orientation for the camera, and the output pixel resolution if you wish to render at a higher resolution than the real camera.  In the :guilabel:`Render Settings` box you can also change the level of anti-aliasing to eliminate sharp edges in the rendered image, which is implemented by rendering the output at higher resolution than desired and then down-sampling again. You can also choose whether to use a black background, as in the display window, or to make any black areas of the image transparent in the output image. If there is a cursor placed in the current view, you can choose whether or not to show the cursor on the output image. Note: If there is a sight-line legend displayed, this will not be included in the output image.

Rendering an un-folded first wall view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It may be useful to have an overview image of the first wall of the device, showing the entire wall in a single image. This can be done using the :guilabel:`Unfolded first wall` option at the top of the :guilabel:`Render / Save Image` tab. This option is only enabled for CAD models which include an :math:`R,Z` wall contour (see :ref:`wall_contour` for how to add this to to the CAD model).

The output of this type of render is an image of the first wall where toroidal angle increases along the horizontal direction of the image and poloidal angle increases in the vertical direction of the image. This can be useful e.g. for 

When the above settings are set as desired, click the :guilabel:`Render Image...` button to save an image file.
