================================
Calibration Tool (Point Fitting)
================================
This is the main GUI tool for performing camera calibrations. An image from a camera to be calibrated and a CAD model of the camera's view are shown side-by-side, and the user then identifies pairs of corresponding points between the image and CAD model. These point pairs are then used to fit the camera calibration.

A screenshot of the calibration tool GUI as it opens is shown below. The large black area on the left-hand side is used to display the image and CAD view, and other inputs and controls are organised on a series of tabs on the right, with the tabs in the order they would typically be needed to perform a calibration. The tabs with the controls can be shown or hidden with the :guilabel:`<<Hide Controls` / :guilabel:`Show Controls>>` button to the left of the tabs, leaving the image and CAD model to take up the whole window. This can be useful on machines with lower screen resolutions to get a closer view.


Loading & Adjusting an Image to Calibrate
-----------------------------------------
At the top of the :guilabel:`Image` tab is a group of controls for loading an image you want to calibrate. The :guilabel:`From` dropdown list selects from where you want to load the image. The options available as standard are loading from a file (default) or loading an image which has previously been used in Calcam. If you define any custom image sources e.g. loading diagnostic data from a data store (see here), they will also appear in this dropdown menu. Once an image source is selected, the relevant inputs to set up the image loading appear below the dropdown list. Once the relevant fields are completed, click the :guilabel:`Load` button to load the image.


Image Adjustments
~~~~~~~~~~~~~~~~~
With an image loaded, a :guilabel:`Current Image` section appears on the :guilabel:`Image` tab, containing information and settings for the current image. Controls include

* **Known Pixel Size**: If you know the pixel size of the camera, it can be entered here. This does not make any difference to the calibration except that focal lengths can be displayed in mm instead of pixels, which can be useful for sanity checking results e.g. comparing with optical designs or known lenses.

* **Geometrical Transformations**: Controls for transforming the image.

* **Histogram Equilise** (only available with OpenCV 3.0+): Toggle adaptive histogram equilistion on the image display; this can make it easier to identify low contrast image features.

Image Mouse Navigation
~~~~~~~~~~~~~~~~~~~~~~
You can interactively zoom and pan the image with the following mouse controls:

- :kbd:`Scroll Wheel` - Zoom in or out.
- :kbd:`Middle Click + Drag` - Drag the image around.


Split Field-of-view Images
~~~~~~~~~~~~~~~~~~~~~~~~~~
Calcam has support for camera systems with multiple fields-of-view on a single detector, e.g. systems with split mirrors at the front so that different prats of the image look in different directions. In Calcam the different parts of the image are referred to as "sub-fields". If loading a split-field image, in order to define which pixels belong to which field, use the :guilabel:`Define Split Image...` button at the bottom of the image controls. This opens a dialog box which allows you to define the split field-of-view.



Loading and manipulating a CAD model
------------------------------------
CAD models are loaded using the :guilabel:`CAD Model` tab, which provides dropdown boxes to select the CAD model to load and the model variant (see here for how to configure these). Clicking the :guilabel:`Apply` button loads the selected model or changes the current model or variant to the selected one.

Turning CAD Features On/Off
~~~~~~~~~~~~~~~~~~~~~~~~~~~
For CAD models composed of multiple parts in separate mesh files, individual mesh files can be turned on or off by ticking/unticking them in the :guilabel:`Enable / Disable Features` panel. Buttons below this panel allow you to enable or disable all features together.

Usually all the features enabled by default in the CAD model definition are loaded automatically when clicking the :guilabel:`Apply` button. If you do not want this to happen, e.g. if using a slow computer where you only want to load select parts individually, untick the :guilabel:`Load default features on Apply` checkbox and the CAD model definition will be loaded without loading any actual mesh data.

CAD View Tab
~~~~~~~~~~~~
With a CAD model loaded, the :guilabel:`CAD View` tab is enabled and can be used to quickly change the CAD viewport. The main feature on this tab is a tree of CAD views including those defined in the CAD model definition, and those defined by existing calibrations and virtual calibrations. Clicking on a view in this pane immediately changes the CAD viewport to that view. You can also manually input coordinates for the CAD view to look from and to using the boxes at the bottom of this tab.

CAD Mouse Navigation
~~~~~~~~~~~~~~~~~~~~
You can interactively navigate around the CAD model using the following mouse controls:

- :kbd:`Right Click + Drag` - Look around (first-person shooter style control)
- :kbd:`Middle Click + Drag` - Pan (translate) the camera in the plane of the screen.
- :kbd:`Scroll Wheel` - Dolly the camera forwards or backwards.
- :kbd:`Ctrl + Scroll Wheel` Reduce or increase the CAD field-of-view angle (i.e. Zoom)


Defining Point Pairs
--------------------
Calcam uses *point pairs* to perform the calibration, where a point pair consists of one point on the CAD model and its corresponding point on the image. Point pairs are displayed on the CAD and image views as red **+** cursors at the point locations. At any given time, one point pair can be selected for editing. The selected point pair will be indicated with larger green **+** cursors. 

Once you have identified a common feature on the image and CAD model, :kbd:`Ctrl + Click`  on the location on either the image or CAD view to create a new point pair. A point will be placed at the mouse location. Then click (without holding :kbd:`Ctrl` the corresponding point on the other view to finish creating the point pair. You should now see green cursors on both the CAD model and image. Clicking either the CAD model or image again will move the green cursor representing the current point to the clicked location. To start another point pair, :kbd:`Ctrl + Click` again and repeat the process. The cursors showing the existing points will turn red, indicating they are no longer selected. In general, left clicking on either the image or CAD model will move the currently selected point to the clicked location. Clicking an existing cursor will select that point pair for editing, and holding :kbd:`Ctrl` while clicking will start a new point pair.

If you start a new point pair but do not specify both CAD and image points (e.g. by :kbd:`Ctrl+Click` on the image twice in a row), this will create "un-paired" points which will be ignored when setting up the fitting. The cursors of these points will be displayed in yellow. You can go back to these later to add their corresponding point by clicking on the yellow cursor to select it, then clicking on the corresponding point in the other view. 

The currently selected point pair can be deleted by pressing the :kbd:`Del` key on the keyboard, or clicking the :guilabel:`Remove current point pair` button on the :guilabel:`Points` tab.

You can see the coordinates of the points in the selected pair also in the :guilabel:`Points` tab.


Loading & Saving Point Pairs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Options for loading and saving point pairs are found on the :guilabel:`Points` tab. A dropdown box at the top of this tab lists previously saved point pairs which can be loaded. Only point pairs for images with the same pixel dimensions as the current image are shown in this list. By default, loading saved point pairs will replace any existing point pairs in the editor with the ones from the save file. To instead add the points from the save file to any existing ones, untick the :guilabel:`Clear existing points first` box.

Below the options for loading is a button to save the current points, clicking on which will prompt for a name to save as (the default is the name of the current image).


Using Additional Intrinsics Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In many examples of real images, only a relatively small number of point pairs can be accurately identified. Due to the large number of free parameters in the camera model fit (focal length(s), centre of perspective and distortion parameters) this can often give rather poor quality results. It is possible to better constrain the fits by using images of a chessboard pattern, with known square size, taken with the same camera & optical system configuration. This adds aditional constraints on the intrinsic model parameters, meaning only enough points to reliably fit the extrinsic parameters need to be identified in the image of the machine. Clicking the :guilabel:`Load Chessboard Images...` button in the :guilabel:`Points` tab opens a dialog box where you can load chessboard images to use as additional constraints, and provides instructions within the interface. Once chessboard images have been loaded, they can be included or excluded from the fitting using the checkbox next to the button. This user guide may later include guidelines on preparing chessboard images; for the time being, suitable advice can be found by searching for OpenCV camera calibration with chessboard images or the MATLAB camera calibration toolbox.

Doing the Calibration
---------------------
Once enough point pairs have been identified, the calibration itself and checking and saving the results are done using the :guilabel:`Fitting` tab. The :guilabel:`Do Fit` button will enabled when there are enough data points to constrain the number of free parameters in the camera model, based on the current fitting options. This is an absolute minimum and using as many points as possible, distributed as widely as possible around the image, is always recommended. However in images where very few points can be identified, disabling some model parameters can allow fitting with very few points - see fitting options below. 

Fitting Options
~~~~~~~~~~~~~~~
Fitting & camera model options are adjusted using the top section on the :guilabel:`Fitting` tab. The default options will typically produce good results for most images, however in some cases they will need to be adjusted to get a good quality result. For images with split fields of view, since each sub-field is fitted separately, tabs are displayed containing independent controls for each sub-field's fit options.

The first option to choose is whether to use the perspective or fisheye projection model: these two can be switched using the radio buttons at the top of the fit options section. The detailed options presented then depend on which model is selected:

Perspective Model Fit Options:


- :guilabel:`Disable k1...k3` These options, when checked, cause the corresponding coefficients in the distortion model to be fixed at 0 in the fit. This changes the order of the radial distortion model (and disables radial distortion entirely if all three are checked). Disabling higher order radial distortion terms can improve fits when the point pairs do not sufficiently constrain the distortion model, when the fitted results can have large erroneous distortions.
- :guilabel:`Disable Tangential Distortion` This option sets the coefficients :math:`p_1` and :math:`p_2` in the distortion model to be fixed at 0 in the fit, i.e. disables tangential distortion in the fitted model. This can be helpful if the fitting results in large erroneous values of these coefficients.
- :guilabel:`Fix Fx = Fy` This option fixes the focal lengths in the horizontal and vertical directions to be equal, i.e. fixes the image aspect ratio to 1. This is enabled by default, since for square pixels and non-anamorphic optics, which is the typical case, :math:`f_x = f_y` is expected. Un-checking this option can sometime help fit quality for some optical systems.
- :guilabel:`Fix Optical Centre at..` This option fixes the location of the centre of perspective at the specified pixel coordinates. I'm not sure why you would ever want to use this, but since it's possible in the underlying OpenCV fitting, I thought I'd include the option.
- :guilabel:`Initial Guess for Focal Length` This is the initial guess for the focal length used when starting the fit. The OpenCV fitter seems quite robust to values far from the final result, and the default value has been chosen to work well for most test images. However, there may be some cases where it is desirable to manually set the initial guess for the focal length for the fitter to find the correct solution.


Fisheye Model Fit Options


- :guilabel:`Disable k1...k4` These options, when checked, cause the corresponding coefficients in the distortion model to be fixed at 0 in the fit, changing the order of the fisheye distortion model.
- :guilabel:`Initial Guess for Focal Length` This is the initial guess for the focal length used for the fitting. The OpenCV fitter seems quite robust to values far from the final result, and the default value has been chosen to work well for most test images. However, there may be some cases where it is desirable to manually set the initial guess for the focal length.

To perform a fit using the current fit options, click the :guilabel:`Do Fit` button underneath the fit options. Alternatively, the keyboard shortcut :kbd:`Ctrl + F` also performs a fit with the current settings.

Checking fit quality & saving
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
As soon as a fit is performed, the fitted points are shown on the image as blue  **+** cursors. These are the current CAD model points converted to image coordinates using the fitted model, i.e. for a good fit these should lie on top of the user-placed image points. Display of the fitted points can be turned on or off using the :guilabel:`Show Fitted Points` checkbox, or pressing :kbd:`Ctrl + P` on the keyboard. The RMS fit error and fitted extrinsic and intrinsic parameters (camera pupil position and view direction, field of view, focal length, centre of perspective and distortion parameters) are dislayed on the lower part of the :guilabel:`Fitting` tab. As with fit options, if the image has a split field-of-view, results for each field of view are shown on separate tabs. Note: for fits with small numbers of points, the camera model has sufficiently many free parameters that a very small RMS fit error and good looking fitted point positions can be obtained with a fit which is actually very bad!

A much more robust, and highly recommended, visual check of the fit quality can be obtained by overlaying the CAD model wireframe on top of the camera image, according to the fit results. This is be done by ticking the :guilabel:`Show wireframe overlay` box, or pressing :kbd:`Ctrl + O` on the keyboard. The CAD model is then rendered in wireframe and superimposed on the image. Note: for large images or CAD models this can be somewhat slow and memory intensive. 

Another way to quickly get an idea of the fit quality is to set the current CAD viewport to approximate the fitted model using the :guilabel:`Set CAD view to match fit` button, which can also be helpful when trying to identify further point pairs.

Once a satisfactory fit has been obtained, the results can be saved using the :guilabel:`Save As...` button at the bottom of the panel. This will also save the point pairs used for the fit, if this has not been done already, and will prompt for save names for both the points and fit. 

Once a satisfactory fit has been saved, this completes the calibration process using the calibration GUI.