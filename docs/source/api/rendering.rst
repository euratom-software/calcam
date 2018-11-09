========================
Rendering Images
========================
You may wish to render images of the CAD model from the point of view of a calibrated camera, e.g. to provide illustrations of the camera view, to use as overlays to provide context when viewing data, etc. For this purpose Calcam provides the render module, containing functions which can produce renders of the camera's view including the effects of distortion in the real camera.

.. py:module:: calcam.render

.. py:function:: render_cam_view(CADModel,FitResults,filename=None,oversampling=1,AA=1,Edges=False,EdgeColour=(1,0,0),EdgeWidth=2,Transparency=False,ROI=None,ROIColour=(0.8,0,0),ROIOpacity=0.3,roi_oversize=0,NearestNeighbourRemap=False,Verbose=True,Coords = 'Display',ScreenSize=None)

	Render an image of the CAD model from the point of view of a calibrated camera, and optionally save the result to an image file.

	:param CADModel: The CAD model to render an image of.
	:type CADModel: :class:`calcam.CADModel`
	:param FitResults: The camera calibration defining the view you want to render from.
	:type FitResults: :class:`calcam.CalibResults`
	:param str filename: Image file name (including extension) to save the results. If not specified, the results are not saved to disk.
	:param float oversampling: Over-sampling factor. The image is rendered at the resolution of the real camera multiplied by this factor. Values <1 can be used for under-sampling.
	:param int AA: Anti-aliasing amount. This smoothens the appearance of edges in the image by rendering at a higher resolution then downsampling it to the final input. Since the size of intermediate images stored goes as the square of this factor, it can become slow and memory intensive very quickly.
	:param bool Edges: Whether to render the CAD as a solid body or wireframe-style. Setting this to True renders in wireframe.
	:param tuple EdgeColour: If using Edges=True, specifies what colour the wireframe lines should be. This should be a tuple of 3 numbers between 0 and 1 specifying the R,G,B colour.
	:param float EdgeWidth: If using Edges=True, this specifies the line thickness for the wireframe render in pixels.
	:param bool Transparency: If set to True, the (black) background of the image will be set to transparent and an RGBA image will be returned (useful e.g. for creating wireframe overlays). Otherwise a black background is used and an RGB image returned.
	:param ROI: Region-of-interest or region-of-interest set to include in the render as a shaded region(s). ROI documentation to come later
	:type ROI: :class:`calcam.roi.ROI` or :class:`calcam.roi.ROISet`
	:param tuple ROIColour: If including an ROI in the render, what colour to shade the ROI. Tuple of 3 values between 0 and 1 specifying the R,G,B colour.
	:param float ROIOpacity: If including an ROI in the render, the opacity to render the ROI between 0 (transparent) and 1 (solid colour).
	:param float roi_oversize: Can be used to expand the size of a rendered ROI by approximately this many pixels.
	:param bool NearestNeighbourRemap: When applying the image distortion, whether to use nearest neighbour rather than the defauly cubic interpolation. I can't remember why this is an option or if it's actually ever used.
	:param bool Verbose: Whether or not to print status messages while performing the render.
	:param str Coords: 'Display' or 'Original', whether to render the image in original or display orientation.
	:param tuple ScreenSize: Resolution of the current display in pixels. This is usually unnecessary and should only be specified if things don't work properly without it.
	:return: Array containing the rendered RGB or RGBA image. Also saves the result to disk if the filename parameter is set.

.. py:function:: render_material_mask(CADModel,FitResults,Coords='Display')

	Create a mask describing what material (according to the CAD model configuration) each pixel is looking at. This is useful e.g. for infra-red cameras where the scene may consist of materials with different emissivities, by creating a mask which can be used to apply the correct emissivity to the correct image regions.

	:param CADModel: The CAD model to create the mask for.
	:type CADModel: :class:`calcam.CADModel`
	:param FitResults: The camera calibration to create the mask for.
	:type FitResults: :class:`clacm.CalibResults`
	:param str Coords: 'Display' or 'Original', whether to render the mask in Display or Original orientation.
	:return: A tuple of (material_list, material_mask). material_list is a list of strings containing the names of all the materials which the camera can see (as defined in the CAD model definition). material_mask is an array the same shape as the camera image containing integer indexes in to material_list which specify what material each pixel is looking at.