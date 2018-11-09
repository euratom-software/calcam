========================
Working with CAD Meshes
========================

For image rendering and ray casting, you will have to work programmatically with CAD meshes. CAD meshes in Calcam are defined in python files as described in :doc:`../installation`. Each CAD model has its own class which is used to load and represent the model, and the names of these classes are defined by the user when defining the CAD model.

To load a CAD model, e.g. for a CAD model whose class is called mr_fusion::

	>>>cadmodel = calcam.machine_geometry.mr_fusion()

This will load the default variant of the mr_fusion model. If you have a model with multiple variants, the variant can be specified when loading the model::

	>>>cadmodel = calcam.machine_geometry.mr_fusion('1955 Model')

This object can then be used in ray casting or image rendering. If you want to adjust the appearance of the CAD model, the base CAD model class provides various methods for doing this which are documented below.

.. py:class:: calcam.CADModel

	Class representing a CAD model.

	This is the base class for Calcam CAD models and cannot be instantiated directly, instead see the above description of how to load a CAD model. The following methods are provided for manipulation of the CAD model object.


	.. py:method:: enable_features(features)
	
		Turn on specified features of the model.

		:param features: Name(s) of features to enable.
		:type features: str or list of str


	.. py:method:: disable_features(features)
		
		Turn off specified features of the model.

		:param features: Name(s) of features to disable.
		:type features: str or list of str


	.. py:method:: enable_only(features)
	
		Turn on only the specified model features ensuring all others are turned off.

		:param features: Name(s) of features to enable.
		:type features: str or list of str

	.. py:method:: get_enabled_features()
	
		Get a list of the currently enabled model features.

		:return: List of enabled features.
		:rtype: list of str

	.. py:method:: enable_featuresset_colour(Colour,features=None)
		
		Set the colour the model will appear.

		:param tuple Colour: Tuple of values between 0 and 1 specifying the R,G,B colour.
		:param features: The features to which to apply the colour. If none are specified, the colour is applied to the whole model at once.
		:type features: str or list of str

	.. py:method:: colour_by_material(on_off)

		Set the colour of the CAD model according to the material of individual parts, as specified in the CAD model definition. The default state for a given CAD model is defined as part of the CAD model definition.

		:param bool on_off: Whether or not to colour the CAD model by material.

	.. py:method:: flat_shading(on_off)
		
		Turn on or off flat shading. When enabled, no lighting effects are applied to the CAD model. The normal state is flat shading off.

		:param bool on_off: Enable or disable flat shading.