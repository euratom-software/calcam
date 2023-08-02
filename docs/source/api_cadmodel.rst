==========
CAD Models
==========

For use with ray casting or rendering images, it is common to need to make use of scene CAD models when using the calcam API. This is done with the :class:`calcam.CADModel` class, documented below. For examples of usage, see the :doc:`api_examples` page.

.. autoclass:: calcam.CADModel
	:members: get_feature_list,set_features_enabled,get_enabled_features,enable_only,get_group_enable_state,intersect_with_line,set_colour,get_colour,reset_colour,set_wireframe, set_linewidth,get_linewidth,set_flat_shading, format_coord, get_extent,set_status_callback,get_status_callback,unload
