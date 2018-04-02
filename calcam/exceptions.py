# Custom exceptions used in calcam.
# They don't really do anything except
# allow me to keep track of 

class NoCADModels(Exception):
    pass

class MeshFileMissing(IOError):
    pass

class UserCodeException(Exception):
	pass