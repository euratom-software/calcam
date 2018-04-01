import os
import json
import glob
from .io import ZipSaveFile


class CalcamConfig():

	def __init__(self,cfg_file= os.path.expanduser('~/.calcam_config'),allow_create=True):

		self.filename = cfg_file
		
		try:
			self.load()
		except IOError:
			if not allow_create:
				raise

			self.default_load_paths = {'calibrations':os.path.expanduser('~')}
			self.default_save_paths = {'calibrations':os.path.expanduser('~/Desktop')}
			self.cad_def_paths = [os.path.expanduser('~')]

			self.save()

	def load(self):

		with open(self.filename,'r') as f:
			load_dict = json.load(f)

		self.default_load_paths = 	load_dict['default_load_paths']
		self.default_save_paths = load_dict['default_save_paths']
		self.cad_def_paths = load_dict['cad_def_paths']


	def save(self):

		save_dict = {
						'default_load_paths' : self.default_load_paths,
						'default_save_paths' : self.default_save_paths,
						'cad_def_paths'		 : self.cad_def_paths
					}

		with open(self.filename,'w') as f:
			json.dump(save_dict,f,indent=4)



	def get_cadmodels(self):

		cadmodels = {}

		for path in self.cad_def_paths:
			filelist = glob.glob(os.path.join(path,'*.ccm'))
			filelist = filelist + glob.glob(os.path.join(path,'*.ccmm'))

			for fname in filelist:

				try:
					with ZipSaveFile(fname,'r') as f:
						with f.open_file('model.json','r') as j: 
							caddef = json.load(j)
				except Exception as e:
					raise Exception('Error loading CAD definition {:s}:{:s}'.format(fname,e))

				cadmodels[caddef['machine_name']] = [fname,[str(x) for x in caddef['features'].keys()],caddef['default_variant']]

		return cadmodels

