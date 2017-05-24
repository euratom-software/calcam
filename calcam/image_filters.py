'''
* Copyright 2015-2017 European Atomic Energy Community (EURATOM)
*
* Licensed under the EUPL, Version 1.1 or - as soon they
  will be approved by the European Commission - subsequent
  versions of the EUPL (the "Licence");
* You may not use this work except in compliance with the
  Licence.
* You may obtain a copy of the Licence at:
*
* https://joinup.ec.europa.eu/software/page/eupl
*
* Unless required by applicable law or agreed to in
  writing, software distributed under the Licence is
  distributed on an "AS IS" basis,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
  express or implied.
* See the Licence for the specific language governing
  permissions and limitations under the Licence.
'''


'''
Image filter classes for Calcam.

Used to manipulate images in calcam. 
Might be expanded / become user-expandable some day.
'''

import cv2

class hist_eq():

    def __init__(self):
        pass

    def __call__(self,image):
        im_out = image.copy()
        hist_equaliser = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        if len(image.shape) == 2:
            im_out = hist_equaliser.apply(im_out.astype('uint8'))
        elif len(image.shape) > 2:
            for channel in range(3):
                im_out[:,:,channel] = hist_equaliser.apply(im_out.astype('uint8')[:,:,channel])
        return im_out