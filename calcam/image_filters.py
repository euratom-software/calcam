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

cv2_version = float('.'.join(cv2.__version__.split('.')[:2]))
cv2_micro_version = int(cv2.__version__.split('.')[2].split('-')[0])

class hist_eq():

    def __init__(self):
        if cv2_version < 2.4 or (cv2_version == 2.4 and cv2_micro_version < 6):
          raise Exception('Histogram equalisation requires OpenCV 2.4.6 or newer; you have {:s}'.format(cv2.__version__))

    def __call__(self,image):
        im_out = image.copy()
        hist_equaliser = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        if len(image.shape) == 2:
            im_out = hist_equaliser.apply(im_out.astype('uint8'))
        elif len(image.shape) > 2:
            for channel in range(3):
                im_out[:,:,channel] = hist_equaliser.apply(im_out.astype('uint8')[:,:,channel])
        return im_out