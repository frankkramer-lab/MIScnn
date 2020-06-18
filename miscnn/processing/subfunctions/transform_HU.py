#==============================================================================#
#  Author:       Michael Lempart                                               #
#  Copyright:    2020 Department of Radiation Physics, Lund, Sweden            #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from batchgenerators.augmentations.spatial_transformations import augment_resize
from batchgenerators.augmentations.utils import resize_segmentation
import numpy as np
# Internal libraries/scripts
from miscnn.processing.subfunctions.abstract_subfunction import Abstract_Subfunction

#-----------------------------------------------------#
#            Subfunction class: TransformHU            #
#-----------------------------------------------------#
""" A function to scale CT raw data according to HU units .

Methods:
    __init__                Object creation function
    preprocessing:          Transforms CT raw data to HU .
    normalize_HU:           Normalizes HU values in a range between 0-1.
    postprocessing:         no postprocessing needed.
"""
class TransformHU(Abstract_Subfunction):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, normalize = True, slope = 1, intercept = -1024,
                 clipScan_value = -2000, minmaxBound = (-1000, 400)):
        """
            Args:
                    normalize (bool):       Normalizes HU units between 0-1 if True.
                    slope (float):          slope value derived from DICOM files.
                    intercept (float):      intercept value derived from DICOM files.
                    ClipScan_value (int):   sets scan values at clipping value to 0 (used for out of scan values).
                    minmaxBound (tuple):    Normalization boundaries.
        """
        self.normalize = normalize
        self.slope = slope
        self.intercept = intercept
        self.clipScan_value = clipScan_value
        self.BOUND_MIN = minmaxBound[0]
        self.BOUND_MAX = minmaxBound[1]

    #---------------------------------------------#
    #                Preprocessing                #
    #---------------------------------------------#
    def preprocessing(self, sample, training=True):
        # Access data
        img_data = sample.img_data

        # Identify current spacing
        try :
            slope = sample.details["slope"]
        except:
            print("'slope' is not initialized in sample details!")
            print("The default value for 'slope' is used.")
            slope = self.slope
        try:
            intercept = sample.details["intercept"]
        except:
            print("'intercept' is not initialized in sample details!")
            print("The default value for 'intercept' is used.")
            intercept = self.intercept

        # Set out of scan values to 0
        if self.clipScan_value is not None:
            img_data[img_data == -2000] = 0

        #Scale image values according to slope and intercept
        img_data = slope * img_data
        img_data += intercept

        #Normalize HU values between 0-1
        if self.normalize:
            img_data = self.normalize_HU(img_data)

        #Update sample values
        sample.img_data = img_data


    def normalize_HU(self, image):
        """     Normalizes an image given in HU units between 0-1
                    Inputs:
                        image (array):              Images in the HU range
                    Returns:
                        img_normalized ( array):    Normalized images
        """
        img_normalized = (image - self.BOUND_MIN) / (self.BOUND_MAX - self.BOUND_MIN)
        img_normalized[img_normalized>1] = 1.
        img_normalized[img_normalized<0] = 0.

        return img_normalized

    #---------------------------------------------#
    #               Postprocessing                #
    #---------------------------------------------#
    def postprocessing(self, prediction):
        return prediction
