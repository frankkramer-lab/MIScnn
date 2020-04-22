#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2020 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
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
#              Subfunction class: Resize              #
#-----------------------------------------------------#
""" A Resize Subfunction class which resizes an images according to a desired shape.

Methods:
    __init__                Object creation function
    preprocessing:          Resize imaging data to the desired shape
    postprocessing:         Resize a prediction to the desired shape
"""
class Resize(Abstract_Subfunction):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, new_shape=(128,128,128)):
        self.new_shape = new_shape
        self.original_shape = None

    #---------------------------------------------#
    #                Preprocessing                #
    #---------------------------------------------#
    def preprocessing(self, sample, training=True):
        # Access data
        img_data = sample.img_data
        seg_data = sample.seg_data
        # Cache current spacing for later postprocessing
        if not training : self.original_shape = (1,) + img_data.shape[0:-1]
        # Transform data from channel-last to channel-first structure
        img_data = np.moveaxis(img_data, -1, 0)
        if training : seg_data = np.moveaxis(seg_data, -1, 0)
        # Resize imaging data
        img_data, seg_data = augment_resize(img_data, seg_data, self.new_shape,
                                            order=3, order_seg=1, cval_seg=0)
        # Transform data from channel-first back to channel-last structure
        img_data = np.moveaxis(img_data, 0, -1)
        if training : seg_data = np.moveaxis(seg_data, 0, -1)
        # Save resized imaging data to sample
        sample.img_data = img_data
        sample.seg_data = seg_data

    #---------------------------------------------#
    #               Postprocessing                #
    #---------------------------------------------#
    def postprocessing(self, prediction):
        # Access original shape of the last sample and reset it
        original_shape = self.original_shape
        self.original_shape = None
        # Transform original shape to one-channel array
        prediction = np.reshape(prediction, prediction.shape + (1,))
        # Transform prediction from channel-last to channel-first structure
        prediction = np.moveaxis(prediction, -1, 0)
        # Resize imaging data
        prediction = resize_segmentation(prediction, original_shape, order=1,
                                         cval=0)
        # Transform data from channel-first back to channel-last structure
        prediction = np.moveaxis(prediction, 0, -1)
        # Transform one-channel array back to original shape
        prediction = np.reshape(prediction, original_shape[1:])
        # Return postprocessed prediction
        return prediction
