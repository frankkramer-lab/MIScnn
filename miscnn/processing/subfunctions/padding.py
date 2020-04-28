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
from batchgenerators.augmentations.utils import pad_nd_image
import numpy as np
# Internal libraries/scripts
from miscnn.processing.subfunctions.abstract_subfunction import Abstract_Subfunction

#-----------------------------------------------------#
#              Subfunction class: Padding             #
#-----------------------------------------------------#
""" A Padding Subfunction class which pads an images if required to a provided size.
    An image will only be padded, if its shape is smaller than the minimum size.

Args:
    min_size (tuple of integers):           Minimum shape of image. Every axis under this minimum size will be padded.
    pad_mode (string):                      Mode for padding. See in NumPy pad(array, mode="constant") documentation.
    pad_value_img (integer):                Value which will be used in padding mode "constant".
    pad_value_seg (integer):                Value which will be used in padding mode "constant".
    shape_must_be_divisible_by (integer):   Ensure that new shape is divisibly by provided number.

Methods:
    __init__                Object creation function
    preprocessing:          Padding to desired size of the imaging data
    postprocessing:         Cropping back to original size of the imaging data
"""
class Padding(Abstract_Subfunction):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, min_size, pad_mode="constant",
                 pad_value_img=0, pad_value_seg=0,
                 shape_must_be_divisible_by=None):
        self.min_size = min_size
        self.pad_mode = pad_mode
        self.pad_value_img = pad_value_img
        self.pad_value_seg = pad_value_seg
        self.shape_must_be_divisible_by = shape_must_be_divisible_by
        self.original_coords = None

    #---------------------------------------------#
    #                Preprocessing                #
    #---------------------------------------------#
    def preprocessing(self, sample, training=True):
        # Access data
        img_data = sample.img_data
        seg_data = sample.seg_data
        # Transform data from channel-last to channel-first structure
        img_data = np.moveaxis(img_data, -1, 0)
        if training : seg_data = np.moveaxis(seg_data, -1, 0)
        # Pad imaging data
        img_data, crop_coords = pad_nd_image(img_data, self.min_size,
                    mode=self.pad_mode,
                    kwargs={"constant_values": self.pad_value_img},
                    return_slicer=True,
                    shape_must_be_divisible_by=self.shape_must_be_divisible_by)
        if training:
            seg_data = pad_nd_image(seg_data, self.min_size,
                    mode=self.pad_mode,
                    kwargs={"constant_values": self.pad_value_seg},
                    return_slicer=False,
                    shape_must_be_divisible_by=self.shape_must_be_divisible_by)
        # Cache current crop coordinates for later postprocessing
        if not training : self.original_coords = crop_coords
        # Transform data from channel-first back to channel-last structure
        img_data = np.moveaxis(img_data, 0, -1)
        if training : seg_data = np.moveaxis(seg_data, 0, -1)
        # Save resampled imaging data to sample
        sample.img_data = img_data
        sample.seg_data = seg_data

    #---------------------------------------------#
    #               Postprocessing                #
    #---------------------------------------------#
    def postprocessing(self, prediction):
        # Access original coordinates of the last sample and reset it
        original_coords = self.original_coords
        self.original_coords = None
        # Transform original shape to one-channel array for cropping
        prediction = np.reshape(prediction, prediction.shape + (1,))
        # Transform prediction from channel-last to channel-first structure
        prediction = np.moveaxis(prediction, -1, 0)
        # Crop prediction data according to original coordinates
        prediction = prediction[tuple(original_coords)]
        # Transform data from channel-first back to channel-last structure
        prediction = np.moveaxis(prediction, 0, -1)
        # Transform one-channel array back to original shape
        prediction = np.reshape(prediction, prediction.shape[:-1])
        # Return postprocessed prediction
        return prediction
