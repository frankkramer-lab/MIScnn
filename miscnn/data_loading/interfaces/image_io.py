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
import os
from PIL import Image
import numpy as np
import re
# Internal libraries/scripts
from miscnn.data_loading.interfaces.abstract_io import Abstract_IO

#-----------------------------------------------------#
#               covidxscan I/O Interface              #
#-----------------------------------------------------#
""" Data I/O Interface for JPEG, PNG and other common 2D image files.
    Images are read by calling the imread function from the Pillow module.

Methods:
    __init__                Object creation function
    initialize:             Prepare the data set and create indices list
    load_image:             Load an image
    load_segmentation:      Load a segmentation
    load_prediction:        Load a prediction
    load_details:           Load optional information
    save_prediction:        Save a prediction to disk

Args:
    classes (int):          Number of classes of the segmentation
    img_type (string):      Type of imaging. Options: "grayscale", "rgb"
    img_format (string):    Imaging format: Popular formats: "png", "tif", "jpg"
    pattern (regex):        Pattern to filter samples
"""
class Image_interface(Abstract_IO):
    #---------------------------------------------#
    #                   __init__                  #
    #---------------------------------------------#
    def __init__(self, classes=2, img_type="grayscale", img_format="png",
                 pattern=None):
        self.classes = classes
        self.img_type = img_type
        self.img_format = img_format
        self.three_dim = False
        self.pattern = pattern
        if img_type == "grayscale" : self.channels = 1
        elif img_type == "rgb" : self.channels = 3

    #---------------------------------------------#
    #                  initialize                 #
    #---------------------------------------------#
    def initialize(self, input_path):
        # Resolve location where imaging data set should be located
        if not os.path.exists(input_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(str(input_path))
            )
        # Cache data directory
        self.data_directory = input_path
        # Identify samples
        sample_list = os.listdir(input_path)
        # IF pattern provided: Remove every file which does not match
        if self.pattern != None and isinstance(self.pattern, str):
            for i in reversed(range(0, len(sample_list))):
                if not re.fullmatch(self.pattern, sample_list[i]):
                    del sample_list[i]
        # Return sample list
        return sample_list

    #---------------------------------------------#
    #                  load_image                 #
    #---------------------------------------------#
    def load_image(self, index):
        # Make sure that the image file exists in the data set directory
        img_path = os.path.join(self.data_directory, index)
        if not os.path.exists(img_path):
            raise ValueError(
                "Sample could not be found \"{}\"".format(img_path)
            )
        # Load image from file
        img_raw = Image.open(os.path.join(img_path, "imaging" + "." + \
                                          self.img_format))
        # Convert image to rgb or grayscale
        if self.img_type == "grayscale":
            img_pil = img_raw.convert("LA")
        elif self.img_type == "rgb":
            img_pil = img_raw.convert("RGB")
        # Convert Pillow image to numpy matrix
        img = np.array(img_pil)
        # Keep only intensity for grayscale images
        if self.img_type =="grayscale" : img = img[:,:,0]
        # Return image
        return img

    #---------------------------------------------#
    #              load_segmentation              #
    #---------------------------------------------#
    def load_segmentation(self, index):
        # Make sure that the segmentation file exists in the data set directory
        seg_path = os.path.join(self.data_directory, index)
        if not os.path.exists(seg_path):
            raise ValueError(
                "Segmentation could not be found \"{}\"".format(seg_path)
            )
        # Load segmentation from file
        seg_raw = Image.open(os.path.join(seg_path, "segmentation" + "." + \
                                          self.img_format))
        # Convert segmentation from Pillow image to numpy matrix
        seg_pil = seg_raw.convert("LA")
        seg = np.array(seg_pil)
        # Keep only intensity and remove maximum intensitiy range
        seg_data = seg[:,:,0]
        # Return segmentation
        return seg_data

    #---------------------------------------------#
    #               load_prediction               #
    #---------------------------------------------#
    def load_prediction(self, index, output_path):
        # Resolve location where data should be living
        if not os.path.exists(output_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(str(output_path))
            )
        # Parse the provided index to the prediction file name
        pred_file = str(index) + "." + self.img_format
        pred_path = os.path.join(output_path, pred_file)
        # Make sure that prediction file exists under the prediction directory
        if not os.path.exists(pred_path):
            raise ValueError(
                "Prediction could not be found \"{}\"".format(pred_path)
            )
        # Load prediction from file
        pred_raw = Image.open(pred_path)
        # Convert segmentation from Pillow image to numpy matrix
        pred_pil = pred_raw.convert("LA")
        pred = np.array(pred_pil)
        # Keep only intensity and remove maximum intensitiy range
        pred_data = pred[:,:,0]
        # Return prediction
        return pred_data

    #---------------------------------------------#
    #                 load_details                #
    #---------------------------------------------#
    def load_details(self, i):
        pass
    #---------------------------------------------#
    #               save_prediction               #
    #---------------------------------------------#
    def save_prediction(self, pred, index, output_path):
        # Resolve location where data should be written
        if not os.path.exists(output_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(output_path)
            )

        # Transform numpy array to a Pillow image
        pred_pillow = Image.fromarray(pred.astype(np.uint8))
        # Save segmentation to disk
        pred_file = str(index) + "." + self.img_format
        pred_pillow.save(os.path.join(output_path, pred_file))
