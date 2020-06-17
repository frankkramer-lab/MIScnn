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
# Internal libraries/scripts
from miscnn.data_loading.interfaces.abstract_io import Abstract_IO

#-----------------------------------------------------#
#               Dictionary I/O Interface              #
#-----------------------------------------------------#
""" Data I/O Interface for python dictionaries. This interface uses the basic-python
    dictionary to load and save data. Therefore the complete data management happens
    in the memory. Therefore, for common data set sizes this is NOT recommended!

    It is advised to use already provided I/O interfaces of this package or to implement
    a custom I/O interface for perfect usability.

    Dictionary structure:
        Key: sample_index
        Value: Tuple containing:    (0) image as numpy array
                                    (1) optional segmentation as numpy array
                                    (2) optional prediction as numpy array
                                    (3) optional details
"""
class Dictionary_interface(Abstract_IO):
    # Class variable initialization
    def __init__(self, dictionary, channels=1, classes=2, three_dim=True):
        self.dictionary = dictionary
        self.channels = channels
        self.classes = classes
        self.three_dim = three_dim

    #---------------------------------------------#
    #                  initialize                 #
    #---------------------------------------------#
    # Initialize the interface and return number of samples
    def initialize(self, input_path):
        # Return sample list
        return list(self.dictionary.keys())

    #---------------------------------------------#
    #                  load_image                 #
    #---------------------------------------------#
    # Read a image from the dictionary
    def load_image(self, index):
        # Return image
        return self.dictionary[index][0]

    #---------------------------------------------#
    #              load_segmentation              #
    #---------------------------------------------#
    # Read a segmentation from the dictionary
    def load_segmentation(self, index):
        # Return segmentation
        return self.dictionary[index][1]

    #---------------------------------------------#
    #               load_prediction               #
    #---------------------------------------------#
    # Read a prediction from the dictionary
    def load_prediction(self, index, output_path):
        # Return prediction
        return self.dictionary[index][2]

    #---------------------------------------------#
    #                 load_details                #
    #---------------------------------------------#
    # Parse additional information
    def load_details(self, index):
        # Return detail dictionary
        if len(self.dictionary[index]) >= 4:
            return self.dictionary[index][3]
        else:
            return None

    #---------------------------------------------#
    #               save_prediction               #
    #---------------------------------------------#
    # Write a segmentation prediction into the dictionary
    def save_prediction(self, pred, index, output_path):
        # Check if a prediction is already present -> overwrite
        if len(self.dictionary[index]) >= 3:
            self.dictionary[index][2]
        # If not, add the prediction to the sample tuple
        elif len(self.dictionary[index]) == 1:
            self.dictionary[index] = self.dictionary[index] + (None, pred,)
        elif len(self.dictionary[index]) == 2:
            self.dictionary[index] = self.dictionary[index] + (pred,)
