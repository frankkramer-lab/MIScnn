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
from abc import ABC, abstractmethod

#-----------------------------------------------------#
#     Abstract Interface for the Subfunction class    #
#-----------------------------------------------------#
""" An abstract base class for a processing Subfcuntion class.

Methods:
    __init__                Object creation function
    preprocessing:          Transform the imaging data
    postprocessing:         Transform the predicted segmentation
"""
class Abstract_Subfunction(ABC):
    #---------------------------------------------#
    #                   __init__                  #
    #---------------------------------------------#
    """ Functions which will be called during the Subfunction object creation.
        This function can be used to pass variables and options in the Subfunction instance.
        The are no mandatory required parameters for the initialization.

        Parameter:
            None
        Return:
            None
    """
    @abstractmethod
    def __init__(self):
        pass
    #---------------------------------------------#
    #                preprocessing                #
    #---------------------------------------------#
    """ Transform the image according to the subfunction during preprocessing (training + prediction).
        This is an in-place transformation of the sample object, therefore nothing is returned.
        It is possible to pass configurations through the initialization function of this class.

        Parameter:
            sample (Sample class):      Sample class object containing the imaging data (sample.img_data)
                                        and optional segmentation data (sample.seg_data)
            training (boolean):         Boolean variable indicating, if segmentation data is present at the
                                        sample object.
                                        If training is true, segmentation data in the sample object is available,
                                        if training is false, sample.seg_data is None
        Return:
            None
    """
    @abstractmethod
    def preprocessing(self, sample, training=True):
        pass
    #---------------------------------------------#
    #                postprocessing               #
    #---------------------------------------------#
    """ Transform the prediction according to the subfunction during postprocessing (prediction).
        This is NOT an in-place transformation of the prediction, therefore it is REQUIRED to
        return the processed prediction array.
        It is possible to pass configurations through the initialization function of this class.

        Parameter:
            prediction (numpy array):   Numpy array of the predicted segmentation
        Return:
            prediction (numpy array):   Numpy array of processed predicted segmentation
    """
    @abstractmethod
    def postprocessing(self, prediction):
        return prediction
