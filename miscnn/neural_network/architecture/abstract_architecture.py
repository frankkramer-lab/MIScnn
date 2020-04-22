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
#     Abstract Interface for an Architecture class    #
#-----------------------------------------------------#
""" An abstract base class for a Architecture class.

Methods:
    __init__                Object creation function
    create_model_2D:        Creating a 2D Keras model
    create_model_3D:        Creating a 3D Keras model
"""
class Abstract_Architecture(ABC):
    #---------------------------------------------#
    #                   __init__                  #
    #---------------------------------------------#
    """ Functions which will be called during the Architecture object creation.
        This function can be used to pass variables and options in the Architecture instance.
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
    #               Create 2D Model               #
    #---------------------------------------------#
    """ Create the 2D version of a deep learning or convolutional neural network model.
        This function will be called inside the pipeline and have to return a functional
        Keras model for 2D images. The model itself should be created here or in a subfunction
        called by this function.
        It is possible to pass configurations through the initialization function of this class.

        Parameter:
            input_shape (Tuple):        Input shape of the image data for the first model layer
            n_labels (Integer):         Number of classes/labels of the segmentation (by default binary problem)
        Return:
            model (Keras model):        A Keras model
    """
    @abstractmethod
    def create_model_2D(self, input_shape, n_labels=2):
        pass
    #---------------------------------------------#
    #               Create 3D Model               #
    #---------------------------------------------#
    """ Create the 3D version of a deep learning or convolutional neural network model.
        This function will be called inside the pipeline and have to return a functional
        Keras model for 3D images. The model itself should be created here or in a subfunction
        called by this function.
        It is possible to pass configurations through the initialization function of this class.

        Parameter:
            input_shape (Tuple):        Input shape of the image data for the first model layer
            n_labels (Integer):         Number of classes/labels of the segmentation (by default binary problem)
        Return:
            model (Keras model):        A Keras model
    """
    @abstractmethod
    def create_model_3D(self, input_shape, n_labels=2):
        pass
