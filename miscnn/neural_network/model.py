#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2019 IT-Infrastructure for Translational Medical Research,    #
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
# Internal libraries/scripts
from miscnn.neural_network.metrics import dice_classwise, tversky_loss

#-----------------------------------------------------#
#            Neural Network (model) class             #
#-----------------------------------------------------#
# Class which represents the Neural Network and which run the whole pipeline
class Neural_Network:
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    """ Initialization function for creating a Neural Network (model) object.
    This class provides functionality for handling all model methods.
    This class runs the whole pipeline and uses a Preprocessor instance to obtain batches.

    With an initialized Neural Network model instance, it is possible to run training, prediction
    and evaluations.

    Args:
        preprocessor (Preprocessor):            Preprocessor class instance which provides the Neural Network with batches.
        architecture (Architecture):            Instance of a neural network model Architecture class instance.
                                                By default, a standard U-Net is used as Architecture.
        loss (Metric Function):                 The metric function which is used as loss for training.
                                                Any Metric Function defined in Keras, in miscnn.neural_network.metrics or any custom
                                                metric function, which follows the Keras metric guidelines, can be used.
        metrics (List of Metric Functions):     List of one or multiple Metric Functions, which will be shown during training.
                                                Any Metric Function defined in Keras, in miscnn.neural_network.metrics or any custom
                                                metric function, which follows the Keras metric guidelines, can be used.
        epochs (integer):                       Number of epochs. A single epoch is defined as one iteration through the complete data set.
        learning_rate (float):                  Learning rate in which weights of the neural network will be updated.
        batch_queue_size (integer):             The batch queue size is the number of previously prepared batches in the cache during runtime.
        gpu_number (integer):                   Number of GPUs, which will be used for training.
    """
    def __init__(self, preprocessor, architecture=Architecture(),
                 loss=tversky_loss, metrics=[dice_classwise],
                 epochs=20, learninig_rate=0.0001,
                 batch_queue_size=2, gpu_number=1):
        pass

    #---------------------------------------------#
    #                  Training                   #
    #---------------------------------------------#
    #
    def train():
        pass

    #---------------------------------------------#
    #                 Prediction                  #
    #---------------------------------------------#
    #
    def predict():
        pass

    #---------------------------------------------#
    #                 Evaluation                  #
    #---------------------------------------------#
    #
    def evaluate():
        pass
