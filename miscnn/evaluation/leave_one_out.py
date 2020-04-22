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
import numpy as np
# Internal libraries/scripts
from miscnn.data_loading.data_io import create_directories
from miscnn.evaluation.detailed_validation import detailed_validation

#-----------------------------------------------------#
#              Leave-One-Out Validation               #
#-----------------------------------------------------#
""" Function for an automatic Leave-One-Out Validation of the Neural Network model by
    running the whole pipeline once by training on the complete data set except one sample
    and then predict the segmentation of the last remaining sample.

Args:
    sample_list (list of indices):          A list of sample indicies which will be used for validation.
    model (Neural Network model):           Instance of a Neural Network model class instance.
    epochs (integer):                       Number of epochs. A single epoch is defined as one iteration through the complete data set.
    iterations (integer):                   Number of iterations (batches) in a single epoch.
    evaluation_path (string):               Path to the evaluation data directory. This directory will be created and
                                            used for storing all kinds of evaluation results during the validation processes.
"""
def leave_one_out(sample_list, model, epochs=20, iterations=None, callbacks=[],
                  evaluation_path="evaluation"):
    # Choose a random sample
    loo = sample_list.pop(np.random.choice(len(sample_list)))
    # Reset Neural Network model weights
    model.reset_weights()
    # Train the model with the remaining samples
    model.train(sample_list, epochs=epochs, iterations=iterations,
                callbacks=callbacks)
    # Initialize evaluation directory
    create_directories(evaluation_path)
    # Make a detailed validation on the LOO sample
    detailed_validation([loo], model, evaluation_path)
