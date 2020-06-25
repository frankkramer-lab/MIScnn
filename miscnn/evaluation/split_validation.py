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
import math
# Internal libraries/scripts
from miscnn.data_loading.data_io import create_directories, backup_history
from miscnn.utils.plotting import plot_validation
from miscnn.evaluation.detailed_validation import detailed_validation

#-----------------------------------------------------#
#             Percentage-Split Validation             #
#-----------------------------------------------------#
""" Function for an automatic Percentage-Split Validation of the Neural Network model by
    running the whole pipeline once with a test and train data set.

Args:
    sample_list (list of indices):          A list of sample indicies which will be used for validation.
    model (Neural Network model):           Instance of a Neural Network model class instance.
    percentage (float):                     Split percentage of how big the testing data set should be.
                                            By default, the percentage value is 0.2 -> 20% testing and 80% training
    epochs (integer):                       Number of epochs. A single epoch is defined as one iteration through the complete data set.
    iterations (integer):                   Number of iterations (batches) in a single epoch.
    evaluation_path (string):               Path to the evaluation data directory. This directory will be created and
                                            used for storing all kinds of evaluation results during the validation processes.
    draw_figures (boolean):                 Option if evaluation figures should be automatically plotted in the evaluation
                                            directory.
    run_detailed_evaluation (boolean):      Option if a detailed evaluation (additional prediction) should be performed.
    callbacks (list of Callback classes):   A list of Callback classes for custom evaluation.
    return_output (boolean):                Option, if computed evaluations will be output as the return of this function or
                                            if the evaluations will be saved on disk in the evaluation directory.
"""
def split_validation(sample_list, model, percentage=0.2, epochs=20,
                     iterations=None, evaluation_path="evaluation",
                     draw_figures=False, run_detailed_evaluation=False,
                     callbacks=[], return_output=False):
    # Calculate the number of samples in the validation set
    validation_size = int(math.ceil(float(len(sample_list) * percentage)))
    # Randomly pick samples until %-split percentage
    validation = []
    for i in range(validation_size):
        validation_sample = sample_list.pop(np.random.choice(len(sample_list)))
        validation.append(validation_sample)
    # Rename the remaining cases as training
    training = sample_list
    # Reset Neural Network model weights
    model.reset_weights()
    # Run training & validation
    history = model.evaluate(training, validation, epochs=epochs,
                             iterations=iterations, callbacks=callbacks)
    # Initialize evaluation directory
    create_directories(evaluation_path)
    # Draw plots for the training & validation
    if draw_figures:
        plot_validation(history.history, model.metrics, evaluation_path)
    # Make a detailed validation
    if run_detailed_evaluation:
        detailed_validation(validation, model, evaluation_path)
    # Return or backup the validation results
    if return_output : return history.history
    else : backup_history(history.history, evaluation_path)
