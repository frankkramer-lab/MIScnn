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
import numpy as np
# Internal libraries/scripts
from miscnn.data_loading.data_io import create_directories, backup_history
from miscnn.utils.plotting import plot_validation
from miscnn.evaluation.detailed_validation import detailed_validation

#-----------------------------------------------------#
#               k-fold Cross-Validation               #
#-----------------------------------------------------#
""" Function for an automatic k-fold Cross-Validation of the Neural Network model by
    running the whole pipeline several times with different data set combinations.

Args:
    sample_list (list of indices):          A list of sample indicies which will be used for validation.
    model (Neural Network model):           Instance of a Neural Network model class instance.
    k_fold (integer):                       The number of k-folds for the Cross-Validationself. By default, a
                                            3-fold Cross-Validation is performed.                        
    epochs (integer):                       Number of epochs. A single epoch is defined as one iteration through the complete data set.
    iterations (integer):                   Number of iterations (batches) in a single epoch.
    evaluation_path (string):               Path to the evaluation data directory. This directory will be created and
                                            used for storing all kinds of evaluation results during the validation processes.
    draw_figures (boolean):                 Option if evaluation figures should be automatically plotted in the evaluation
                                            directory.
    run_detailed_evaluation (boolean):      Option if a detailed evaluation (additional prediction) should be performed.
    callbacks (list of Callback classes):   A list of Callback classes for custom evaluation.
    direct_output (boolean):                Option, if computed evaluations will be output as the return of this function or
                                            if the evaluations will be saved on disk in the evaluation directory.
"""
def cross_validation(sample_list, model, k_fold=3, epochs=20,
                     iterations=None, evaluation_path="evaluation",
                     draw_figures=True, run_detailed_evaluation=True,
                     callbacks=[], direct_output=False):
    # Initialize result cache
    if direct_output : validation_results = []
    # Randomly permute the sample list
    samples_permuted = np.random.permutation(sample_list)
    # Split sample list into folds
    folds = np.array_split(samples_permuted, k_fold)
    fold_indices = list(range(len(folds)))
    # Start cross-validation
    for i in fold_indices:
        # Reset Neural Network model weights
        model.reset_weights()
        # Subset training and validation data set
        training = np.concatenate([folds[x] for x in fold_indices if x!=i],
                                  axis=0)
        validation = folds[i]
        # Initialize evaluation subdirectory for current fold
        subdir = create_directories(evaluation_path, "fold_" + str(i))
        # Run training & validation
        history = model.evaluate(training, validation, epochs=epochs,
                                 iterations=iterations, callbacks=callbacks)
        # Backup current history dictionary
        if direct_output : validation_results.append(history.history)
        else : backup_history(history.history, subdir)
        # Draw plots for the training & validation
        if draw_figures:
            plot_validation(history.history, model.metrics, subdir)
        # Make a detailed validation of the current cv-fold
        if run_detailed_evaluation:
            detailed_validation(validation, model, subdir)
    # Return the validation results
    if direct_output : return validation_results
