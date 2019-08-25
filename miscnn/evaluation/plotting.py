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

#-----------------------------------------------------#
#      Create evaluation figures of a validation      #
#-----------------------------------------------------#
""" Function for an automatic k-fold Cross-Validation of the Neural Network model by
    running the whole pipeline several times with different data set combinations.

Args:
    sample_list (list of indices):          A list of sample indicies which will be used for validation.
    model (Neural Network model):           Instance of a Neural Network model class instance.
    k_fold (integer):                       The number of k-folds for the Cross-Validationself. By default, a
                                            3-fold Cross-Validation is performed.
    evaluation_path (string):               Path to the evaluation data directory. This directory will be created and
                                            used for storing all kinds of evaluation results during the validation processes.
    draw_figures (boolean):                 Option if evaluation figures should be automatically plotted in the evaluation
                                            directory.
    detailed_evaluation (boolean):          Option if a detailed evaluation (additional prediction) should be performed.
    callbacks (list of Callback classes):   A list of Callback classes for custom evaluation.
"""
def plot_validation(history, metrics, evaluation_directory):
    pass
