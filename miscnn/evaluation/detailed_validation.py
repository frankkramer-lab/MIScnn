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
from miscnn.utils.visualizer import visualize_evaluation
from miscnn.data_loading.data_io import backup_evaluation

#-----------------------------------------------------#
#                 Detailed Validation                 #
#-----------------------------------------------------#
""" Function for detailed validation of a validation sample data set. The segmentation
    of these samples will be predicted with an already fitted model and evaluated.

Args:
    model (Neural Network model):           Instance of an already fitted Neural Network model class instance.
    validation_samples (list of indices):   A list of sample indicies which will be used for validation.
    evaluation_path (string):               Path to the evaluation data directory. This directory will be created and
                                            used for storing all kinds of evaluation results during the validation processes.
                                            if the evaluations will be saved on disk in the evaluation directory.
"""
def detailed_validation(validation_samples, model, evaluation_path):
    # Initialize detailed validation scoring file
    classes = list(map(lambda x: "dice_class-" + str(x),
                       range(model.preprocessor.data_io.interface.classes)))
    header = ["sample_id"]
    header.extend(classes)
    backup_evaluation(header, evaluation_path, start=True)
    # Iterate over each sample
    for sample_index in validation_samples:
        # Predict the sample with the provided model
        model.predict([sample_index], return_output=False)
        # Load the sample
        sample = model.preprocessor.data_io.sample_loader(sample_index,
                                                          load_seg=True,
                                                          load_pred=True)
        # Access image, truth and predicted segmentation data
        img, seg, pred = sample.img_data, sample.seg_data, sample.pred_data
        # Calculate classwise dice score
        dice_scores = compute_dice(seg, pred, len(classes))
        # Save detailed validation scores to file
        scores = [sample_index]
        scores.extend(dice_scores)
        backup_evaluation(scores, evaluation_path, start=False)
        # Visualize the truth and prediction segmentation
        visualize_evaluation(sample_index, img, seg, pred, evaluation_path)

#-----------------------------------------------------#
#                     Subroutines                     #
#-----------------------------------------------------#
# Calculate class-wise dice similarity coefficient
def compute_dice(truth, pred, classes):
    dice_scores = []
    # Compute Dice for each class
    for i in range(classes):
        try:
            pd = np.equal(pred, i)
            gt = np.equal(truth, i)
            dice = 2*np.logical_and(pd, gt).sum()/(pd.sum() + gt.sum())
            dice_scores.append(dice)
        except ZeroDivisionError:
            dice_scores.append(0.0)
    # Return computed Dice scores
    return dice_scores
