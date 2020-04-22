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
#External libraries
from tensorflow.keras.callbacks import Callback
#Internal libraries
from miscnn.data_io import save_evaluation

#-----------------------------------------------------#
#                Keras Callback Class                 #
#-----------------------------------------------------#
class TrainingCallback(Callback):
    # Initialize variables
    current_epoch = None
    eval_path = None

    # Initialize Class
    def __init__(self, eval_path):
        self.eval_path = eval_path
        # Create evaluation tsv file
        save_evaluation(["epoch", "tversky_loss", "dice_coef",
                         "dice_classwise", "categorical_accuracy",
                         "categorical_crossentropy", "val.tversky_loss",
                         "val.dice_coef", "val.dice_classwise",
                         "val.categorical_accuracy",
                         "val.categorical_crossentropy"],
                        eval_path,
                        "validation.tsv",
                        start=True)
        # Create training tsv file
        save_evaluation(["epoch", "batch", "tversky_loss",
                         "dice_coef", "dice_classwise"],
                        eval_path,
                        "training.tsv",
                        start=True)

    # Update current epoch
    def on_epoch_begin(self, epoch, logs={}):
        self.current_epoch = epoch

    # Backup training and validation scores to the evaluation tsv
    def on_epoch_end(self, epoch, logs={}):
        data_point = [epoch, logs["loss"],
                      logs["dice_coefficient"], logs["dice_classwise"],
                      logs["categorical_accuracy"],
                      logs["categorical_crossentropy"],
                      logs["val_loss"], logs["val_dice_coefficient"],
                      logs["val_dice_classwise"],
                      logs["val_categorical_accuracy"],
                      logs["val_categorical_crossentropy"]]
        save_evaluation(data_point, self.eval_path, "validation.tsv")

    # Save the current training scores for each batch in the training tsv
    def on_batch_end(self, batch, logs={}):
        data_point = [self.current_epoch, batch, logs["loss"],
                      logs["dice_coefficient"], logs["dice_classwise"]]
        save_evaluation(data_point, self.eval_path, "training.tsv")
