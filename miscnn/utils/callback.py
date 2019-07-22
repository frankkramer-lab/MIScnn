#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
import keras
#Internal libraries
from miscnn.data_io import save_evaluation

#-----------------------------------------------------#
#                Keras Callback Class                 #
#-----------------------------------------------------#
class TrainingCallback(keras.callbacks.Callback):
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
