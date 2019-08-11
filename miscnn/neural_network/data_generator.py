#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
import keras
import numpy as np
#Internal libraries/scripts
from miscnn.data_io import batch_load

#-----------------------------------------------------#
#              MRI Data Generator (Keras)             #
#-----------------------------------------------------#
# MRI Data Generator for training and predicting (WITH-/OUT segmentation)
## Returns a batch containing multiple patches for each call
class DataGenerator(keras.utils.Sequence):
    # Class Initialization
    def __init__(self, batchPointer, model_path, training=False, shuffle=False):
        # Create a working environment from the handed over variables
        self.batchPointer = batchPointer
        self.model_path = model_path
        self.training = training
        self.shuffle = shuffle

    # Return the next batch for associated index
    def __getitem__(self, idx):
        # Load next volume batch
        batch_vol = batch_load(self.batchPointer[idx],
                               self.model_path,
                               vol=True)
        # IF batch is for training -> return next vol & seg batch
        if self.training:
            # Load next segmentation batch
            batch_seg = batch_load(self.batchPointer[idx],
                                   self.model_path,
                                   vol=False)
            # Return volume and segmentation batch
            return batch_vol, batch_seg
        # IF batch is for predicting -> return next vol batch
        else:
            # Return volume batch
            return batch_vol

    # Return the number of batches for one epoch
    def __len__(self):
        return len(self.batchPointer)

    # At every epoch end: Shuffle batchPointer list
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.batchPointer)
