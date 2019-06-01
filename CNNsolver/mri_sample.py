#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
import numpy
import math
from keras.utils import to_categorical
#Internal libraries/scripts
from utils.matrix_operations import slice_3Dmatrix, concat_3Dmatrices

#-----------------------------------------------------#
#           Magnetic Resonance Image - class          #
#-----------------------------------------------------#
class MRI:
    # Initialize class variable
    vol_data = None
    seg_data = None
    pred_data = None
    patches_vol = None
    patches_seg = None
    window = None
    overlap = None

    # Create a MRI Sample object
    def __init__(self, volume):
        # Add and preprocess volume data
        vol_data = volume.get_data()
        self.vol_data = numpy.reshape(vol_data, vol_data.shape + (1,))

    # Add and preprocess segmentation annotation
    def add_segmentation(self, segmentation, truth=True):
        # Segmentation is the truth from training
        if truth:
            seg_data = segmentation.get_data()
            self.seg_data = numpy.reshape(seg_data, seg_data.shape + (1,))
        # Segmentation is a prediction from the model
        else:
            pred_data = segmentation
            self.pred_data = numpy.reshape(pred_data, pred_data.shape + (1,))

    # Create patches from a 3D matrix for later batching
    def create_patches(self, matrix, window, overlap):
        # Slice the volume into patches with a provided window size
        if matrix=="vol" and self.patches_vol == None:
            self.window = window
            self.overlap = overlap
            self.patches_vol = slice_3Dmatrix(self.vol_data, window, overlap)
        # Slice the segmentation into patches with a provided window size
        elif matrix=="seg" and self.patches_seg == None:
            self.window = window
            self.overlap = overlap
            self.patches_seg = slice_3Dmatrix(self.seg_data, window, overlap)

    #-----------------------------------#
    #     MRI Data Generator (Keras)    #
    #-----------------------------------#
    ## Returns 3D slices of the MRI for each call
    # MRI Data Generator for training and predicting (WITH-/OUT segmentation)
    def data_generator(self, batch_size, steps, training=False):
        # Start while loop (Keras specific requirement)
        while True:
            for i in range(0, steps):
                # Assign patches to the next batch
                start = i * batch_size
                end = start + batch_size
                if end > len(self.patches_vol):
                    end = len(self.patches_vol)
                # Concatenate volume patches into the batch
                batch_vol = concat_3Dmatrices(self.patches_vol[start:end])
                # IF batch is for training -> return next vol & seg batch
                if training:
                    # Concatenate segmentation patches into Batches
                    batch_seg = concat_3Dmatrices(self.patches_seg[start:end])
                    # Transform digit segmentation classes into categorical
                    batch_seg = to_categorical(batch_seg, num_classes=3)
                    # Return volume and segmentation batch
                    yield(batch_vol, batch_seg)
                # IF batch is for predicting -> return next vol batch
                else:
                    yield(batch_vol)
