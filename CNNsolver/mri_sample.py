#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
import numpy
from keras.utils import to_categorical

#-----------------------------------------------------#
#           Magnetic Resonance Image - class          #
#-----------------------------------------------------#
class MRI:
    # Initialize class variable
    vol_data = None
    seg_data = None
    size = None

    # Create a MRI Sample object
    def __init__(self, volume):
        # Add and preprocess volume data
        vol_data = volume.get_data()
        self.vol_data = numpy.reshape(vol_data, vol_data.shape + (1,))
        # Identify and store number of MRI slices
        self.size = self.vol_data.shape[0]

    # Add and preprocess segmentation annotation
    def add_segmentation(self, segmentation):
        seg_data = segmentation.get_data()
        self.seg_data = numpy.reshape(seg_data, seg_data.shape + (1,))

    #-----------------------------------#
    #     MRI Data Generator (Keras)    #
    #-----------------------------------#
    ## Returns 2D slices of the MRI for each call
    # MRI Data Generator for training (WITH segmentation)
    def generator_train(self, batch_size, steps):
        while True:
            for slice in range(0, steps):
                # Calculate window/batch
                window_start = slice * batch_size
                window_end = slice * batch_size + batch_size
                if window_end >= self.size:
                    window_end = self.size
                window_length = window_end - window_start
                # Cut window/batch out of MRI slices
                vol = numpy.reshape(self.vol_data[window_start:window_end],
                                (window_length,) + self.vol_data[slice].shape)
                seg = numpy.reshape(self.seg_data[window_start:window_end],
                                (window_length,) + self.seg_data[slice].shape)
                # Transform digit classes into categorical
                seg_categorical = to_categorical(seg, num_classes=3)
                # Return batch
                yield(vol, seg_categorical)

    # MRI Data Generator for predicting (WITHOUT segmentation)
    def generator_predict(self, batch_size, steps):
        while True:
            for slice in range(0, steps):
                # Calculate window/batch
                window_start = slice * batch_size
                window_end = slice * batch_size + batch_size
                if window_end >= self.size:
                    window_end = self.size
                window_length = window_end - window_start
                # Cut window/batch out of MRI slices
                vol = numpy.reshape(self.vol_data[window_start:window_end],
                                (window_length,) + self.vol_data[slice].shape)
                # Return batch
                yield(vol)
