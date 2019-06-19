#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
import numpy
import math

#-----------------------------------------------------#
#           Magnetic Resonance Image - class          #
#-----------------------------------------------------#
class MRI:
    # Initialize class variable
    vol_data = None
    seg_data = None
    batches_vol = None
    batches_seg = None

    # Create a MRI Sample object
    def __init__(self, volume):
        # Add and preprocess volume data
        vol_data = volume.get_data()
        self.vol_data = numpy.reshape(vol_data, vol_data.shape + (1,))

    # Add and preprocess segmentation annotation
    def add_segmentation(self, segmentation):
        seg_data = segmentation.get_data()
        self.seg_data = numpy.reshape(seg_data, seg_data.shape + (1,))

    # Create patches from a 3D matrix for later batching
    def add_batches(self, batches, vol):
        # Save volume batches
        if vol:
            self.batches_vol = batches
        # Save segmentation batches
        else:
            self.batches_seg = batches
