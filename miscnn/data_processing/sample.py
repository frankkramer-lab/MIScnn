#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
import numpy
import math

#-----------------------------------------------------#
#                 Image Sample - class                #
#-----------------------------------------------------#
# Object containing an image and the associated segmentation
class Sample:
    # Initialize class variable
    vol_data = None
    seg_data = None
    shape = None
    channels = None

    # Create a Sample object
    def __init__(self, volume, channels):
        # Add and preprocess volume data
        vol_data = volume.get_data()
        self.vol_data = numpy.reshape(vol_data, vol_data.shape + (channels,))
        self.channels = channels
        self.shape = self.vol_data.shape

    # Add and preprocess segmentation annotation
    def add_segmentation(self, segmentation):
        seg_data = segmentation.get_data()
        self.seg_data = numpy.reshape(seg_data, seg_data.shape + (1,))
